"""Day 2, code part 4: the two vector stores at 100,000 vectors, where speed and recall start to trade.

    python lab2_ingest_index/l2_scale.py            the sweep, exact search and the filter test, about 5 minutes
    python lab2_ingest_index/l2_scale.py --quick    fewer questions, about 3 minutes

At 21 pieces every store checks everything and is exact. At 100,000 they use a graph (HNSW) that visits only part of
the data, and ef is the dial: how many candidates the search keeps while it walks the graph. Your trainer loaded the
same 100,000 made-up vectors into both stores before class (trainer_build_scale.py); this script makes the same
vectors and questions on your laptop, finds the exact answers by checking every vector, and scores each store
against them. You only read; nothing is loaded. No model calls, no cost.
"""
import math
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from azure_common import search_client, setting
from scale_set import N, exact_top_k, make_benchmark_set, recall

INDEX, TABLE, SWEEP = "freshcart-scale-100k", "scale_100k", [10, 20, 40, 80, 160]
HEADER = f"{'Store and setting':32} {'recall@10':>9} {'median ms':>10} {'95% under ms':>13} {'inside the store ms':>20}"


def timed(ask, questions):
    """Each question once; returns (row numbers per question, median ms, 95th percentile ms)."""
    got, ms = [], []
    for q in questions:
        start = time.perf_counter()
        got.append(ask(q))
        ms.append((time.perf_counter() - start) * 1000)
    ms.sort()
    return got, statistics.median(ms), ms[math.ceil(0.95 * len(ms)) - 1]


def connect_postgres(bench):
    """The class container, checked before any timing: the table complete, its graph index in use, the labels right.
    Returns None, with the reason printed, when it cannot be used; the AI Search part still runs."""
    import psycopg
    from pgvector.psycopg import register_vector
    try:
        db = psycopg.connect(setting("PG_CONNECTION"), autocommit=True)
    except psycopg.OperationalError as e:
        print(f"Cannot sign in to the class PostgreSQL server: {str(e).splitlines()[0][:110]}")
        print("Check PG_CONNECTION in .env. If it says timeout, your network may block port 5432: watch the trainer's run.")
        print("The AI Search rows still run.")
        return None
    register_vector(db)
    try:
        db.execute("SET enable_seqscan = off")                 # walk the HNSW graph, as on a large table
        count = db.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
        graph = db.execute("SELECT indexname FROM pg_indexes WHERE tablename = %s AND indexdef ILIKE %s",
                           [TABLE, "%USING hnsw%"]).fetchone()
        plan = " ".join(r[0] for r in db.execute(f"EXPLAIN SELECT id FROM {TABLE} ORDER BY embedding <=> %s LIMIT 10",
                                                 [bench.Q[0]]).fetchall())
        sample = db.execute(f"SELECT id, label FROM {TABLE} WHERE id IN (0, 4242, 99999)").fetchall()
    except psycopg.Error as e:
        print(f"The class PostgreSQL server answered, but: {str(e).splitlines()[0][:110]}")
        print("Ask your trainer; the AI Search rows still run.")
        return None
    graph_used = graph is not None and graph[0] in plan          # the HNSW index itself, not any other index
    labels_ok = len(sample) == 3 and all(int(bench.labels[i]) == lab for i, lab in sample)
    if count != N or not graph_used or not labels_ok:
        print(f"The pgvector table is not as the trainer built it (rows {count:,}, graph index "
              f"{'used' if graph_used else 'MISSING'}, labels {'right' if labels_ok else 'WRONG'}). "
              "Ask your trainer to rebuild it; the AI Search rows still run.")
        return None
    return db


def main():
    quick = "--quick" in sys.argv
    n, n_inside = (50, 15) if quick else (100, 30)
    print("Making the 100,000 vectors and the answer key on your laptop...")
    bench = make_benchmark_set()
    Q, truth = bench.Q[:n], bench.truth[:n]
    db = connect_postgres(bench)

    from azure.search.documents.models import VectorizedQuery
    search = search_client(INDEX)

    def pg(q, label=None):
        where = "WHERE label = %s " if label is not None else ""
        args = ([label] if label is not None else []) + [q]
        return [r[0] for r in db.execute(f"SELECT id FROM {TABLE} {where}ORDER BY embedding <=> %s LIMIT 10", args)]

    def inside_ms(q):
        """Time spent inside PostgreSQL for one search, without the trip over the network."""
        plan = db.execute(f"EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) SELECT id FROM {TABLE} "
                          f"ORDER BY embedding <=> %s LIMIT 10", [q]).fetchone()[0]
        return plan[0]["Execution Time"]

    def ai(q, exhaustive=False, label=None):
        found = search.search(search_text=None, top=10, select=["id"], filter=f"label eq {label}" if label is not None else None,
                              vector_queries=[VectorizedQuery(vector=q.tolist(), k_nearest_neighbors=10, fields="vector",
                                                              exhaustive=exhaustive)])
        return [int(p["id"]) for p in found]

    ai(Q[0])                                                    # warm up the connection
    print(f"\n1. The dial, {n} questions, one at a time\n{HEADER}")
    if db is not None:
        pg(Q[0])
        _, p50, p95 = timed(lambda q: db.execute("SELECT 1").fetchall(), Q)      # the trip alone, no search at all
        print(f"{'network round trip only':32} {'':>9} {p50:10.1f} {p95:13.1f} {'':>20}")
        for ef in SWEEP:
            db.execute(f"SET hnsw.ef_search = {ef}")
            got, p50, p95 = timed(pg, Q)
            inside = statistics.median(inside_ms(q) for q in Q[:n_inside])
            print(f"{'pgvector, ef ' + str(ef):32} {recall(truth, got):9.3f} {p50:10.1f} {p95:13.1f} {inside:20.2f}")
        db.execute("SET enable_seqscan = on")                   # exact: check every row, no graph, one process
        db.execute("SET enable_indexscan = off")
        db.execute("SET max_parallel_workers_per_gather = 0")
        got, p50, p95 = timed(pg, Q)
        inside = statistics.median(inside_ms(q) for q in Q[:n_inside])
        print(f"{'pgvector, exact (every row)':32} {recall(truth, got):9.3f} {p50:10.1f} {p95:13.1f} {inside:20.2f}")
        db.execute("RESET max_parallel_workers_per_gather")
        db.execute("SET enable_indexscan = on")
        db.execute("SET enable_seqscan = off")
    for name, exhaustive in (("AI Search, HNSW (ef 100)", False), ("AI Search, exhaustive (exact)", True)):
        got, p50, p95 = timed(lambda q: ai(q, exhaustive), Q)
        print(f"{name:32} {recall(truth, got):9.3f} {p50:10.1f} {p95:13.1f} {'not shown':>20}")

    print("\n2. A filter: only rows with one label (a tenth of the data), 30 questions")
    labels = [int(bench.labels[int(t[0])]) for t in truth[:30]]       # the label of each question's best match
    keys = [exact_top_k(bench.X, q, allowed=bench.labels == lab) for q, lab in zip(Q[:30], labels)]
    runs = [("AI Search, filter before search", lambda q, lab: ai(q, label=lab), None)]
    if db is not None:
        db.execute("SET hnsw.ef_search = 40")
        runs = [("pgvector, ef 40, plain", pg, "off"),
                ("pgvector, ef 40, iterative scan", pg, "relaxed_order")] + runs   # keep walking until 10 rows pass
    for name, ask, iterative in runs:
        if iterative is not None:
            db.execute(f"SET hnsw.iterative_scan = {iterative}")
        got = [ask(q, lab) for q, lab in zip(Q[:30], labels)]
        print(f"{name:32} rows back {np.mean([len(g) for g in got]):4.1f} of 10   recall@10 {recall(keys, got):.3f}")
    if db is not None:
        db.execute("SET hnsw.iterative_scan = off")
    print("\nMedian ms is what your laptop waits: mostly the network trip, which varies by tens of ms from run to run.")
    print("Inside the store is the search itself; compare the pgvector settings on that column.")
    print("A filter applied after a graph walk keeps only the matches among the few candidates it found.")


if __name__ == "__main__":
    main()
