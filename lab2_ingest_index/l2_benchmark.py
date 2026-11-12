"""Day 2, code part 3: the same pieces in two vector stores, Azure AI Search and pgvector, compared on what they
find (recall@10, nDCG@10) and how fast (milliseconds a search).

    python lab2_ingest_index/l2_benchmark.py          copy the pieces into your pgvector table, then compare
    python lab2_ingest_index/l2_benchmark.py --keep   compare again without copying

The pieces and their numbers come from the trainer's shared index, freshcart-chunks, so both stores hold exactly
the same thing and only the store differs. Your pgvector table is named after you on the class Postgres server.
Cost: the 20 judged questions are embedded once (about $0.0001); everything after is free.
"""
import pathlib
import statistics     # median and spread of the timings
import sys
import time           # time each search

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from azure_common import (golden, mean, models, ndcg_at_10, question_vectors, recall_at_10, search_client, setting,
                          student)

FIELDS = ["id", "file", "content"]
TABLE = "chunks_" + student().replace("-", "_")         # one table per student: STUDENT in .env must be YOUR account


def from_search():
    """Every piece in the shared index, with its 1,536 numbers."""
    return list(search_client().search(search_text="*", top=1000, select=FIELDS + ["vector"]))


def postgres():
    import psycopg
    from pgvector.psycopg import register_vector
    db = psycopg.connect(setting("PG_CONNECTION"), autocommit=True)
    if not db.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone():
        sys.exit("The vector extension is not enabled on this database. Ask your trainer to run: CREATE EXTENSION vector;")
    register_vector(db)
    # With only 21 rows PostgreSQL would read every row instead of using the HNSW index. Switching the full read off
    # makes it use the index, so the benchmark measures the same kind of search as AI Search.
    db.execute("SET enable_seqscan = off")
    return db


def load(db, every_piece):
    import numpy
    db.execute(f"DROP TABLE IF EXISTS {TABLE}")
    db.execute(f"CREATE TABLE {TABLE} (id text PRIMARY KEY, file text, content text, embedding vector(1536))")
    with db.cursor() as cur:
        cur.executemany(f"INSERT INTO {TABLE} VALUES (%s, %s, %s, %s)",
                        [(p["id"], p["file"], p["content"], numpy.array(p["vector"])) for p in every_piece])
    # HNSW: the same kind of graph index AI Search uses. Cosine distance, as the embedding model expects.
    db.execute(f"CREATE INDEX ON {TABLE} USING hnsw (embedding vector_cosine_ops)")
    print(f"pgvector: {len(every_piece)} pieces copied into {TABLE}, with an HNSW index")


def ask_search(client, vector):
    from azure.search.documents.models import VectorizedQuery
    return list(client.search(search_text=None, top=10, select=FIELDS,
                              vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=10, fields="vector")]))


def ask_postgres(db, vector):
    import numpy
    rows = db.execute(f"SELECT id, file, content FROM {TABLE} ORDER BY embedding <=> %s LIMIT 10",
                      (numpy.array(vector),)).fetchall()
    return [{"id": i, "file": f, "content": c} for i, f, c in rows]


def main():
    client, db = search_client(), postgres()
    every_piece = from_search()
    print(f"Azure AI Search: {len(every_piece)} pieces in the shared index")
    if "--keep" not in sys.argv:
        load(db, every_piece)
    questions = golden("retrieval_golden.json")["queries"]
    vectors = question_vectors(models(), [q["query"] for q in questions])
    stores = {"Azure AI Search": lambda v: ask_search(client, v), "pgvector": lambda v: ask_postgres(db, v)}
    for ask in stores.values():                                  # one search each to warm up the connections
        ask(vectors[0])
    results = {}
    print(f"\n{'Store':16} {'recall@10':>9} {'nDCG@10':>8} {'median ms':>10} {'slowest ms':>11}")
    for name, ask in stores.items():
        timings, recalls, ndcgs, tops = [], [], [], {}
        for q, v in zip(questions, vectors):
            start = time.perf_counter()
            ranked = ask(v)
            timings.append((time.perf_counter() - start) * 1000)
            recalls.append(recall_at_10(q["rules"], ranked, every_piece))
            ndcgs.append(ndcg_at_10(q["rules"], ranked, every_piece))
            tops[q["id"]] = [p["id"] for p in ranked]
        results[name] = tops
        print(f"{name:16} {mean(recalls):9.3f} {mean(ndcgs):8.3f} {statistics.median(timings):10.1f} {max(timings):11.1f}")
    same = mean([len(set(results['Azure AI Search'][i]) & set(results['pgvector'][i])) / 10 for i in results['pgvector']])
    print(f"\nThe two top-10 lists share {same:.0%} of their pieces on average.")
    print("Same numbers in, nearly the same pieces out: the store mostly decides speed, cost, filters and who runs it.")


main()
