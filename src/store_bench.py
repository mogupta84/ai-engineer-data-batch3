"""Day 2, notebook 2: the helpers that measure Azure AI Search and PostgreSQL with pgvector the same way.

Each store answers one question with the ids of its top 10. Everything is timed from the notebook, which runs in the
same Azure region as both stores, the way an application server would; each store's own time is read as well.
"""
from __future__ import annotations

import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def percentile(values, p):
    """The p-th percentile of a list of milliseconds: p95 means 95 of 100 searches were at least this fast."""
    values = sorted(values)
    return values[max(0, math.ceil(p / 100 * len(values)) - 1)]


class PgStore:
    """PostgreSQL with pgvector. One connection per thread, because a connection runs one query at a time."""

    def __init__(self, conninfo, table):
        self.conninfo, self.table = conninfo, table
        self.local = threading.local()
        self.settings = {"enable_seqscan": "off"}                # walk the HNSW graph, as on a large table

    def db(self):
        import psycopg
        from pgvector.psycopg import register_vector
        if getattr(self.local, "db", None) is None or self.local.applied != dict(self.settings):
            if getattr(self.local, "db", None) is None:
                # prepare_threshold=None: never reuse a saved query plan. A plan saved while the graph index was
                # allowed would keep using it after we switch to exact search, and time the wrong thing.
                self.local.db = psycopg.connect(self.conninfo, autocommit=True, prepare_threshold=None)
                register_vector(self.local.db)
            for name, value in self.settings.items():
                self.local.db.execute(f"SET {name} = {value}")
            self.local.applied = dict(self.settings)
        return self.local.db

    def set(self, **settings):
        """Change a search setting (such as hnsw_ef_search=40) for every connection from now on."""
        self.settings.update({k.replace("hnsw_", "hnsw."): v for k, v in settings.items()})

    def search(self, q, label=None):
        where, args = ("WHERE label = %s ", [label]) if label is not None else ("", [])
        rows = self.db().execute(f"SELECT id FROM {self.table} {where}ORDER BY embedding <=> %s LIMIT 10", args + [q])
        return [r[0] for r in rows]

    def inside_ms(self, q):
        """Time spent inside PostgreSQL for one search, from its own query plan (no network, no planning)."""
        plan = self.db().execute(f"EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) SELECT id FROM {self.table} "
                                 f"ORDER BY embedding <=> %s LIMIT 10", [q]).fetchone()[0]
        return plan[0]["Execution Time"]

    def check(self, n_expected):
        """Stop before timing anything if the table is not as the trainer built it."""
        db = self.db()
        count = db.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]
        graph = db.execute("SELECT indexname FROM pg_indexes WHERE tablename = %s AND indexdef ILIKE %s",
                           [self.table, "%USING hnsw%"]).fetchone()
        if count != n_expected or graph is None:
            raise RuntimeError(f"{self.table}: {count:,} rows, graph index {'found' if graph else 'MISSING'}. "
                               "Ask your trainer to rebuild it.")
        return count


class SearchStore:
    """Azure AI Search. Its own time comes back with every reply, in the elapsed-time header (whole ms)."""

    def __init__(self, endpoint, key, index):
        self.endpoint, self.key, self.index = endpoint, key, index
        self.local = threading.local()
        self.client = self._new_client()
        self.exhaustive = False
        self.own_ms = []

    def _new_client(self):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        return SearchClient(self.endpoint, self.index, AzureKeyCredential(self.key))

    def _client(self):
        """One client per thread, so many users at once do not queue for the same pool of connections."""
        if getattr(self.local, "client", None) is None:
            self.local.client = self._new_client()
        return self.local.client

    def search(self, q, label=None):
        from azure.search.documents.models import VectorizedQuery
        def keep(response):
            value = response.http_response.headers.get("elapsed-time")
            if value is not None:
                self.own_ms.append(float(value))
        found = self._client().search(search_text=None, top=10, select=["id"],
                                   filter=f"label eq {label}" if label is not None else None,
                                   vector_queries=[VectorizedQuery(vector=q.tolist(), k_nearest_neighbors=10,
                                                                   fields="vector", exhaustive=self.exhaustive)],
                                   raw_response_hook=keep)
        return [int(p["id"]) for p in found]

    def check(self, n_expected):
        count = self.client.get_document_count()
        if count != n_expected:
            raise RuntimeError(f"the search index holds {count:,} rows, not {n_expected:,}. Ask your trainer.")
        return count


def recall(truth_rows, got_rows, k=10):
    """Share of the true top k that came back, averaged over the questions."""
    return float(np.mean([len(set(map(int, t[:k])) & set(map(int, g[:k]))) / max(1, min(k, len(t)))
                          for t, g in zip(truth_rows, got_rows)]))


def one_at_a_time(store, questions):
    """Each question once, one after another. Returns (ids per question, list of ms)."""
    got, ms = [], []
    for q in questions:
        start = time.perf_counter()
        got.append(store.search(q))
        ms.append((time.perf_counter() - start) * 1000)
    return got, ms


def under_load(store, questions, users, total=400):
    """`users` searches running at the same time, `total` searches in all. Returns searches per second and the
    p50, p95 and p99 milliseconds, the way a load test reports them."""
    work = [questions[i % len(questions)] for i in range(total)]
    ms, lock = [], threading.Lock()

    def run(q):
        start = time.perf_counter()
        store.search(q)
        with lock:
            ms.append((time.perf_counter() - start) * 1000)

    ready = threading.Barrier(users, timeout=60)

    def warm(i):
        """Each worker opens its own connection and searches once before the clock starts. The barrier makes every
        one of the `users` workers do this, not just the first few. If one fails, all stop with its error."""
        try:
            store.search(questions[i % len(questions)])
        except Exception:
            ready.abort()
            raise
        ready.wait()

    with ThreadPoolExecutor(max_workers=users) as pool:
        list(pool.map(warm, range(users)))
        start = time.perf_counter()
        list(pool.map(run, work))
        seconds = time.perf_counter() - start
    return {"users": users, "searches a second": round(total / seconds, 1), "p50 ms": round(percentile(ms, 50), 1),
            "p95 ms": round(percentile(ms, 95), 1), "p99 ms": round(percentile(ms, 99), 1)}


def median(values):
    values = [v for v in values if v == v]
    return statistics.median(values) if values else float("nan")
