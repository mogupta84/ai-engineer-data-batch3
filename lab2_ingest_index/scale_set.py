"""The 100,000-vector benchmark set for the Day 2 scale test, the same one the earlier course labs used.

Made-up vectors grouped in 1,000 clusters, the way real embeddings are, plus questions that sit near stored vectors,
and the exact top 10 for each question (the answer key, found by checking every vector). The seed is fixed, so the
trainer's load script and every student's laptop make exactly the same numbers without downloading them.
"""
from dataclasses import dataclass

import numpy as np

N, DIM, N_QUESTIONS, N_GROUPS, N_LABELS, SEED = 100_000, 128, 200, 1_000, 10, 7


@dataclass
class BenchSet:
    X: np.ndarray            # the stored vectors, one row each, unit length
    Q: np.ndarray            # the questions, unit length
    labels: np.ndarray       # one label per stored vector, 0 to 9, for the filter test
    truth: np.ndarray        # exact top 10 per question: the answer key


# The two sets the course uses: the earlier labs' 100,000 x 128, and 20,000 x 1,536, the size of a real
# text-embedding-3-small vector. Each has its own index and table name.
SETS = {"128": {"n": 100_000, "dim": 128, "index": "freshcart-scale-100k", "table": "scale_100k"},
        "1536": {"n": 20_000, "dim": 1536, "index": "freshcart-scale-1536", "table": "scale_1536"}}


def make_benchmark_set(k=10, n=N, dim=DIM):
    rng = np.random.default_rng(SEED)
    centres = rng.normal(size=(N_GROUPS, dim)).astype("float32")
    X = centres[rng.integers(0, N_GROUPS, n)] + 0.35 * rng.normal(size=(n, dim)).astype("float32")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = X[rng.choice(n, N_QUESTIONS, replace=False)] + 0.05 * rng.normal(size=(N_QUESTIONS, dim)).astype("float32")
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    labels = rng.integers(0, N_LABELS, n)
    X, Q = X.astype("float32"), Q.astype("float32")
    truth = np.argsort(-(Q @ X.T), axis=1)[:, :k]
    return BenchSet(X=X, Q=Q, labels=labels, truth=truth)


def exact_top_k(X, q, k=10, allowed=None):
    """Exact search, optionally only over the rows where allowed is True: the answer key for a filtered search."""
    scores = X @ q
    if allowed is not None:
        scores = np.where(allowed, scores, -np.inf)
    return [int(i) for i in np.argsort(-scores)[:k] if np.isfinite(scores[i])]


def recall(truth_rows, got_rows, k=10):
    """Share of the true top k that came back, averaged over the questions."""
    return float(np.mean([len(set(map(int, t[:k])) & set(map(int, g[:k]))) / max(1, min(k, len(t)))
                          for t, g in zip(truth_rows, got_rows)]))
