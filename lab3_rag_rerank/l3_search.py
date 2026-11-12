"""Day 3, code part 1: six ways to search the same index, each scored on the 20 judged questions with nDCG@10.

    python lab3_rag_rerank/l3_search.py              stages 1 to 3: meaning, keyword, both together (hybrid)
    python lab3_rag_rerank/l3_search.py --rerank     also 4 to 6: the reranker, current documents only, then 50 candidates
    python lab3_rag_rerank/l3_search.py --show R01   the top 3 pieces of one question in every stage

In the portal you saw one question at a time. Here every stage answers the same 20 questions, and nDCG@10 (from 0
to 1; 1 means the best pieces came first) shows the lift each stage adds. The index is the trainer's shared
freshcart-chunks: 500-character pieces, each labelled current or not.

The semantic reranker is free for 1,000 searches a month per search service, shared by the whole class.
--rerank uses 60. Run it once, not in a loop.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from azure_common import golden, grade, mean, models, ndcg_at_10, question_vectors, search_client

FIELDS = ["id", "file", "content", "current"]


def stages(client):
    """Each stage is one way to ask the same index. v is the question as numbers, q is its words."""
    from azure.search.documents.models import VectorizedQuery
    meaning = lambda v, k=10: [VectorizedQuery(vector=v, k_nearest_neighbors=k, fields="vector")]
    out = {
        "1 meaning": lambda q, v: client.search(search_text=None, vector_queries=meaning(v), top=10, select=FIELDS),
        "2 keyword (BM25)": lambda q, v: client.search(search_text=q, top=10, select=FIELDS),
        "3 hybrid": lambda q, v: client.search(search_text=q, vector_queries=meaning(v), top=10, select=FIELDS),
    }
    if "--rerank" in sys.argv:
        rerank = {"query_type": "semantic", "semantic_configuration_name": "rerank"}
        out["4 hybrid + reranker"] = lambda q, v: client.search(search_text=q, vector_queries=meaning(v), top=10,
                                                                select=FIELDS, **rerank)
        out["5 + current only"] = lambda q, v: client.search(search_text=q, vector_queries=meaning(v), top=10,
                                                             select=FIELDS, filter="current eq true", **rerank)
        # The fix to try: give the reranker more to choose from. Meaning search hands over 50 candidates, not 10.
        out["6 = 5, 50 candidates"] = lambda q, v: client.search(search_text=q, vector_queries=meaning(v, 50), top=10,
                                                                 select=FIELDS, filter="current eq true", **rerank)
    return out


def main():
    client = search_client()
    every_piece = list(client.search(search_text="*", top=1000, select=FIELDS))       # for the best possible top 10
    questions = golden("retrieval_golden.json")["queries"]
    vectors = question_vectors(models(), [q["query"] for q in questions])
    i = sys.argv.index("--show") if "--show" in sys.argv else -1
    show = sys.argv[i + 1].upper() if 0 <= i < len(sys.argv) - 1 else None
    scores, tops = {}, {}
    for name, ask in stages(client).items():
        start, per_question = time.time(), {}
        for q, v in zip(questions, vectors):
            ranked = list(ask(q["query"], v))
            per_question[q["id"]] = ndcg_at_10(q["rules"], ranked, every_piece)
            tops[name, q["id"]] = ranked[:3]
            if show == q["id"]:
                print(f"\n{name}, {q['id']} {q['query']!r}")
                for p in ranked[:3]:
                    score = p.get("@search.reranker_score") or p["@search.score"]
                    print(f"   {score:7.3f}  {p['file']:36} current={p['current']!s:5} {p['content'][:60]!r}")
        scores[name] = per_question
        print(f"{name:22} nDCG@10 {mean(per_question.values()):.3f}   "
              f"{(time.time() - start) / len(questions) * 1000:4.0f} ms a question")
    if "4 hybrid + reranker" in scores:
        a, b = scores["3 hybrid"], scores["4 hybrid + reranker"]
        worse = [i for i in a if None not in (a[i], b[i]) and b[i] < a[i] - 0.01]
        print(f"\nThe reranker made {len(worse)} of 20 questions worse. Their top 3 before and after, with the grade a "
              "judge gave each piece (3 best, 0 not useful):")
        by_id = {q["id"]: q for q in questions}
        for i in worse:
            q = by_id[i]
            print(f"\n{i} {q['query']!r}: nDCG {a[i]:.2f} to {b[i]:.2f}, then {scores['5 + current only'][i]:.2f} "
                  f"with the filter and {scores['6 = 5, 50 candidates'][i]:.2f} with 50 candidates")
            for name in ("3 hybrid", "4 hybrid + reranker"):
                print(f"   {name}: " + ", ".join(f"{p['file'].split('_')[0]}... grade {grade(q['rules'], p)}"
                                                for p in tops[name, i]))
        print("\nAn average that goes up can still hide questions that went down. Read the pieces before you ship.")


main()
