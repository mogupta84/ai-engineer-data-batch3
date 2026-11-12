"""Day 3, code part 2: a RAG pipeline you can read end to end: search, rerank, answer with citations.

    python lab3_rag_rerank/l3_rag.py                         the default question
    python lab3_rag_rerank/l3_rag.py "your own question"     any question
    python lab3_rag_rerank/l3_rag.py --no-rerank "..."       the same without the reranker, to compare

It prints the pieces sent to the model, so you can judge the answer against what it was given.
Cost: one embedding, one reranked search and one short model answer, well under one cent.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from azure_common import models
from rag import answer

args = [a for a in sys.argv[1:] if not a.startswith("--")]
question = args[0] if args else "How many refund claims can a customer make in thirty days?"
r = answer(models(), question, rerank="--no-rerank" not in sys.argv)

print(f"Question: {question}\n\nPieces sent to the model:")
for p in r["retrieved"]:
    print(f"  [{p['id']}] {p['score']:6.3f}  {p['file']:36} {p['text'][:70]!r}")
print(f"\nAnswer:\n{r['answer']}\n")
print(f"Citations: {r['citations'] or 'none'}; not from the search: {r['unknown_citations'] or 'none'}")
if r["refused"]:
    print(f"Refused: {r['refusal_reason']}")
