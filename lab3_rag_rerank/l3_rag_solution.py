"""Day 3, code part 2, THE SOLUTION: l3_rag.py with GAP 4 filled. Try the gap yourself first.
Build the RAG pipeline step by step: retrieve, augment, generate, check.
You write one part of it: GAP 4, the augment step that puts the pieces the search found into the prompt.

    python lab3_rag_rerank/l3_rag_solution.py                         the default question
    python lab3_rag_rerank/l3_rag_solution.py "your own question"     any question
    python lab3_rag_rerank/l3_rag_solution.py --no-rerank "..."       the same without the reranker, to compare

It prints the pieces sent to the model, so you can judge the answer against what it was given.
Cost: one embedding, one reranked search and one short model answer, well under one cent.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from azure_common import chat, models
from rag import PROMPT, retrieve, verify


# ===================================================== YOUR CODE: one gap ========================================
def gap4_augment(pieces):
    """GAP 4. The augment step: turn the pieces the search found into the text the model reads.
    Each piece is a dict with "id", "file" and "text". The model must see every piece with its id in square brackets,
    because the prompt tells it to cite ids like [3f2a9c1d7e44], and the check in step 4 compares those ids with the
    pieces you sent. One block per piece, a blank line between blocks, for example:
        [3f2a9c1d7e44] (refund_policy_v3_2024.pdf)
        A customer may make up to three refund claims ...
    Return the whole text as one string."""
    return "\n\n".join(f"[{p['id']}] ({p['file']})\n{p['text']}" for p in pieces)
# ==================================================================================================================


args = [a for a in sys.argv[1:] if not a.startswith("--")]
question = args[0] if args else "How many refund claims can a customer make in thirty days?"
rerank = "--no-rerank" not in sys.argv
probe = gap4_augment([{"id": "3f2a9c1d7e44", "file": "test.pdf", "text": "test"}])     # free check, before any search
if not probe:
    sys.exit("GAP 4 is not filled in yet: write gap4_augment at the top of this file, then run it again.")
if not isinstance(probe, str):
    sys.exit(f"GAP 4 must return one string (the text the model reads), not a {type(probe).__name__}. Join the blocks.")
client = models()

# 1. Retrieve: hybrid search, current documents only, the reranker; the best 5 pieces.
pieces = retrieve(client, question, rerank=rerank)
print(f"Question: {question}\n\nPieces the search found:")
for p in pieces:
    print(f"  [{p['id']}] {p['score']:6.3f}  {p['file']:36} {p['text'][:70]!r}")

if not pieces or (rerank and pieces[0]["score"] < 1.5):      # the gate: no piece matches well enough
    sys.exit("\nRefused before any model call: no sufficiently relevant piece (best reranker score under 1.5 of 4).")

# 2. Augment: your GAP 4.
context = gap4_augment(pieces)
if not isinstance(context, str) or not context:
    sys.exit("\nGAP 4 must return one string: the pieces, each with its id in square brackets.")
if any(f"[{p['id']}]" not in context for p in pieces):
    print("\nWarning: some piece ids are missing from your text, so the model cannot cite those pieces.")

# 3. Generate: the model answers from your text only.
text, usage = chat(client, [{"role": "system", "content": PROMPT},
                            {"role": "user", "content": f"{context}\n\nQuestion: {question}"}])

# 4. Check: every cited id must be one the search returned (the check you write yourself on Day 4).
r = verify({"question": question, "retrieved": pieces, "answer": text})
print(f"\nAnswer:\n{r['answer']}\n")
print(f"Citations: {r['citations'] or 'none'}; not from the search: {r['unknown_citations'] or 'none'}")
print(f"Tokens for the answer: {usage.total_tokens}")
if r["refused"]:
    print(f"Refused: {r['refusal_reason']}")
