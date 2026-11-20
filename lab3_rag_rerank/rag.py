"""The RAG pipeline of Days 3 and 4 in one place. Every Day 3 and Day 4 script that answers a question calls this file.

RAG (retrieval-augmented generation) means: search your own documents first, then let the model answer from what the
search found, not from its memory. This file does it in four steps:

    1. retrieve()  search FreshCart's documents and keep up to 5 best pieces (hybrid search plus the semantic reranker)
    2. answer()    gate: if the best piece scores under min_score, refuse before asking the model
    3. answer()    generate: send the pieces, each labelled with its id, and the question to the model, with PROMPT
    4. verify()    check: every id the answer cites must be one of the ids the search returned

Who uses it: lab3_rag_rerank/l3_rag.py (one question, step by step), lab4_rag_api/l4_api.py (the API) and
lab4_rag_api/l4_score.py (25 test questions). You do not run this file yourself.

A piece is one chunk of a document in the trainer's search index freshcart-chunks: about 500 characters, with an id
(12 hex characters, 0 to 9 and a to f), the file it came from, and a label saying whether that document is current.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from azure_common import chat, embed, search_client

# The exact sentence the model must use when the pieces do not answer the question. Code recognises a refusal by it.
REFUSAL = "I could not find the answer in FreshCart's documents."
# Prompt v1, the instructions the model gets with every question. Day 4's prompt v2 (in l4_score.py) adds a scope rule.
PROMPT = f"""You answer FreshCart support staff using only the numbered pieces of FreshCart documents below.
Cite every fact with the id of the piece it came from in square brackets, like [3f2a9c1d7e44].
If the pieces do not answer the question, reply exactly: {REFUSAL}"""


def retrieve(client, question, rerank=True, current_only=True, top=5):
    """Step 1, retrieve: find the pieces of FreshCart's documents that best match the question.

    Hybrid search runs keyword search and meaning (vector) search together and merges the two lists. The semantic
    reranker then rereads the top results with the question and scores each from 0 to 4. current_only keeps
    withdrawn documents, such as the 2022 refund policy, out of the results.
    Returns a list of up to `top` pieces, best first, each a dict: id, file, text and score (the reranker score,
    or the search score when rerank is False)."""
    from azure.search.documents.models import VectorizedQuery
    vector = embed(client, [question])[0]                       # the question as numbers, for meaning search
    options = {"query_type": "semantic", "semantic_configuration_name": "rerank"} if rerank else {}
    found = search_client().search(search_text=question, top=top, select=["id", "file", "content"],
                                   vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=10, fields="vector")],
                                   filter="current eq true" if current_only else None, **options)
    return [{"id": p["id"], "file": p["file"], "text": p["content"],
             "score": p.get("@search.reranker_score") or p["@search.score"]} for p in found]


def cited_ids(answer):
    """Every id the answer cites, in lower case, in the order written.

    A citation is an id in square brackets, like [3f2a9c1d7e44]. One bracket may hold several ids separated by
    commas, like [3f2a9c1d7e44, 8b0e5a6f1c22]. Only brackets that hold hex characters (0 to 9, a to f), commas and
    spaces count, so [sic] is not a citation. An id of the wrong length is kept as written, so verify() reports it
    as unknown. Example: "Three claims [3F2A9C1D7E44]." gives ["3f2a9c1d7e44"]."""
    groups = re.findall(r"\[([0-9a-fA-F,\s]+)\]", answer)            # brackets of words, like [sic], are not citations
    return [i.strip().lower() for group in groups for i in group.split(",") if i.strip()]


def answer(client, question, min_score=1.5, rerank=True, prompt=PROMPT):
    """The whole pipeline for one question: retrieve, gate, generate, check. Returns one dict with every result.

    min_score is on the reranker's 0 to 4 scale: when the best piece scores lower, refuse before asking the model
    (the gate). prompt is the instructions the model gets; Day 4 passes prompt v2 to compare the two.
    The dict holds: question, retrieved (the pieces), answer, citations, unknown_citations, refused, refusal_reason,
    and tokens when the model was asked."""
    pieces = retrieve(client, question, rerank=rerank)
    result = {"question": question, "retrieved": pieces, "citations": [], "unknown_citations": [],
              "refused": False, "refusal_reason": None, "answer": REFUSAL}
    # Step 2, the gate: no piece, or the best piece matches too poorly. Refuse without a model call.
    if not pieces or (rerank and pieces[0]["score"] < min_score):
        result.update(refused=True, refusal_reason="no sufficiently relevant piece")
        return result
    # Step 3, generate: each piece goes into the prompt with its id in square brackets, so the model can cite it.
    context = "\n\n".join(f"[{p['id']}] ({p['file']})\n{p['text']}" for p in pieces)
    text, usage = chat(client, [{"role": "system", "content": prompt},
                                {"role": "user", "content": f"{context}\n\nQuestion: {question}"}])
    result.update(answer=text, tokens=usage.total_tokens)
    return verify(result)                                        # step 4, the citation check


def verify(result):
    """Step 4, the citation check, done in code and never by asking the model.

    Reads the ids the answer cites and compares each with the ids of the pieces the search returned. A cited id the
    search never returned is a made-up source. Sets citations, unknown_citations, refused and refusal_reason, in
    this order of reasons: the model's own refusal sentence; any made-up source; no citation at all.
    Called again after any change to the answer text, for example when l4_api.py plants a fake id.
    What it cannot catch: a real id that does not support the sentence, such as India's refund rules cited for a
    question about Singapore. That needs a test set with right answers, as in Day 4 handbook part P2."""
    text = result["answer"]
    cited = cited_ids(text)
    unknown = [c for c in cited if c not in {p["id"] for p in result["retrieved"]}]    # the whole check, one line
    result.update(citations=cited, unknown_citations=unknown, refused=False, refusal_reason=None)
    if text.startswith(REFUSAL[:30]):
        result.update(refused=True, refusal_reason="the model found no answer in the pieces")
    elif unknown:
        result.update(refused=True, refusal_reason=f"cites pieces the search never returned: {unknown}")
    elif not cited:
        result.update(refused=True, refusal_reason="no citation")
    return result
