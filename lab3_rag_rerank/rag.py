"""The RAG pipeline of Days 3 and 4 in one place: find pieces, gate on the match, write an answer with citations,
check the citations in code. l3_rag.py runs it on one question; Day 4's API and scoring import it.
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from azure_common import chat, embed, search_client

REFUSAL = "I could not find the answer in FreshCart's documents."
PROMPT = f"""You answer FreshCart support staff using only the numbered pieces of FreshCart documents below.
Cite every fact with the id of the piece it came from in square brackets, like [3f2a9c1d7e44].
If the pieces do not answer the question, reply exactly: {REFUSAL}"""


def retrieve(client, question, rerank=True, current_only=True, top=5):
    """Hybrid search (keyword plus meaning), then the semantic reranker, keeping the best `top` pieces."""
    from azure.search.documents.models import VectorizedQuery
    vector = embed(client, [question])[0]
    options = {"query_type": "semantic", "semantic_configuration_name": "rerank"} if rerank else {}
    found = search_client().search(search_text=question, top=top, select=["id", "file", "content"],
                                   vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=10, fields="vector")],
                                   filter="current eq true" if current_only else None, **options)
    return [{"id": p["id"], "file": p["file"], "text": p["content"],
             "score": p.get("@search.reranker_score") or p["@search.score"]} for p in found]


def cited_ids(answer):
    """Every id the answer cites, exactly as written. One bracket may hold several ids, like [3f2a9c1d7e44, 8b0e5a6f1c22].
    An id that is not exactly 12 hex characters is kept as written, so the check below reports it as unknown."""
    groups = re.findall(r"\[([0-9a-fA-F,\s]+)\]", answer)            # brackets of words, like [sic], are not citations
    return [i.strip().lower() for group in groups for i in group.split(",") if i.strip()]


def answer(client, question, min_score=1.5, rerank=True, prompt=PROMPT):
    """The whole pipeline. min_score is on the reranker's 0 to 4 scale: below it, refuse before asking the model."""
    pieces = retrieve(client, question, rerank=rerank)
    result = {"question": question, "retrieved": pieces, "citations": [], "unknown_citations": [],
              "refused": False, "refusal_reason": None, "answer": REFUSAL}
    if not pieces or (rerank and pieces[0]["score"] < min_score):
        result.update(refused=True, refusal_reason="no sufficiently relevant piece")
        return result
    context = "\n\n".join(f"[{p['id']}] ({p['file']})\n{p['text']}" for p in pieces)
    text, usage = chat(client, [{"role": "system", "content": prompt},
                                {"role": "user", "content": f"{context}\n\nQuestion: {question}"}])
    result.update(answer=text, tokens=usage.total_tokens)
    return verify(result)


def verify(result):
    """Check the answer's citations in code, never by asking the model: each cited id must be one the search
    returned. Called again after any change to the answer text."""
    text = result["answer"]
    cited = cited_ids(text)
    unknown = [c for c in cited if c not in {p["id"] for p in result["retrieved"]}]
    result.update(citations=cited, unknown_citations=unknown, refused=False, refusal_reason=None)
    if text.startswith(REFUSAL[:30]):
        result.update(refused=True, refusal_reason="the model found no answer in the pieces")
    elif unknown:
        result.update(refused=True, refusal_reason=f"cites pieces the search never returned: {unknown}")
    elif not cited:
        result.update(refused=True, refusal_reason="no citation")
    return result
