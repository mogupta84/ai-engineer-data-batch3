"""Day 4, code part: the Day 3 RAG pipeline as an API, a web service that any program can call.

WHAT IT DOES
    RAG (retrieval-augmented generation) answers a question in four steps:
      1. Retrieve: search FreshCart's documents and keep up to 5 best pieces (hybrid search plus the reranker).
      2. Gate:     if even the best piece matches the question poorly (reranker score under min_score), refuse
                   at once, without asking the model.
      3. Generate: the model writes an answer from those pieces only, and puts the id of the piece it used after
                   each fact, in square brackets, like [25dc91fccfdf]. That id is the citation.
      4. Check:    code compares every cited id with the ids the search returned. An id the search never returned
                   is a made-up source, and the answer is refused.
    Steps 1 to 4 live in lab3_rag_rerank/rag.py (the function answer). This file only wraps them in an API.

WHY AN API
    A portal agent answers people in a chat window. An API answers other programs: a support screen, a chatbot,
    a nightly test. Every reply has the same seven fields, refused or not, so the calling program never guesses.

HOW TO RUN IT (from the repository folder, in the VS Code terminal)
    python lab4_rag_api/l4_api.py              start it; wait for "API ready"
    python lab4_rag_api/l4_api.py --port 8001  the same on another port, if 8000 is busy
    python lab4_rag_api/l4_api.py --my-check   optional extra: YOUR check from l4_checks.py decides every refusal
    Then open http://127.0.0.1:8000/docs in a browser, click POST /ask, Try it out, Execute.
    Stop it with Ctrl+C in the terminal. On Windows the path also works with backslashes: lab4_rag_api\\l4_api.py

WHAT YOU SEND (the Request body on the /docs page)
    question         the question, in plain words
    min_score        0 to 4, default 1.5: refuse before the model is asked when the best piece scores lower
    add_fake_source  test only, default false: when true, the API adds a made-up id, [deadbeef0000], to the
                     answer before the check runs, to show what the check does with an invented source

WHAT COMES BACK (the reply)
    answer             the answer text, or the refusal sentence
    citations          every id the answer cites
    unknown_citations  cited ids the search never returned (made-up sources); normally empty
    confidence         the reranker score of the best piece, 0 to 4: how well it matches the question,
                       not how sure the model is; null when the search found nothing
    refused            true means the calling program must not show the answer as fact
    refusal_reason     why it was refused, or null
    retrieved          the pieces the search returned (up to 5): id, file name and score, so every citation can be traced
                       to a file

COST
    One embedding (the question turned into numbers), one search and at most one short model answer per question
    (none when the gate refuses): well under one cent.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lab3_rag_rerank"))
from azure_common import models
from fastapi import FastAPI
from pydantic import BaseModel, Field
from rag import REFUSAL, answer, verify

# Optional extra: with --my-check, the student's own citation check (lab4_rag_api/l4_checks.py) decides every reply the
# model wrote, instead of the check in rag.py. A question the score gate refused stays refused. Before starting, make sure that file is filled in, so the API never runs half-written.
MY_CHECK = "--my-check" in sys.argv
if MY_CHECK:
    try:
        import l4_checks                                   # your check; run python lab4_rag_api/l4_checks.py first
        ready = l4_checks.written()
    except Exception as e:
        sys.exit(f"--my-check: your l4_checks.py raised {type(e).__name__}: {e}. Run python lab4_rag_api/l4_checks.py.")
    if not ready:
        sys.exit("--my-check needs l4_checks.py filled in. Run python lab4_rag_api/l4_checks.py until all 9 pass.")

app = FastAPI(title="FreshCart answers with sources")     # FastAPI builds the /docs page from the classes below
client = models()                                          # one signed-in client for the embedding and chat models


class Question(BaseModel):
    """What a caller sends to POST /ask. FastAPI checks the types and ranges and rejects a bad request itself."""
    question: str = Field(examples=["How many refund claims can a customer make in thirty days?"])
    min_score: float = Field(1.5, ge=0, le=4, description="Refuse when the best piece's reranker score is lower (0 to 4)")
    add_fake_source: bool = Field(False, description="Test only: add a made-up citation to the answer. Does the check stop it?")


class Reply(BaseModel):
    """What POST /ask always returns: the same seven fields, whether the answer was given or refused."""
    answer: str
    citations: list[str]
    unknown_citations: list[str]
    confidence: float | None = Field(description="The reranker score of the best piece: how well it matches the "
                                                 "question, not how sure the model is")
    refused: bool
    refusal_reason: str | None
    retrieved: list[dict]


@app.post("/ask", response_model=Reply)
def ask(q: Question):
    """Answer one question with the RAG pipeline and return the seven fields described at the top of this file."""
    # Steps 1 to 4 (retrieve, gate, generate, check) all happen inside rag.answer.
    r = answer(client, q.question, min_score=q.min_score)
    # True when step 2 refused: the best piece matched too poorly, so the model was never asked.
    gated = r["refusal_reason"] == "no sufficiently relevant piece"
    # Test switch: plant a made-up id at the end of a real answer, then run the citation check again on the new
    # text. deadbeef0000 is not one of the ids the search returned, so the check lists it in unknown_citations and
    # refuses the answer. Skipped when there is no real answer to plant it in.
    if q.add_fake_source and not gated and not r["answer"].startswith(REFUSAL[:30]):
        r = verify({**r, "answer": r["answer"] + " [deadbeef0000]"})
    # Optional extra: the student's own check replaces the verdict from rag.py. "(your check)" marks its reason.
    if MY_CHECK and not gated:
        try:
            cited, unknown, reason = l4_checks.check(r["answer"], [p["id"] for p in r["retrieved"]])
            r.update(citations=[str(c) for c in cited or []], unknown_citations=[str(u) for u in unknown or []],
                     refused=reason is not None, refusal_reason=None if reason is None else f"{reason} (your check)")
        except Exception as e:
            r.update(refused=True, refusal_reason=f"your check raised {type(e).__name__}: {e}")
    # Shape the result into the reply. confidence is the best piece's score; retrieved keeps id, file and score only.
    return Reply(answer=r["answer"], citations=r["citations"], unknown_citations=r["unknown_citations"],
                 confidence=r["retrieved"][0]["score"] if r["retrieved"] else None, refused=r["refused"],
                 refusal_reason=r["refusal_reason"],
                 retrieved=[{"id": p["id"], "file": p["file"], "score": p["score"]} for p in r["retrieved"]])


@app.get("/health")
def health():
    """GET /health: answers {"status": "ok"} while the service is running. Monitoring tools call this."""
    return {"status": "ok"}


if __name__ == "__main__":
    # uvicorn is the small web server that runs the FastAPI app on this laptop only (127.0.0.1).
    import uvicorn
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    print(f"API ready: open http://127.0.0.1:{port}/docs" + ("   (your citation check is on)" if MY_CHECK else ""))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
