"""Day 4, code part 1: the Day 3 RAG pipeline as an API that any program can call.

    python lab4_rag_api/l4_api.py            start it, then open http://127.0.0.1:8000/docs in your browser
    python lab4_rag_api/l4_api.py --port 8001

A portal agent answers people in a chat window. An API answers other programs: a support screen, a nightly test.
Every reply has the same fields, refused or not, so the calling program never has to guess.
Stop it with Ctrl+C.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lab3_rag_rerank"))
from azure_common import models
from fastapi import FastAPI
from pydantic import BaseModel, Field
from rag import answer

app = FastAPI(title="FreshCart answers with sources")
client = models()


class Question(BaseModel):
    question: str = Field(examples=["How many refund claims can a customer make in thirty days?"])
    min_score: float = Field(1.5, ge=0, le=4, description="Refuse when the best piece's reranker score is lower (0 to 4)")


class Reply(BaseModel):
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
    r = answer(client, q.question, min_score=q.min_score)
    return Reply(answer=r["answer"], citations=r["citations"], unknown_citations=r["unknown_citations"],
                 confidence=r["retrieved"][0]["score"] if r["retrieved"] else None, refused=r["refused"],
                 refusal_reason=r["refusal_reason"],
                 retrieved=[{"id": p["id"], "file": p["file"], "score": p["score"]} for p in r["retrieved"]])


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    print(f"API ready: open http://127.0.0.1:{port}/docs")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
