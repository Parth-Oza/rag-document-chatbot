"""FastAPI endpoint for the RAG pipeline."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from rag import RagPipeline

app = FastAPI(title="RAG Document Chatbot", version="1.0.0")
_pipe: RagPipeline | None = None


class Query(BaseModel):
    question: str


def pipe() -> RagPipeline:
    global _pipe
    if _pipe is None:
        _pipe = RagPipeline()
    return _pipe


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Query):
    resp = pipe().ask(q.question)
    return {"answer": resp.answer, "sources": resp.sources, "grounded": resp.grounded}
