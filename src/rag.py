"""Core RAG chain: retrieve -> guard -> grounded answer with citations."""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS

from settings import INDEX_DIR, SCORE_THRESHOLD, TOP_K, get_embeddings, get_llm

PROMPT = """You are a careful assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't have enough information in the indexed documents."

Context:
{context}

Question: {question}

Answer (cite sources as [source: filename]):"""


@dataclass
class RagResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    grounded: bool = True


class RagPipeline:
    def __init__(self) -> None:
        self.store = FAISS.load_local(
            INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True
        )
        self.llm = get_llm()

    def retrieve(self, query: str):
        results = self.store.similarity_search_with_relevance_scores(query, k=TOP_K)
        return [(doc, score) for doc, score in results if score >= SCORE_THRESHOLD]

    def ask(self, query: str) -> RagResponse:
        hits = self.retrieve(query)
        if not hits:
            return RagResponse(
                answer="I don't have enough information in the indexed documents "
                "to answer that reliably.",
                grounded=False,
            )

        context = "\n\n".join(
            f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d, _ in hits
        )
        sources = [
            {"source": d.metadata.get("source", "unknown"), "score": round(s, 3)}
            for d, s in hits
        ]

        if self.llm is None:
            # Local extractive fallback: return the most relevant chunk
            best = hits[0][0]
            return RagResponse(
                answer=f"(local mode — most relevant passage)\n{best.page_content}",
                sources=sources,
            )

        msg = self.llm.invoke(PROMPT.format(context=context, question=query))
        return RagResponse(answer=msg.content, sources=sources)
