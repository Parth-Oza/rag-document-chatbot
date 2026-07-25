"""Shared configuration: embeddings, LLM, retrieval thresholds."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = "index"
TOP_K = 4
SCORE_THRESHOLD = 0.30  # below this similarity, refuse to answer


def get_embeddings():
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm():
    """Returns an LLM if an API key is configured, else None (local mode)."""
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return None
