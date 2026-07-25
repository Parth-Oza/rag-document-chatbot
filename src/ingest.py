"""Ingestion pipeline: load documents, chunk, embed, persist FAISS index."""
from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import get_embeddings, INDEX_DIR

LOADERS = {".pdf": PyPDFLoader, ".txt": TextLoader, ".md": TextLoader}


def load_documents(docs_dir: str):
    docs = []
    for path in sorted(Path(docs_dir).rglob("*")):
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls:
            docs.extend(loader_cls(str(path)).load())
    if not docs:
        raise SystemExit(f"No supported documents found in {docs_dir}")
    return docs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="docs_sample")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--chunk-overlap", type=int, default=120)
    args = p.parse_args()

    docs = load_documents(args.docs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")

    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(INDEX_DIR)
    print(f"Saved FAISS index -> {INDEX_DIR}/")


if __name__ == "__main__":
    main()
