# RAG Document Chatbot

Production-style **Retrieval-Augmented Generation** chatbot: ingest documents, chunk + embed them into a FAISS vector store, and answer questions grounded in retrieved context — with source citations and a hallucination guard.

Built with the same architecture patterns I use for enterprise RAG systems (document ingestion → chunking → embedding → retrieval → grounded generation → evaluation).

## Features

- **Ingestion pipeline**: PDF/TXT/Markdown loaders, recursive chunking with overlap
- **Vector search**: FAISS index with sentence-transformer embeddings (runs fully local) or OpenAI embeddings
- **Grounded answers**: LLM answers only from retrieved context, returns source chunks with every response
- **Hallucination guard**: refuses to answer when retrieval confidence is low
- **FastAPI backend** + simple CLI chat mode
- **Evaluation harness**: retrieval hit-rate and answer faithfulness scoring

## Architecture

```
documents/ -> ingest.py (load, chunk) -> embeddings -> FAISS index
query -> retriever (top-k + score threshold) -> prompt template -> LLM -> answer + citations
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY (optional — local mode works without it)

# 1. Build the index from the sample docs
python src/ingest.py --docs docs_sample

# 2. Chat in the terminal
python src/chat.py --query "What are the main risk factors described?"

# 3. Or serve the API
uvicorn src.api:app --reload
```

## Tech Stack

Python · LangChain · FAISS · sentence-transformers · OpenAI API · FastAPI

## Evaluation

`python src/evaluate.py` runs a small QA set against the index and reports retrieval hit-rate and grounding score. Extend `eval_set.json` with your own domain questions.

## License

MIT
