# Document Chunking Guide

Chunk size and overlap are the most important ingestion hyperparameters in a
RAG pipeline.

- **Chunk size** controls the granularity of retrieval. Too large and chunks
  contain diluted, off-topic content; too small and answers lose context.
- **Overlap** preserves continuity across boundaries so facts split across
  two chunks remain retrievable.

A common starting point is 500–1000 characters with 10–20% overlap, tuned
against a retrieval-evaluation set.
