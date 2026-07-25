# Retrieval-Augmented Generation (RAG) Overview

Retrieval-augmented generation combines an information retrieval system with a
large language model. Instead of relying only on the model's parametric
knowledge, relevant documents are retrieved from a vector store at query time
and injected into the prompt as context.

## Why RAG

- Grounds answers in up-to-date, domain-specific documents
- Reduces hallucinations by constraining generation to retrieved context
- Enables citations, which makes answers auditable

## Reducing hallucinations

Common techniques include similarity-score thresholds (refuse to answer on
weak retrieval), instructing the model to answer only from context, and
post-hoc faithfulness scoring of generated answers against source chunks.
