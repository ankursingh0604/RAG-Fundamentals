# RAG Fundamentals

A hands-on implementation of core RAG (Retrieval Augmented Generation) concepts built from scratch.

## What's in this repo

- **bm25.py** — BM25 keyword-based retrieval
- **rrf.py** — Reciprocal Rank Fusion combining multiple retrievers
- **semanticretriever.py** — Semantic search using sentence transformers
- **weaviate_vector_db.py** — Full RAG pipeline with Weaviate vector database

## Tech Stack
- Python
- Weaviate
- Sentence Transformers
- rank-bm25

## Concepts Covered
- BM25 keyword search
- Semantic search using embeddings
- Reciprocal Rank Fusion (RRF)
- Hybrid search (BM25 + Semantic)
- Text chunking with overlap
- Vector database indexing and querying