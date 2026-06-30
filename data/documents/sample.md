# Introduction to Retrieval-Augmented Generation

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI architecture that enhances
large language model responses by retrieving relevant information from an
external knowledge base before generating an answer.

Unlike standard LLMs that rely solely on training data, RAG systems can
access up-to-date, domain-specific information and provide answers grounded
in real documents.

## How RAG Works

The RAG pipeline has two phases:

**Ingestion:** Documents are split into chunks, converted into vector
embeddings, and stored in a vector database like ChromaDB.

**Retrieval:** When a user asks a question, it is embedded using the same
model, and the nearest matching chunks are retrieved via cosine similarity.
These chunks are passed as context to the LLM alongside the question.

## Key Components

**Embeddings** are numerical representations of text in high-dimensional
space. Semantically similar texts produce vectors that are geometrically
close, enabling semantic search beyond keyword matching.

**Vector databases** such as ChromaDB, Pinecone, and Weaviate store these
vectors and support fast approximate nearest-neighbor search.

**Chunking** is the process of splitting documents into overlapping segments
so that no important context is lost at chunk boundaries.

**Prompt engineering** determines how retrieved context, conversation
history, and the user question are assembled into the final LLM input.

## Advantages of RAG

- Reduces hallucination by grounding answers in retrieved evidence
- Supports domain-specific knowledge without fine-tuning
- Knowledge base can be updated without retraining the model
- Provides source citations for transparency and trust