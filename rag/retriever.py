import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from rag.embedder import embed_text
import config


COLLECTION_NAME = "rag_documents"


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Create or open the ChromaDB persistent client.

    PersistentClient stores vectors on disk at config.CHROMA_PATH.
    The data survives restarts — you only ingest documents once.

    Returns:
        A ChromaDB client connected to the local vector store.
    """
    return chromadb.PersistentClient(
        path=str(config.CHROMA_PATH),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient):
    """
    Get the documents collection, creating it if it doesn't exist.
    We use cosine similarity — standard for sentence embeddings.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    client: chromadb.PersistentClient,
) -> int:
    """
    Embed all chunks and store them in ChromaDB.

    This is the ingestion step — called once when documents are loaded,
    or when new documents are added.

    Args:
        chunks: List of chunk dicts from chunker.load_and_chunk()
                Each must have: text, source, chunk_id
        model:  Loaded embedding model.
        client: ChromaDB persistent client.

    Returns:
        Number of chunks successfully stored.
    """
    if not chunks:
        print("[retriever] No chunks to ingest.")
        return 0

    collection = get_or_create_collection(client)

    # Check which chunk_ids are already stored — avoid re-embedding duplicates
    existing = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

    if not new_chunks:
        print(f"[retriever] All {len(chunks)} chunks already in ChromaDB. Skipping.")
        return 0

    print(f"[retriever] Ingesting {len(new_chunks)} new chunks...")

    texts     = [c["text"]     for c in new_chunks]
    ids       = [c["chunk_id"] for c in new_chunks]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]}
                 for c in new_chunks]

    from rag.embedder import embed_batch
    vectors = embed_batch(model, texts)

    collection.add(
        ids        = ids,
        documents  = texts,
        embeddings = vectors,
        metadatas  = metadatas,
    )

    print(f"[retriever] Stored {len(new_chunks)} chunks. "
          f"Collection total: {collection.count()}")
    return len(new_chunks)


def retrieve(
    query: str,
    model: SentenceTransformer,
    client: chromadb.PersistentClient,
    top_k: int   = config.RETRIEVAL_TOP_K,
    top_n: int   = config.RERANK_TOP_N,
) -> list[dict]:
    """
    Find the most relevant chunks for a query using vector similarity,
    then re-rank and return only the best top_n.

    Pipeline:
      1. Embed the query using the same model used at ingestion
      2. Query ChromaDB for top_k nearest neighbors by cosine similarity
      3. Re-rank: filter out low-confidence results (similarity < threshold)
      4. Return top_n chunks with text, source, and similarity score

    Args:
        query:  The user's question as a plain string.
        model:  Loaded embedding model (same as used at ingestion).
        client: ChromaDB persistent client.
        top_k:  How many candidates to fetch from ChromaDB.
        top_n:  How many to keep after re-ranking.

    Returns:
        List of chunk dicts, best match first:
        [{"text": "...", "source": "file.txt", "score": 0.87}, ...]
    """
    collection = get_or_create_collection(client)

    if collection.count() == 0:
        print("[retriever] ChromaDB is empty — no documents ingested yet.")
        return []

    # Step 1: Embed the query
    query_vector = embed_text(model, query)

    # Step 2: Fetch top_k candidates from ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Step 3: Parse and re-rank
    # ChromaDB returns cosine DISTANCE (0=identical, 2=opposite)
    # Convert to similarity: similarity = 1 - distance
    candidates = []
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = 1.0 - dist

        candidates.append({
            "text":   doc,
            "source": meta.get("source", "unknown"),
            "score":  round(similarity, 4),
        })

    # Sort by similarity descending (ChromaDB usually returns sorted,
    # but we re-sort to be safe after any future filtering steps)
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Step 4: Apply minimum similarity threshold and take top_n
    SIMILARITY_THRESHOLD = 0.3   # Discard chunks with < 30% relevance
    ranked = [c for c in candidates if c["score"] >= SIMILARITY_THRESHOLD]
    final  = ranked[:top_n]

    print(f"[retriever] Query: '{query[:50]}...' | "
          f"Fetched: {len(candidates)} | "
          f"After re-rank: {len(final)}")

    for i, chunk in enumerate(final):
        print(f"  [{i+1}] score={chunk['score']:.3f} | source={chunk['source']} | "
              f"preview='{chunk['text'][:60]}...'")

    return final