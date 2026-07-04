import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from rag.embedder import embed_text, embed_batch
import config


# ── Client ────────────────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Create or open the ChromaDB persistent client.
    PersistentClient stores all collections on disk at config.CHROMA_PATH.
    """
    return chromadb.PersistentClient(
        path=str(config.CHROMA_PATH),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


# ── Collection helpers ────────────────────────────────────────────────────────

def get_or_create_collection(
    client: chromadb.PersistentClient,
    domain: str = config.DEFAULT_DOMAIN,
):
    """
    Get or create the ChromaDB collection for a specific domain.
    """

    collection_name = config.domain_to_collection_name(domain)

    try:
        return client.get_collection(collection_name)
    except Exception:
        return client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "domain": domain,
            },
        )

def list_all_collections(client: chromadb.PersistentClient) -> list[str]:
    """
    Return the human-readable domain names of all existing collections.
    Filters to only collections created by this app (prefixed with COLLECTION_PREFIX).
    """
    prefix = config.COLLECTION_PREFIX + "_"
    return [
        c.name.replace(prefix, "").replace("_", " ").title()
        for c in client.list_collections()
        if c.name.startswith(prefix)
    ]


# ── Fingerprint cache ─────────────────────────────────────────────────────────

def get_documents_fingerprint(documents_path: str) -> str:
    """
    Compute an MD5 fingerprint of the documents directory.
    Changes when files are added, removed, or modified.
    """
    path      = Path(documents_path)
    supported = {".txt", ".md", ".py", ".csv", ".html", ".pdf", ".docx"}

    if not path.exists():
        return "no_documents_directory"

    file_records = sorted(
        [
            {
                "name":  f.name,
                "size":  f.stat().st_size,
                "mtime": round(f.stat().st_mtime, 2),
            }
            for f in path.iterdir()
            if f.suffix.lower() in supported
        ],
        key=lambda r: r["name"],
    )

    if not file_records:
        return "no_supported_documents"

    return hashlib.md5(
        json.dumps(file_records, sort_keys=True).encode()
    ).hexdigest()


def needs_reingestion(
    client: chromadb.PersistentClient,
    documents_path: str,
    domain: str = config.DEFAULT_DOMAIN,
) -> bool:
    """
    Check whether documents changed since last ingestion.
    Updates only the fingerprint metadata.
    """

    collection = get_or_create_collection(client, domain)

    current_fp = get_documents_fingerprint(documents_path)

    metadata = collection.metadata or {}

    stored_fp = metadata.get("documents_fingerprint", "")

    if current_fp == stored_fp:
        print(f"[cache] Domain '{domain}' fingerprint match — skipping ingestion.")
        return False

    print(f"[cache] Domain '{domain}' fingerprint changed — re-ingesting.")

    # Remove reserved Chroma metadata
    metadata = {
        k: v
        for k, v in metadata.items()
        if k != "hnsw:space"
    }

    metadata["documents_fingerprint"] = current_fp

    collection.modify(metadata=metadata)

    return True

# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_chunks(
    chunks: list[dict],
    model:  SentenceTransformer,
    client: chromadb.PersistentClient,
    domain: str = config.DEFAULT_DOMAIN,
) -> int:
    """
    Embed all chunks and store them in the domain's ChromaDB collection.

    Skips chunks whose chunk_id already exists in the collection
    to avoid re-embedding unchanged documents.

    Args:
        chunks: List of chunk dicts from chunker.load_and_chunk().
        model:  Loaded SentenceTransformer.
        client: ChromaDB persistent client.
        domain: Target domain for this batch of chunks.

    Returns:
        Number of new chunks stored.
    """
    if not chunks:
        print(f"[retriever] No chunks to ingest for domain '{domain}'.")
        return 0

    collection = get_or_create_collection(client, domain)
    existing   = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

    if not new_chunks:
        print(f"[retriever] All {len(chunks)} chunks already indexed in '{domain}'.")
        return 0

    print(f"[retriever] Ingesting {len(new_chunks)} new chunks into '{domain}'...")

    texts     = [c["text"]     for c in new_chunks]
    ids       = [c["chunk_id"] for c in new_chunks]
    metadatas = [
        {
            "source":   c["source"],
            "chunk_id": c["chunk_id"],
            "domain":   domain,
            **( {"page": c["page"]} if "page" in c else {} ),
        }
        for c in new_chunks
    ]
    vectors = embed_batch(model, texts)

    collection.add(
        ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas
    )
    print(
        f"[retriever] Stored {len(new_chunks)} chunks in '{domain}'. "
        f"Collection total: {collection.count()}"
    )
    return len(new_chunks)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query:   str,
    model:   SentenceTransformer,
    client:  chromadb.PersistentClient,
    domain:  str = config.DEFAULT_DOMAIN,
    top_k:   int = config.RETRIEVAL_TOP_K,
    top_n:   int = config.RERANK_TOP_N,
) -> list[dict]:
    """
    Retrieve relevant chunks for a query from a specific domain.

    Pipeline:
        1. Embed the query
        2. ChromaDB cosine similarity search (top_k candidates)
        3. Filter by similarity threshold
        4. Return top_n best matches

    Args:
        query:  User question.
        model:  Loaded SentenceTransformer (same model as ingestion).
        client: ChromaDB persistent client.
        domain: Domain to search. Searches only that collection.
        top_k:  Candidates fetched from ChromaDB.
        top_n:  Candidates returned after re-ranking.

    Returns:
        List of chunk dicts sorted by relevance score, best first:
        [{"text": "...", "source": "file.pdf", "score": 0.87, "page": 3}, ...]
    """
    collection = get_or_create_collection(client, domain)

    if collection.count() == 0:
        print(f"[retriever] Domain '{domain}' is empty — nothing to search.")
        return []

    query_vector = embed_text(model, query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append({
            "text":   doc,
            "source": meta.get("source", "unknown"),
            "domain": meta.get("domain", domain),
            "score":  round(1.0 - dist, 4),
            **( {"page": meta["page"]} if "page" in meta else {} ),
        })

    SIMILARITY_THRESHOLD = 0.3
    ranked = sorted(
        [c for c in candidates if c["score"] >= SIMILARITY_THRESHOLD],
        key=lambda x: x["score"],
        reverse=True,
    )
    final = ranked[:top_n]

    print(
        f"[retriever] '{domain}' | query='{query[:40]}...' | "
        f"fetched={len(candidates)} | returned={len(final)}"
    )
    return final


def retrieve_all_domains(
    query:  str,
    model:  SentenceTransformer,
    client: chromadb.PersistentClient,
    top_n:  int = config.RERANK_TOP_N,
) -> list[dict]:
    """
    Search every existing domain collection and merge results.

    Each domain contributes up to top_n chunks. Final results are
    sorted by score across all domains so the best chunks always
    surface first regardless of which domain they came from.

    Args:
        query:  User question.
        model:  Loaded SentenceTransformer.
        client: ChromaDB persistent client.
        top_n:  Chunks to return per domain (merged and re-sorted).

    Returns:
        Merged list of chunk dicts from all domains, sorted by score.
    """
    all_chunks: list[dict] = []
    domains = list_all_collections(client)

    if not domains:
        print("[retriever] No domain collections found.")
        return []

    for domain in domains:
        chunks = retrieve(query, model, client, domain=domain, top_n=top_n)
        all_chunks.extend(chunks)

    # Re-sort the merged results so best chunks are always first
    merged = sorted(all_chunks, key=lambda x: x["score"], reverse=True)
    print(f"[retriever] All-domain search: {len(merged)} chunks from {len(domains)} domains")
    return merged[:top_n * 2]   # Return a generous window across domains


# ── Upload helpers ────────────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file, documents_path: Path) -> Path:
    """
    Save a Streamlit UploadedFile to the documents directory.
    Raises ValueError if the filename already exists.
    """
    documents_path = Path(documents_path)
    documents_path.mkdir(parents=True, exist_ok=True)
    destination    = documents_path / uploaded_file.name

    if destination.exists():
        raise ValueError(
            f"'{uploaded_file.name}' already exists. "
            "Rename the file or delete the existing one first."
        )

    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())

    print(f"[upload] Saved: {destination.name} ({destination.stat().st_size} bytes)")
    return destination


def ingest_uploaded_file(
    uploaded_file,
    model,
    client,
    documents_path: Path,
    domain: str = config.DEFAULT_DOMAIN,
) -> int:
    """
    Save + chunk + embed + store a single uploaded file into a specific domain.
    Refreshes the fingerprint so the next full restart skips this file.
    """
    from rag.chunker import load_text_document, load_pdf, load_docx, clean_text, split_text

    saved_path = save_uploaded_file(uploaded_file, documents_path)
    ext        = saved_path.suffix.lower()

    if ext == ".pdf":
        raw_docs = load_pdf(saved_path)
    elif ext == ".docx":
        doc = load_docx(saved_path)
        raw_docs = [doc] if doc else []
    else:
        doc = load_text_document(saved_path)
        raw_docs = [doc] if doc else []

    if not raw_docs:
        print(f"[upload] No extractable content in {saved_path.name}")
        return 0

    chunks = []
    for doc in raw_docs:
        cleaned = clean_text(doc["text"])
        pieces  = split_text(cleaned)
        is_pdf  = ext == ".pdf"

        for idx, piece in enumerate(pieces):
            if is_pdf:
                page = doc.get("page", 1)
                chunks.append({
                    "text": piece, "source": doc["source"],
                    "chunk_id": f"{doc['source']}_p{page}_{idx}", "page": page,
                })
            else:
                chunks.append({
                    "text": piece, "source": doc["source"],
                    "chunk_id": f"{doc['source']}_{idx}",
                })

    count = ingest_chunks(chunks, model, client, domain=domain)
    needs_reingestion(client, str(documents_path), domain)
    return count

