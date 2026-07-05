import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from rag.embedder import embed_text, embed_batch
import config
from rag.bm25_index import BM25Index
from logger import get_logger

log = get_logger(__name__)
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
):
    """
    Check whether documents changed since last ingestion.
    Only compares fingerprints.
    """

    collection = get_or_create_collection(client, domain)

    current_fp = get_documents_fingerprint(documents_path)

    stored_fp = collection.metadata.get(
        "documents_fingerprint",
        ""
    )

    if current_fp == stored_fp:
        log.info(
    "Fingerprint match (%s...) — skipping ingestion.",
    current_fp[:8],
)
        return False

    log.info(
    "Fingerprint changed: %s... → %s...",
    stored_fp[:8] or "none",
    current_fp[:8],
)
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
        log.warning(
    "No chunks to ingest for domain '%s'.",
    domain,
)
        return 0

    collection = get_or_create_collection(client, domain)
    existing   = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

    if not new_chunks:
        log.info(
    "All %d chunks already indexed in '%s'.",
    len(chunks),
    domain,
)
        return 0

    log.info(
    "Ingesting %d new chunks into '%s'...",
    len(new_chunks),
    domain,
)

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
    metadata = dict(collection.metadata or {})
    metadata.pop("hnsw:space", None)
    metadata["documents_fingerprint"] = get_documents_fingerprint(
    str(config.DOCUMENTS_DIR)
)

    try:
        collection.modify(metadata=metadata)
    except Exception:
        pass

    log.info(
    "Stored %d chunks in '%s'. Collection total: %d",
    len(new_chunks),
    domain,
    collection.count(),
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
        log.warning(
    "Domain '%s' is empty — nothing to search.",
    domain,
)
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
            "chunk_id": meta.get("chunk_id"), 
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

    log.info(
    "'%s' | fetched=%d | returned=%d",
    domain,
    len(candidates),
    len(final),
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
        log.warning("No domain collections found.")
        return []

    for domain in domains:
        chunks = retrieve(query, model, client, domain=domain, top_n=top_n)
        all_chunks.extend(chunks)

    # Re-sort the merged results so best chunks are always first
    merged = sorted(all_chunks, key=lambda x: x["score"], reverse=True)
    log.info(
    "All-domain search: %d chunks from %d domains",
    len(merged),
    len(domains),
)
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

    log.info(
    "Saved: %s (%d bytes)",
    destination.name,
    destination.stat().st_size,
)
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
        log.warning(
    "No extractable content in %s",
    saved_path.name,
)
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

# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results:     list[dict],
    k:                int   = config.RRF_K,
    semantic_weight:  float = config.SEMANTIC_WEIGHT,
    bm25_weight:      float = config.BM25_WEIGHT,
) -> list[dict]:
    """
    Merge semantic and BM25 result lists using Reciprocal Rank Fusion.

    RRF is scale-independent — it works purely on rank position, not
    raw scores. This solves the problem of combining cosine similarities
    (0.0–1.0) with BM25 scores (0.0–∞) without arbitrary normalization.

    Formula: rrf_score = semantic_weight / (k + rank_s)
                       + bm25_weight    / (k + rank_b)

    Where rank_s and rank_b are 1-indexed positions in their respective lists.
    A chunk absent from one list is simply not added from that list —
    it may still appear via the other list.

    Args:
        semantic_results: Chunks from retrieve(), ordered by cosine similarity.
        bm25_results:     Chunks from BM25Index.search(), ordered by BM25 score.
        k:                RRF smoothing constant (default 60, per literature).
        semantic_weight:  Contribution weight of semantic rankings.
        bm25_weight:      Contribution weight of BM25 rankings.

    Returns:
        Merged list of chunk dicts sorted by combined RRF score descending.
        Each chunk has a "score" key containing the final fused score,
        and a "retrieval_method" key for debug visibility.
    """
    # Map chunk_id → accumulated RRF score
    fused_scores: dict[str, float] = {}

    # Map chunk_id → the chunk dict (for reconstruction after scoring)
    chunk_map: dict[str, dict] = {}

    # Score each chunk from semantic results by its rank position
    for rank, chunk in enumerate(semantic_results, start=1):
        cid = chunk["chunk_id"]
        fused_scores[cid]   = fused_scores.get(cid, 0.0) + semantic_weight / (k + rank)
        chunk_map[cid]      = {**chunk, "retrieval_method": "semantic"}

    # Add BM25 rank contributions — accumulates with semantic if already present
    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + bm25_weight / (k + rank)

        if cid not in chunk_map:
            # BM25-only result — build a compatible chunk dict
            chunk_map[cid] = {
                "text":             chunk["text"],
                "source":           chunk["source"],
                "chunk_id":         cid,
                "retrieval_method": "bm25",
                **( {"page": chunk["page"]} if chunk.get("page") else {} ),
            }
        else:
            # Appeared in both lists — mark as hybrid
            chunk_map[cid]["retrieval_method"] = "hybrid"

    # Attach the final fused score and sort
    results = []
    for cid, rrf_score in fused_scores.items():
        chunk = {**chunk_map[cid], "score": round(rrf_score, 6)}
        results.append(chunk)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── Hybrid retrieval ──────────────────────────────────────────────────────────

def hybrid_retrieve(
    query:      str,
    model,
    client,
    bm25_index: BM25Index,
    domain:     str = config.DEFAULT_DOMAIN,
    top_k:      int = config.RETRIEVAL_TOP_K,
    top_n:      int = config.RERANK_TOP_N,
) -> list[dict]:
    """
    Retrieve relevant chunks using hybrid semantic + BM25 search.

    Pipeline:
        1. Run semantic (vector) search via ChromaDB
        2. Run BM25 (keyword) search via in-memory BM25Index
        3. Fuse both result lists using Reciprocal Rank Fusion
        4. Return the top_n fused results

    The BM25 index is built from all chunks currently in the given
    domain's ChromaDB collection. This means it always stays in sync
    with ChromaDB — no separate index management needed.

    Args:
        query:      User's question string.
        model:      Loaded SentenceTransformer.
        client:     ChromaDB persistent client.
        bm25_index: BM25Index instance for this domain. May be empty
                    (not yet built) — falls back to semantic-only.
        domain:     Domain to search.
        top_k:      Candidates from each retriever before fusion.
        top_n:      Final results returned after fusion.

    Returns:
        List of chunk dicts sorted by fused score, best first.
        Each chunk includes a "retrieval_method" field:
            "semantic" — found by vector search only
            "bm25"     — found by keyword search only
            "hybrid"   — found by both (highest quality signal)
    """
    # Step 1: Semantic retrieval (existing ChromaDB search)
    semantic_chunks = retrieve(query, model, client, domain=domain, top_k=top_k)

    # Step 2: BM25 retrieval (keyword search)
    if bm25_index.is_empty():
       log.warning(
    "BM25 index empty for '%s' — using semantic only.",
    domain,
)
       for chunk in semantic_chunks:
            chunk["retrieval_method"] = "semantic"
            return semantic_chunks[:top_n]

    bm25_chunks = bm25_index.search(query, top_k=top_k)

    # Step 3: Fuse results
    fused = reciprocal_rank_fusion(semantic_chunks, bm25_chunks)

    log.info(
    "Domain '%s' | semantic=%d | bm25=%d | fused=%d | returning=%d",
    domain,
    len(semantic_chunks),
    len(bm25_chunks),
    len(fused),
    min(top_n, len(fused)),
)

    # Log retrieval methods for debug visibility
    log.debug(
    "[%s] score=%.5f | source=%s | preview=%s...",
    chunk["retrieval_method"],
    chunk["score"],
    chunk["source"],
    chunk["text"][:50],
)

    return fused[:top_n]


def hybrid_retrieve_all_domains(
    query:             str,
    model,
    client,
    domain_bm25_map:   dict[str, BM25Index],
    top_n:             int = config.RERANK_TOP_N,
) -> list[dict]:
    """
    Hybrid search across all existing domain collections.

    Each domain is searched independently using its own BM25 index.
    Results from all domains are merged and re-sorted by fused score.

    Args:
        query:           User's question.
        model:           Loaded SentenceTransformer.
        client:          ChromaDB persistent client.
        domain_bm25_map: Dict mapping domain name → BM25Index.
                         Keys are domain names from config.DOMAINS.
        top_n:           Chunks returned per domain before cross-domain merge.

    Returns:
        Merged list of hybrid chunk dicts from all domains, sorted by score.
    """
    all_chunks: list[dict] = []
    domains = list_all_collections(client)

    if not domains:
        log.warning("No domain collections found.")
        return []

    for domain in domains:
        bm25_index = domain_bm25_map.get(domain, BM25Index())
        chunks     = hybrid_retrieve(
            query, model, client, bm25_index, domain=domain, top_n=top_n
        )
        all_chunks.extend(chunks)

    merged = sorted(all_chunks, key=lambda x: x["score"], reverse=True)
    log.info(
    "All-domain: %d chunks from %d domains",
    len(merged),
    len(domains),
)
    return merged[:top_n * 2]


# ── BM25 index builder ────────────────────────────────────────────────────────

def build_bm25_index_for_domain(
    client: chromadb.PersistentClient,
    domain: str = config.DEFAULT_DOMAIN,
) -> BM25Index:
    """
    Build a BM25 index from all chunks currently stored in a domain's
    ChromaDB collection.

    This is called once per domain at startup and after any new
    document is ingested. The BM25 index lives in memory — it is
    rebuilt cheaply from ChromaDB's stored documents rather than
    maintaining a separate persistence layer.

    Args:
        client: ChromaDB persistent client.
        domain: Domain whose collection to index.

    Returns:
        A built BM25Index ready for search(). Returns an empty
        BM25Index if the collection has no documents.
    """
    collection = get_or_create_collection(client, domain)

    if collection.count() == 0:
        log.warning(
    "Domain '%s' is empty — returning empty index.",
    domain,
)
        return BM25Index()

    # Fetch all stored documents and their metadata from ChromaDB
    data      = collection.get(include=["documents", "metadatas"])
    documents = data["documents"]
    metadatas = data["metadatas"]
    ids       = data["ids"]

    chunks = [
        {
            "text":     doc,
            "chunk_id": cid,
            "source":   meta.get("source", "unknown"),
            "page":     meta.get("page"),
        }
        for doc, meta, cid in zip(documents, metadatas, ids)
    ]

    return BM25Index().build(chunks)