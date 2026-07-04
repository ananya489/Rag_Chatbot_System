import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
DOCUMENTS_DIR   = DATA_DIR / "documents"
DB_PATH         = DATA_DIR / "chat.db"
CHROMA_PATH     = ROOT_DIR / "chroma_db"

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384          # Fixed output size for this model

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500    # Max characters per chunk
CHUNK_OVERLAP = 50     # Characters shared between adjacent chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 8    # Chunks fetched from ChromaDB
RERANK_TOP_N    = 5    # Chunks kept after re-ranking

# ── Chat memory ───────────────────────────────────────────────────────────────
HISTORY_LIMIT = 10     # Max past messages loaded into prompt

# ── LLM ───────────────────────────────────────────────────────────────────────
# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1024"))
# ── Ensure required directories exist ─────────────────────────────────────────
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Multi-domain knowledge base ───────────────────────────────────────────────

# Available domains — each maps to its own ChromaDB collection.
# Add or remove domains here without touching any other file.
DOMAINS: list[str] = [
    "General",
    "Customer Support",
    "Finance",
    "Legal",
    "HR",
    "Education",
]

# The domain selected when the app first loads.
DEFAULT_DOMAIN: str = "General"

# Internal prefix for ChromaDB collection names.
# "General" → "domain_general", "Customer Support" → "domain_customer_support"
COLLECTION_PREFIX: str = "domain"


def domain_to_collection_name(domain: str) -> str:
    """
    Convert a human-readable domain name to a valid ChromaDB collection name.

    ChromaDB collection names must be lowercase alphanumeric with underscores.
    Spaces and special characters are replaced with underscores.

    Examples:
        "General"          → "domain_general"
        "Customer Support" → "domain_customer_support"
        "HR"               → "domain_hr"
    """
    sanitised = domain.lower().strip().replace(" ", "_")
    # Remove any character that isn't alphanumeric or underscore
    sanitised = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitised)
    return f"{COLLECTION_PREFIX}_{sanitised}"

# ── Hybrid retrieval ──────────────────────────────────────────────────────────

# Weight given to semantic (vector) search score in RRF fusion.
# Must sum to 1.0 with BM25_WEIGHT.
SEMANTIC_WEIGHT: float = 0.6

# Weight given to BM25 (keyword) search score in RRF fusion.
# 0.4 gives keyword search meaningful influence without overriding semantics.
BM25_WEIGHT: float = 0.4

# Smoothing constant for Reciprocal Rank Fusion.
# k=60 is the standard value from the original RRF paper (Cormack et al. 2009).
# Higher k reduces the advantage of top-ranked results.
RRF_K: int = 60