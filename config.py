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
RETRIEVAL_TOP_K = 5    # Chunks fetched from ChromaDB
RERANK_TOP_N    = 3    # Chunks kept after re-ranking

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