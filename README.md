# RAG Chatbot System

> A locally-hosted, privacy-first Retrieval-Augmented Generation chat system. Ask questions about your own documents and receive answers grounded strictly in retrieved evidence — with hybrid BM25 + semantic search, multi-domain knowledge bases, persistent conversation memory, and streaming responses.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Hybrid Retrieval Pipeline](#hybrid-retrieval-pipeline)
- [Conversation Memory](#conversation-memory)
- [Multi-Domain Knowledge Base](#multi-domain-knowledge-base)
- [Logging](#logging)
- [Testing](#testing)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

Standard LLMs answer from training data alone — they hallucinate when asked about private documents, internal reports, or domain-specific knowledge they were never trained on.

**RAG Chatbot System** grounds every answer in your actual documents. Files are chunked, embedded into vectors, and stored in ChromaDB. When you ask a question, the system runs **hybrid BM25 + vector retrieval** — combining keyword precision with semantic understanding — and passes the most relevant chunks to a local Ollama LLM as context.

Everything runs locally. No API keys. No cloud. No data leaves your machine.

---

## Features

**Document Ingestion**
- Supports PDF, DOCX, TXT, MD, PY, CSV, and HTML
- Recursive text splitting with configurable chunk size and overlap
- Page-aware chunking for PDFs — page numbers preserved in metadata
- In-app drag-and-drop upload via Streamlit sidebar
- Fingerprint-based ingestion cache — skips re-embedding unchanged documents on restart

**Retrieval**
- Hybrid BM25 + ChromaDB vector search fused via Reciprocal Rank Fusion
- Per-query BM25 scoring with IDF-weighted term frequency
- Cosine similarity retrieval from ChromaDB
- Re-ranking with configurable similarity threshold
- Query expansion resolves vague follow-up questions before retrieval

**Multi-Domain Knowledge Base**
- Isolated ChromaDB collection per domain (General, HR, Finance, Legal, Education, Customer Support)
- Search within one domain or across all domains simultaneously
- Upload documents into specific domains from the UI

**Conversation Memory**
- Two-layer memory: LLM-generated summary of older turns + verbatim recent turns
- Summarization triggers after a configurable turn threshold
- Summaries cached in SQLite — regenerated only when stale
- Context window stays bounded regardless of conversation length

**Chat Management**
- Persistent multi-session history via SQLite
- Create, rename, and delete sessions from the sidebar
- Session titles auto-generated from first user message
- Export any conversation as a Markdown file

**Generation**
- Streaming responses token-by-token via `st.write_stream`
- Strict grounding prompt — answers only from retrieved context
- Source citations with filename, relevance score, page number, and retrieval method
- Retrieval method icon per source: 🔀 hybrid · 🔤 BM25 · 🔍 semantic

**Observability**
- Analytics dashboard: total queries, average response time, top cited sources
- Last-query debug panel: per-chunk retrieval scores as progress bars
- Structured logging to terminal (INFO+) and `logs/rag_chat.log` (DEBUG+)

**Deployment**
- Docker + Docker Compose with separate Ollama and Streamlit containers
- Pluggable LLM layer — swap providers by implementing one abstract interface

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                         │
│                                                               │
│  Documents → Chunker → Metadata tag → Embedder → ChromaDB    │
│  (fingerprint cache skips unchanged files on restart)         │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                            │
│                                                               │
│  User question                                                │
│       │                                                       │
│       ▼                                                       │
│  ConversationMemory                                           │
│  (summary of old turns + recent turns verbatim)               │
│       │                                                       │
│       ▼                                                       │
│  Query Expander                                               │
│  (resolves vague follow-ups using recent history)             │
│       │                          │                            │
│       ▼                          ▼                            │
│  ChromaDB (semantic)        BM25Index (keyword)               │
│       │                          │                            │
│       └──────────────┬───────────┘                           │
│                       ▼                                       │
│             Reciprocal Rank Fusion                            │
│             (scale-independent merge)                         │
│                       │                                       │
│                       ▼                                       │
│             Prompt Builder                                    │
│    [system · context · summary · recent · question]           │
│                       │                                       │
│                       ▼                                       │
│             Ollama LLM (stream)                               │
│                       │                                       │
│          ┌────────────┴────────────┐                         │
│          ▼                         ▼                          │
│     SQLite (persist)          Streamlit UI                    │
│     Analytics log             (token-by-token render)         │
└──────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| UI | Streamlit 1.35 |
| LLM | Ollama — qwen2.5:7b (local, no API key) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-dim) |
| Vector search | ChromaDB 0.5 — cosine similarity, HNSW index |
| Keyword search | rank-bm25 — BM25 with IDF weighting |
| Fusion | Reciprocal Rank Fusion (Cormack et al., 2009) |
| Persistent storage | SQLite (sessions, messages, summaries, analytics) |
| PDF ingestion | pypdf |
| DOCX ingestion | python-docx |
| Deployment | Docker + Docker Compose |
| Language | Python 3.11 |

---

## Folder Structure

```
rag_chatbot_system/
│
├── app.py                    # Streamlit entry point — orchestrates all modules
├── config.py                 # All settings, paths, model names, domain list
├── database.py               # SQLite CRUD — sessions, messages, summaries
├── analytics.py              # Query analytics — response time, top sources
├── logger.py                 # Centralized logging — console + file handlers
│
├── rag/
│   ├── chunker.py            # Document loading (PDF, DOCX, text) and chunking
│   ├── embedder.py           # sentence-transformers wrapper (single + batch)
│   ├── retriever.py          # ChromaDB client, ingestion, cosine retrieval
│   ├── bm25_index.py         # In-memory BM25 index — tokenize, build, search
│   ├── prompt_builder.py     # Assembles context + summary + history + query
│   ├── query_expander.py     # Resolves vague follow-ups using history context
│   └── memory.py             # Two-layer conversation memory manager
│
├── llm/
│   ├── base.py               # Abstract BaseLLM interface (generate + stream)
│   └── ollama_llm.py         # Concrete Ollama implementation
│
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py       # 18 tests — clean_text, split_text, load_and_chunk
│   ├── test_bm25.py          # 14 tests — tokenize, build, search, scoring
│   └── test_prompt_builder.py # 14 tests — sections, ordering, summary handling
│
├── data/
│   └── documents/            # Drop knowledge base files here
│
├── logs/
│   └── rag_chat.log          # Auto-generated — DEBUG level, persistent
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed

### Clone and install

```bash
git clone https://github.com/ananya489/Rag_Chatbot_System.git
cd Rag_Chatbot_System

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # for running tests
```

---

## Configuration

Copy the example environment file and edit as needed:

```bash
cp .env.example .env
```

`.env.example` contents:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024
```

All other settings (chunk size, retrieval top-K, memory thresholds, domain list) live in `config.py` with inline documentation.

---

## Running the Project

**1. Start Ollama**

```bash
ollama serve
ollama pull qwen2.5:7b
```

**2. Add documents** (optional — you can also upload from the UI)

```bash
cp your_file.pdf data/documents/
```

**3. Start the app**

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

### Docker

```bash
docker compose up --build
docker exec -it <ollama_container> ollama pull qwen2.5:7b
```

Open `http://localhost:8501`.

---

## Hybrid Retrieval Pipeline

Standard RAG uses only vector similarity. This project combines two retrieval strategies and fuses them using Reciprocal Rank Fusion:

**Semantic search (ChromaDB)**
Documents and queries are embedded into 384-dimensional vectors using `all-MiniLM-L6-v2`. ChromaDB returns the top-K nearest chunks by cosine similarity. This finds semantically similar content even when the wording differs from the query.

**BM25 keyword search**
An in-memory BM25 index is built from all chunks at startup. BM25 scores documents using TF-IDF weighting with document length normalization. This finds exact keyword and rare-term matches that semantic search misses.

**Reciprocal Rank Fusion**
Both result lists are merged by rank position (not raw score) using the formula:

```
rrf_score = semantic_weight / (k + rank_s) + bm25_weight / (k + rank_b)
```

Where `k=60` (Cormack et al., 2009 standard). This is scale-independent — cosine similarities and BM25 scores are never compared directly. Each retrieved chunk carries a `retrieval_method` field (`hybrid`, `bm25`, or `semantic`) visible in the sources expander.

---

## Conversation Memory

Long conversations overflow the LLM context window if full history is sent every turn. This project implements a two-layer memory system:

**Layer 1 — Compression (older turns)**
When a session exceeds `SUMMARY_THRESHOLD` turns, all messages older than the recency window are compressed into a single summary paragraph by the LLM. The summary is stored in SQLite and reused on subsequent turns until new messages push more content beyond the boundary.

**Layer 2 — Verbatim (recent turns)**
The most recent `RECENT_TURNS_KEEP` turns (configurable, default 3) are always kept verbatim. These provide the immediate conversational context the LLM needs for follow-up questions.

**Prompt structure:**
```
[System + grounding] → [Context chunks] → [Summary] → [Recent turns] → [Question] → Answer:
```

---

## Multi-Domain Knowledge Base

Each domain maps to an isolated ChromaDB collection:

| Domain | Collection name |
|---|---|
| General | `domain_general` |
| HR | `domain_hr` |
| Finance | `domain_finance` |
| Legal | `domain_legal` |
| Education | `domain_education` |
| Customer Support | `domain_customer_support` |

Select a domain in the sidebar before uploading. To search across all domains simultaneously, select "All Domains" — results from every collection are merged and re-sorted by RRF score.

---

## Logging

All modules use Python's `logging` module via a central `logger.py` configuration:

- **Console (stdout):** INFO and above — clean, timestamped output during development
- **File (`logs/rag_chat.log`):** DEBUG and above — full detail for post-mortem debugging

Log format:
```
2025-07-05 14:23:11 | INFO     | rag_chat.rag.retriever | 'General' | fetched=5 | returned=3
2025-07-05 14:23:11 | INFO     | rag_chat.rag.prompt_builder | chunks=3 | recent_turns=4 | has_summary=yes
```

No `print()` calls exist in the codebase — every log line carries a timestamp, severity, and module name.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=rag --cov-report=term-missing

# Single module
pytest tests/test_bm25.py -v
```

Test coverage:

| File | Tests | What is covered |
|---|---|---|
| `test_chunker.py` | 18 | `clean_text`, `split_text`, `load_text_document`, `load_and_chunk` |
| `test_bm25.py` | 14 | `tokenize`, `BM25Index.build`, `BM25Index.search`, scoring correctness |
| `test_prompt_builder.py` | 14 | Section ordering, summary inclusion, grounding instructions |


## Future Improvements

- [ ] OCR support for scanned PDFs via pytesseract
- [ ] Cross-encoder re-ranking for higher retrieval precision
- [ ] User authentication and per-user knowledge bases
- [ ] Cloud deployment guide (AWS / GCP / Render)
- [ ] Hybrid search weight tuning via UI slider
- [ ] Streaming summarization for very long sessions

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit with a conventional commit message: `git commit -m "feat: add X"`
4. Push and open a pull request

Please run `pytest tests/` before submitting a PR.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Ananya** · [@ananya489](https://github.com/ananya489)

Built as a B.Tech CSE internship evaluation project, incrementally developed into a production-grade RAG system covering hybrid retrieval, vector databases, local LLM integration, conversation memory management, and software engineering best practices.
