# RAG Chat System

> A production-grade, locally-hosted Retrieval-Augmented Generation (RAG) chat system built with Streamlit, ChromaDB, SQLite, and Ollama. Ask questions about your own documents and get accurate, cited answers — with full conversation memory across sessions.

---

## Overview

Most LLMs answer from training data alone — they hallucinate when asked about your specific documents, internal knowledge, or recent information. **RAG Chat System** solves this by grounding every answer in your actual documents.

The pipeline works in two stages:

**Ingestion (once):** Your documents are split into overlapping chunks → converted to 384-dimensional semantic vectors via `all-MiniLM-L6-v2` → stored in ChromaDB on disk.

**Query (every message):** Your question is embedded with the same model → the most relevant chunks are retrieved via cosine similarity → re-ranked by relevance score → combined with your conversation history into a structured prompt → sent to a local Ollama LLM → streamed back token by token.

Everything runs locally. No API keys. No cloud. No data leaves your machine.

---

## Features

- **Semantic search** — finds relevant content even when wording differs from the document
- **Streaming responses** — tokens appear in real time, no waiting for full generation
- **Persistent chat memory** — conversations saved in SQLite, accessible across restarts
- **Multi-session support** — maintain separate conversations, switch between them via sidebar
- **Query expansion** — vague follow-up questions resolved using conversation context before retrieval
- **Source citations** — every answer shows which document chunks were used and their relevance score
- **Re-ranking** — top-K candidates filtered by similarity threshold before prompting the LLM
- **Pluggable LLM layer** — swap Ollama for any provider by implementing one abstract interface
- **Zero external dependencies** — runs entirely offline after initial model downloads

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│  Documents → Chunker → Metadata tag → Embedder → ChromaDB       │
│  (runs once on startup, skips unchanged documents)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                           │
│                                                                   │
│  User question                                                    │
│       │                                                           │
│       ▼                                                           │
│  Query Expander ──► Embedder ──► ChromaDB (top-K retrieval)      │
│                                       │                           │
│                                       ▼                           │
│                                  Re-ranker                        │
│                                       │                           │
│  SQLite (history) ────────────────────┤                           │
│                                       ▼                           │
│                               Prompt Builder                      │
│                          [system + context + history + query]     │
│                                       │                           │
│                                       ▼                           │
│                              Ollama LLM (stream)                  │
│                                       │                           │
│                    ┌──────────────────┤                           │
│                    │                  ▼                           │
│               SQLite (save)     Streamlit UI                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| UI | Streamlit 1.35 | Chat interface, session sidebar, streaming display |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | 384-dim semantic vectors |
| Vector store | ChromaDB 0.5 | Persistent cosine similarity search |
| Chat memory | SQLite (built-in) | Session and message persistence |
| LLM | Ollama (`qwen2.5:7b`) | Local inference, streaming |
| Language | Python 3.10+ | — |

---

## Folder Structure

```
rag-chat-system/
│
├── app.py                      ← Streamlit entry point — run this
├── config.py                   ← All settings, paths, model names
├── database.py                 ← SQLite session and message CRUD
│
├── rag/
│   ├── chunker.py              ← Document loading and recursive splitting
│   ├── embedder.py             ← sentence-transformers wrapper (single + batch)
│   ├── retriever.py            ← ChromaDB queries, cosine similarity, re-ranking
│   ├── prompt_builder.py       ← Assembles context + history + query into prompt
│   └── query_expander.py       ← Resolves vague follow-ups using history context
│
├── llm/
│   ├── base.py                 ← Abstract BaseLLM interface (generate + stream)
│   └── ollama_llm.py           ← Concrete Ollama implementation
│
├── data/
│   ├── documents/              ← Drop your knowledge base files here
│   └── chat.db                 ← SQLite database (auto-created)
│
├── chroma_db/                  ← ChromaDB vector store (auto-created)
│
├── test_chunk.py               ← Chunker unit tests
├── test_llm.py                 ← LLM connectivity tests
├── test_retrieval.py           ← Retrieval pipeline tests
│
├── .env                        ← Local config overrides (not committed)
├── .env.example                ← Template for required environment variables
├── .gitignore
└── requirements.txt
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) installed and running

### 1. Clone the repository

```bash
git clone https://github.com/ananya489/Rag_Chatbot_System.git
cd Rag_Chatbot_System
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if you want to change the model or Ollama URL
```

---

## Ollama Setup

Install Ollama (if not already installed):

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from https://ollama.com/download
```

Pull the model:

```bash
ollama pull qwen2.5:7b
```

Start the Ollama server (it may start automatically after installation):

```bash
ollama serve
```

Verify it's working:

```bash
ollama run qwen2.5:7b "Hello, are you working?"
```

Ollama runs at `http://localhost:11434` by default. The app will show a green status indicator in the sidebar when it can reach it.

**Alternative lightweight models:**

```bash
ollama pull llama3.2:3b     # Faster, smaller (2GB)
ollama pull phi3:mini        # Good quality, efficient (2.3GB)
```

To use a different model, update `OLLAMA_MODEL` in your `.env` file.

---

## Adding Documents

Drop any supported files into the `data/documents/` folder before starting the app:

```
data/documents/
├── company_handbook.txt
├── research_paper.md
├── api_documentation.md
└── notes.txt
```

Supported formats: `.txt` `.md` `.py` `.csv` `.html`

The app ingests all documents automatically on startup. Re-ingest happens only when files change.

---

## Running the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

**First run workflow:**
1. Ollama server must be running
2. App loads embedding model (~90MB, cached after first download)
3. Documents in `data/documents/` are chunked and indexed into ChromaDB
4. Click **＋ New chat** in the sidebar to start a conversation
5. Ask questions about your documents

---

## ChromaDB Setup

ChromaDB requires no separate installation or server. It runs as an embedded library and stores vector data in the `chroma_db/` folder automatically. The folder is created on first run.

To reset the vector store (e.g. after changing documents significantly):

```bash
rm -rf chroma_db/
```

The app will re-ingest all documents on the next startup.

---

## Supported File Types

| Extension | Description |
|---|---|
| `.txt` | Plain text files |
| `.md` | Markdown documents |
| `.py` | Python source files |
| `.csv` | Comma-separated data |
| `.html` | HTML pages (tags stripped) |

PDF support is on the roadmap. To add a PDF manually, convert it first:

```bash
# Using pdftotext (Linux/macOS)
pdftotext your_file.pdf your_file.txt
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024
```

All variables have sensible defaults — you only need to set what you want to override.

---

## Screenshots

> _Add screenshots here after running the app_

| Chat Interface | Session History | Source Citations |
|---|---|---|
| `screenshots/chat.png` | `screenshots/sessions.png` | `screenshots/sources.png` |

---

## How the RAG Pipeline Works

**1. Chunking** — Documents are split into overlapping 500-character chunks with 50-character overlap. Overlap ensures sentences that cross chunk boundaries appear in both chunks, preserving context.

**2. Embedding** — Each chunk is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`. Semantically similar text produces vectors that are geometrically close, regardless of exact wording.

**3. Storage** — Vectors, text, and metadata (source filename, chunk ID) are stored in ChromaDB using the HNSW index with cosine distance metric.

**4. Query expansion** — Short or vague follow-up questions ("what about the second type?") are expanded using the previous assistant response before retrieval, improving relevance.

**5. Retrieval** — The query is embedded with the same model, then ChromaDB returns the top-5 nearest vectors by cosine similarity. Results below a 0.3 similarity threshold are discarded.

**6. Re-ranking** — The top-3 chunks after threshold filtering are passed to the prompt builder.

**7. Prompt assembly** — A structured prompt is built: system role → grounding instruction → context chunks with source labels → conversation history → current question → "Answer:".

**8. Generation** — The prompt is sent to Ollama, which streams tokens back. Streamlit renders each token as it arrives.

**9. Persistence** — Both the user message and assistant answer are saved to SQLite with timestamps and source metadata.

---

## Plugging In a Different LLM

The LLM layer is fully abstracted. To add a new provider:

1. Create `llm/your_provider.py`
2. Subclass `BaseLLM` from `llm/base.py`
3. Implement `generate(prompt: str) -> str` and `stream(prompt: str) -> Generator`
4. Update `OLLAMA_MODEL` or add a new config key in `config.py`
5. Change the import in `app.py` — nothing else in the codebase changes

```python
# Example: adding Groq support
from llm.base import BaseLLM

class GroqLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        # your implementation
        ...

    def stream(self, prompt: str):
        # your implementation
        ...
```

---

## Future Improvements

- [ ] PDF ingestion support via `pypdf`
- [ ] In-app document upload (drag and drop in Streamlit UI)
- [ ] Ingestion fingerprint cache (skip re-embedding unchanged documents)
- [ ] Analytics dashboard (query count, average response time, top sources)
- [ ] Delete / rename sessions from the UI
- [ ] Multi-document knowledge base selection
- [ ] Hybrid search (BM25 + vector similarity)
- [ ] Docker deployment with `docker-compose.yml`
- [ ] Export conversation as PDF or Markdown

---

## Scalability Notes

The current architecture is designed for single-user local deployment. For production scaling:

- **Vector store:** Replace ChromaDB with Qdrant or Weaviate for multi-user concurrent access
- **Database:** Migrate SQLite to PostgreSQL for concurrent writes
- **LLM:** Switch `OllamaLLM` to `GroqLLM` or `OpenAILLM` for higher throughput
- **Embedding:** Use a GPU-accelerated embedding server for faster batch ingestion
- **Deployment:** Containerise with Docker, deploy behind nginx with `--server.headless true`

The LLM abstraction layer and modular RAG package make all of these migrations possible without touching application logic.

---

## Running Tests

```bash
# Test chunking pipeline
python test_chunk.py

# Test Ollama connectivity and generation
python test_llm.py

# Test ChromaDB retrieval pipeline
python test_retrieval.py
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built by [@ananya489](https://github.com/ananya489) as part of a B.Tech CSE internship evaluation project.

**Stack:** Python · Streamlit · ChromaDB · sentence-transformers · SQLite · Ollama
