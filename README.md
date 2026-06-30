# RAG Chatbot System — Document Intelligence Chatbot

![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-1.35-red?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

A Retrieval-Augmented Generation (RAG) chatbot that turns any collection of documents into a queryable knowledge base. Upload PDFs, markdown, or text files, ask questions in natural language, and get answers grounded strictly in your own content — with source citations, persistent chat history, and a live analytics dashboard.

---

## Overview

Large language models are powerful but limited to what they were trained on — they cannot answer questions about your private documents, internal reports, or recent research, and they tend to hallucinate when they don't know something.

**RAG Chatbot System** solves this by combining semantic search with local LLM generation. Documents are split into chunks, converted into vector embeddings, and stored in ChromaDB. When a user asks a question, the most relevant chunks are retrieved and passed to the LLM as context — so every answer is grounded in real, retrievable evidence rather than the model's memory.

The system runs entirely locally using Ollama for inference, meaning no API keys, no per-token costs, and no data ever leaving your machine.

---

## Features

- **Multi-format document ingestion** — PDF, TXT, MD, PY, CSV, and HTML files supported out of the box
- **Automatic chunking and embedding** — documents are split, embedded, and indexed without manual configuration
- **Semantic retrieval with re-ranking** — ChromaDB cosine similarity search filtered by relevance threshold
- **Query expansion** — vague follow-up questions are resolved using conversation context before retrieval
- **Strict grounding prompt engineering** — answers are generated only from retrieved context, reducing hallucination
- **Source-aware responses** — every answer cites the document (and page number, for PDFs) it came from
- **Persistent multi-session chat memory** — conversations survive app restarts via SQLite
- **Full session management** — create, rename, and delete conversations from the UI
- **In-app document upload** — add files directly through the Streamlit sidebar, indexed immediately
- **Analytics dashboard** — tracks total queries, average response time, and most-referenced sources
- **Conversation export** — download any chat as a Markdown file
- **Streaming responses** — answers appear token by token in real time
- **Containerized deployment** — Docker and Docker Compose configuration included

---

## Architecture

```
                    User
                     │
                     ▼
              Streamlit UI
                     │
                     ▼
              RAG Pipeline
                     │
                     ▼
          Query Expansion + Retriever
                     │
                     ▼
                ChromaDB
          (vector similarity search)
                     │
                     ▼
             Prompt Builder
     (context + chat history + question)
                     │
                     ▼
              Ollama LLM
              (qwen2.5:7b)
                     │
                     ▼
                 Response
          (streamed + source citations)
```

Chat history and session metadata are persisted independently in SQLite, read and written on every turn so conversations remain available across restarts.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11 |
| Frontend | Streamlit |
| LLM | Ollama — qwen2.5:7b |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| Persistent Storage | SQLite |
| Deployment | Docker, Docker Compose |

---

## Project Structure

```
rag_chatbot_system/
│
├── app.py                   # Streamlit entry point — UI and orchestration
├── config.py                # Centralized configuration and environment loading
├── database.py               # SQLite session and chat history management
├── analytics.py              # Query analytics and usage tracking
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── rag/
│   ├── chunker.py            # Document loading and chunking (PDF + text)
│   ├── embedder.py           # Sentence-transformer embedding generation
│   ├── retriever.py          # ChromaDB indexing and similarity retrieval
│   ├── prompt_builder.py     # Context-aware prompt construction
│   └── query_expander.py     # Follow-up query resolution
│
├── llm/
│   └── ollama_llm.py         # Ollama LLM integration (streaming + generate)
│
└── data/
    └── documents/             # Knowledge base source files
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.com) installed

### Setup

```bash
# Clone the repository
git clone https://github.com/ananya489/Rag_Chatbot_System.git
cd Rag_Chatbot_System

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running Locally

**1. Start Ollama and pull the model**

```bash
ollama serve
ollama pull qwen2.5:7b
```

**2. Launch the application**

```bash
streamlit run app.py
```

**3. Open your browser**

Navigate to `http://localhost:8501`

---

## Docker Deployment

The project includes a full container setup with separate services for the application and the Ollama LLM backend.

```bash
docker compose up --build
```

This starts two containers:

- **`ollama`** — runs the Ollama inference server, exposing port `11434`. Model weights persist in a named volume across restarts.
- **`app`** — runs the Streamlit application on port `8501`, connected to the Ollama container over the internal Docker network.

After the containers are running, pull the model into the Ollama container once:

```bash
docker exec -it <ollama_container_name> ollama pull qwen2.5:7b
```

Then open `http://localhost:8501` in your browser.

---

## How to Use

1. **Upload a document** — open the "Upload document" panel in the sidebar and add a PDF, TXT, MD, PY, CSV, or HTML file
2. **Wait for indexing** — the file is chunked, embedded, and stored in ChromaDB automatically; a confirmation message appears once complete
3. **Ask questions** — type a question about your document in the chat box and receive a streamed, grounded answer
4. **View sources** — expand the "Sources" panel under any answer to see which document chunks (and page numbers, for PDFs) were used to generate it

## Example Questions

Once a document is indexed, try asking:

- "What is this document about?"
- "Summarize this paper."
- "Explain the important points."
- "What does it say about [specific topic]?"
- "Can you elaborate on the second point you mentioned?"

---

## Future Improvements

- [ ] User authentication and access control
- [ ] Cloud deployment (AWS / GCP / Azure)
- [ ] Improved retrieval re-ranking with cross-encoder models
- [ ] Multi-user support with isolated knowledge bases
- [ ] Hybrid search combining keyword and semantic retrieval
- [ ] OCR support for scanned PDFs

---

## License

This project is licensed under the [MIT License](LICENSE).
