import streamlit as st
import uuid
import time as _time

import config
import database
import analytics

from rag.chunker import load_and_chunk
from rag.embedder import load_embedder
from rag.retriever import (
    get_chroma_client,
    get_or_create_collection,
    ingest_chunks,
    needs_reingestion,
    ingest_uploaded_file,
    list_all_collections,
    hybrid_retrieve,
    hybrid_retrieve_all_domains,
    build_bm25_index_for_domain,
)
from rag.bm25_index import BM25Index
from rag.prompt_builder import build_prompt
from rag.query_expander import expand_query
from llm.ollama_llm import OllamaLLM


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Intelligence Chatbot",
    page_icon="🔍",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource                          # ← single decorator only
def load_resources():
    """
    Load all heavy resources once per app lifetime.

    Startup sequence:
        1. Initialise SQLite tables
        2. Load sentence-transformer embedding model
        3. Connect ChromaDB client
        4. Health-check Ollama — halt with clear error if unreachable
        5. Fingerprint-check documents — re-ingest only if changed
        6. Build BM25 indexes for every populated domain collection

    Returns:
        Tuple of (model, client, llm, bm25_indexes)
        bm25_indexes: dict[domain_name → BM25Index]
    """
    database.init_db()

    model  = load_embedder()
    client = get_chroma_client()

    llm = OllamaLLM()
    if not llm.health_check():
        st.error(
            "⚠️ Cannot reach Ollama. "
            f"Make sure it is running at {config.OLLAMA_BASE_URL} "
            f"and the model is pulled: `ollama pull {config.OLLAMA_MODEL}`",
            icon="🔴",
        )
        st.stop()

    # Fingerprint-based ingestion cache — skip if documents unchanged
    if needs_reingestion(client, str(config.DOCUMENTS_DIR)):
        print("[app] Documents changed — re-ingesting...")
        chunks = load_and_chunk()
        if chunks:
            ingest_chunks(chunks, model, client)
        else:
            print("[app] No documents found in", config.DOCUMENTS_DIR)
    else:
        print("[app] Documents unchanged — skipping ingestion.")

    # Build BM25 index for every domain that already has documents
    bm25_indexes: dict[str, BM25Index] = {}
    for domain in config.DOMAINS:
        collection = get_or_create_collection(client, domain)
        if collection.count() > 0:
            print(f"[app] Building BM25 index for domain: {domain}")
            bm25_indexes[domain] = build_bm25_index_for_domain(client, domain)

    return model, client, llm, bm25_indexes   # ← 4 values


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """
    Initialise all st.session_state keys with safe defaults.
    Streamlit reruns the whole script on every interaction — these
    guards ensure existing state is never overwritten between reruns.
    """
    defaults = {
        "session_id":    None,
        "messages":      [],
        "sessions_list": [],
        "last_chunks":   [],
        "renaming_id":   None,
        "last_uploaded": None,
        "active_domain": config.DEFAULT_DOMAIN,
        "bm25_indexes":  {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_new_session() -> None:
    """Create a new session in SQLite and activate it."""
    sid = str(uuid.uuid4())
    database.create_session(sid, title="New chat")
    st.session_state.session_id    = sid
    st.session_state.messages      = []
    st.session_state.last_chunks   = []
    st.session_state.sessions_list = database.list_sessions()
    st.rerun()


def load_session(sid: str) -> None:
    """Switch to an existing session and restore its message history."""
    st.session_state.session_id  = sid
    st.session_state.last_chunks = []
    history = database.get_history(sid, limit=100)
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]} for m in history
    ]


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_session_markdown(session_id: str, title: str) -> str:
    """Render a full chat session as a Markdown string for download."""
    history = database.get_history(session_id, limit=1000)
    lines   = [f"# {title}", ""]
    for msg in history:
        role = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"{role}: {msg['content']}")
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_query_stream(query: str, model, client, llm: OllamaLLM):
    """
    Execute the full RAG pipeline for one user message, streaming tokens.

    Pipeline (runs exactly once per call):
        1. Load recent chat history from SQLite
        2. Expand vague follow-up queries using history context
        3. Hybrid retrieval — BM25 + semantic via Reciprocal Rank Fusion
        4. Build the structured prompt
        5. Stream tokens from Ollama
        6. Persist both turns to SQLite
        7. Log analytics (response time, source count)
        8. Auto-title the session from the first user message

    Yields:
        str: Individual tokens from the LLM response stream.
    """
    sid          = st.session_state.session_id
    domain       = st.session_state.get("active_domain", config.DEFAULT_DOMAIN)
    bm25_indexes = st.session_state.get("bm25_indexes", {})

    # Step 1 — Chat history
    history = database.get_history(sid)

    # Step 2 — Query expansion
    expanded_query = expand_query(query, history)

    # Step 3 — Hybrid retrieval (single execution path)
    if domain == "All Domains":
        chunks = hybrid_retrieve_all_domains(
            expanded_query, model, client, bm25_indexes
        )
    else:
        bm25_index = bm25_indexes.get(domain, BM25Index())
        chunks     = hybrid_retrieve(
            expanded_query, model, client, bm25_index, domain=domain
        )

    # Step 4 — Prompt assembly
    prompt = build_prompt(chunks, history, query)

    # Step 5 — Stream tokens
    start: float          = _time.time()
    collected_tokens: list[str] = []

    for token in llm.stream(prompt):
        collected_tokens.append(token)
        yield token                         # renders in real time via st.write_stream

    elapsed_ms = int((_time.time() - start) * 1000)
    answer     = "".join(collected_tokens)

    # Step 6 — Persist both turns
    database.save_message(sid, "user", query)
    database.save_message(
        sid, "assistant", answer,
        sources=[
            {
                "source": c["source"],
                "score":  c["score"],
                "domain": c.get("domain", domain),
                "method": c.get("retrieval_method", "semantic"),
                **({"page": c["page"]} if c.get("page") else {}),
            }
            for c in chunks
        ],
    )

    # Step 7 — Analytics
    analytics.log_query(sid, elapsed_ms, chunks)

    # Step 8 — Auto-title from first message
    sessions = database.list_sessions()
    current  = next((s for s in sessions if s["session_id"] == sid), None)
    if current and current["title"] == "New chat":
        database.update_session_title(
            sid, query[:50] + ("..." if len(query) > 50 else "")
        )

    st.session_state.sessions_list = database.list_sessions()
    st.session_state.last_chunks   = chunks


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD WIDGET
# ─────────────────────────────────────────────────────────────────────────────

def render_upload_widget(model, client) -> None:
    """
    Sidebar file uploader — saves and immediately indexes the uploaded file.

    Upload is blocked when 'All Domains' is selected so the file lands
    in the correct domain collection. After a successful ingest the BM25
    index for that domain is rebuilt without requiring an app restart.
    """
    with st.sidebar.expander("📄 Upload Document", expanded=False):

        domain = st.session_state.get("active_domain", config.DEFAULT_DOMAIN)

        if domain == "All Domains":
            st.info("Select a specific domain before uploading.")
            return

        st.caption(f"Uploading to: **{domain}**")

        uploaded_file = st.file_uploader(
            "Add document",
            type=["txt", "md", "py", "csv", "html", "pdf", "docx"],
            key="doc_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.caption("Supported: pdf, docx, txt, md, py, csv, html")
            return

        # Guard: skip re-processing the same file on subsequent reruns
        if st.session_state.get("last_uploaded") == uploaded_file.name:
            return

        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                count = ingest_uploaded_file(
                    uploaded_file,
                    model,
                    client,
                    config.DOCUMENTS_DIR,
                    domain=domain,
                )
                st.session_state.last_uploaded = uploaded_file.name

                if count > 0:
                    # Rebuild BM25 index immediately so hybrid retrieval
                    # reflects the new document without restarting
                    updated_index = build_bm25_index_for_domain(client, domain)
                    bm25_indexes  = st.session_state.get("bm25_indexes", {})
                    bm25_indexes[domain] = updated_index
                    st.session_state.bm25_indexes = bm25_indexes

                    st.success(
                        f"✅ Indexed {count} chunks from "
                        f"{uploaded_file.name} into '{domain}'"
                    )
                else:
                    st.warning("No extractable text found.")

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Upload failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(llm: OllamaLLM, client) -> None:
    """
    Render the left sidebar with:
        - New Chat button
        - Ollama connection status
        - Domain selector with chunk count
        - Session list with rename and delete controls
        - Knowledge base info (chunk count per domain)
        - Last query debug panel (retrieval scores)
        - Analytics dashboard
    """
    with st.sidebar:

        st.title("🔍 RAG Chat")

        if st.button("＋ New Chat", use_container_width=True, type="primary"):
            start_new_session()

        st.divider()

        # ── Ollama status ─────────────────────────────────────────────────
        if llm.health_check():
            st.success(f"Ollama · {config.OLLAMA_MODEL}", icon="✅")
        else:
            st.error("Ollama Offline", icon="❌")
            st.caption(
                f"Run: `ollama serve` → `ollama pull {config.OLLAMA_MODEL}`"
            )

        st.divider()

        # ── Domain selector ───────────────────────────────────────────────
        st.markdown("### Knowledge Domain")

        options  = ["All Domains"] + config.DOMAINS
        selected = st.selectbox(
            "Search in",
            options,
            index=(
                options.index(st.session_state.active_domain)
                if st.session_state.active_domain in options
                else 0
            ),
            label_visibility="collapsed",
        )

        if selected != st.session_state.active_domain:
            st.session_state.active_domain = selected
            st.rerun()

        if selected == "All Domains":
            domains      = list_all_collections(client)
            total_chunks = sum(
                get_or_create_collection(client, d).count()
                for d in config.DOMAINS
            )
            st.caption(f"{total_chunks} chunks across {len(domains)} domain(s)")
        else:
            col = get_or_create_collection(client, selected)
            st.caption(f"{col.count()} chunks in this domain")

        st.divider()

        # ── Session list ──────────────────────────────────────────────────
        st.subheader("Chats")

        sessions   = st.session_state.sessions_list or database.list_sessions()
        active_sid = st.session_state.session_id

        if not sessions:
            st.caption("No conversations yet.")

        for session in sessions:
            is_active = session["session_id"] == active_sid
            label     = ("▶ " if is_active else "") + session["title"]
            cols      = st.columns([0.7, 0.15, 0.15])

            with cols[0]:
                if st.button(
                    label,
                    key=f"open_{session['session_id']}",
                    use_container_width=True,
                ):
                    load_session(session["session_id"])
                    st.rerun()

            with cols[1]:
                if st.button("✏️", key=f"ren_{session['session_id']}"):
                    st.session_state.renaming_id = session["session_id"]
                    st.rerun()

            with cols[2]:
                if st.button("🗑️", key=f"del_{session['session_id']}"):
                    database.delete_session(session["session_id"])
                    if is_active:
                        st.session_state.session_id = None
                        st.session_state.messages   = []
                    st.session_state.sessions_list = database.list_sessions()
                    st.rerun()

            # Inline rename form
            if st.session_state.get("renaming_id") == session["session_id"]:
                new_title = st.text_input(
                    "New name",
                    value=session["title"],
                    key=f"input_{session['session_id']}",
                    label_visibility="collapsed",
                )
                rcols = st.columns([0.5, 0.5])
                with rcols[0]:
                    if st.button("Save", key=f"save_{session['session_id']}"):
                        database.update_session_title(
                            session["session_id"], new_title
                        )
                        st.session_state.renaming_id   = None
                        st.session_state.sessions_list = database.list_sessions()
                        st.rerun()
                with rcols[1]:
                    if st.button("Cancel", key=f"cancel_{session['session_id']}"):
                        st.session_state.renaming_id = None
                        st.rerun()

        st.divider()

        # ── Knowledge base info ───────────────────────────────────────────
        with st.expander("📚 Knowledge Base"):
            if selected == "All Domains":
                total = sum(
                    get_or_create_collection(client, d).count()
                    for d in config.DOMAINS
                )
                st.metric("Total Chunks", total)
            else:
                col = get_or_create_collection(client, selected)
                st.metric("Chunks", col.count())

                if col.count() > 0:
                    all_meta = col.get()["metadatas"]
                    sources  = sorted({m["source"] for m in all_meta})
                    st.caption(f"{len(sources)} file(s) indexed:")
                    for src in sources:
                        st.caption(f"• {src}")

        # ── Last query debug panel ────────────────────────────────────────
        with st.expander("🔬 Last Query Debug"):
            chunks = st.session_state.last_chunks
            if chunks:
                for i, chunk in enumerate(chunks):
                    method = chunk.get("retrieval_method", "semantic")
                    icon   = {
                        "hybrid":   "🔀",
                        "bm25":     "🔤",
                        "semantic": "🔍",
                    }.get(method, "🔍")
                    st.progress(
                        min(chunk["score"] * 60, 1.0),
                        text=(
                            f"{icon} #{i + 1} · {chunk['source']} "
                            f"· {chunk['score']:.5f} · {method}"
                        ),
                    )
            else:
                st.caption("Ask a question to see retrieval scores.")

        # ── Analytics ─────────────────────────────────────────────────────
        with st.expander("📊 Analytics"):
            stats = analytics.get_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", stats["total_queries"])
            with col2:
                st.metric("Avg Response", f"{stats['avg_response_ms']} ms")
            if stats["top_sources"]:
                st.caption("Top Sources")
                for name, count in stats["top_sources"]:
                    st.caption(f"• {name} ({count}×)")


# ─────────────────────────────────────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────────────────────────────────────

def render_chat(model, client, llm: OllamaLLM) -> None:
    """
    Render the main chat panel with:
        - Export button for the active session
        - Full message history
        - Chat input box
        - Streaming assistant response with sources expander
    """
    st.title("📄 Document Intelligence Chatbot")

    if not st.session_state.session_id:
        st.info("👈 Click **＋ New Chat** in the sidebar to begin.")
        return

    # ── Export button ─────────────────────────────────────────────────────
    current_title = next(
        (
            s["title"]
            for s in database.list_sessions()
            if s["session_id"] == st.session_state.session_id
        ),
        "chat",
    )
    md = export_session_markdown(st.session_state.session_id, current_title)
    st.download_button(
        "⬇️ Export Conversation",
        data=md,
        file_name=f"{current_title[:30].replace(' ', '_')}.md",
        mime="text/markdown",
    )

    st.divider()

    # ── Message history ───────────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # ── Chat input ────────────────────────────────────────────────────────
    query = st.chat_input("Ask something about your documents...")

    if not query:
        return

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # ── Stream assistant response ─────────────────────────────────────────
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                process_query_stream(query, model, client, llm)
            )

            # Sources expander with retrieval method icons
            chunks = st.session_state.last_chunks
            if chunks:
                with st.expander(f"📚 Sources ({len(chunks)})"):
                    for chunk in chunks:
                        page_info   = f" · Page {chunk['page']}" if chunk.get("page") else ""
                        method      = chunk.get("retrieval_method", "semantic")
                        method_icon = {
                            "hybrid":   "🔀",
                            "bm25":     "🔤",
                            "semantic": "🔍",
                        }.get(method, "🔍")
                        domain_label = chunk.get(
                            "domain", st.session_state.active_domain
                        )
                        st.markdown(
                            f"{method_icon} **{chunk['source']}**{page_info}\n\n"
                            f"Domain: **{domain_label}** · "
                            f"Relevance: `{chunk['score']:.5f}` · `{method}`"
                        )
                        st.caption(chunk["text"][:250] + "...")

        except Exception as e:
            answer = f"Error: {e}"
            st.error(answer)
            st.caption("Check Ollama: `ollama serve`")

    st.session_state.messages.append({"role": "assistant", "content": answer})


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Application entry point.

    Order:
        1. Initialise session state
        2. Load cached resources (model, ChromaDB, Ollama, BM25 indexes)
        3. Merge BM25 indexes into session state without overwriting
           any indexes rebuilt mid-session after a file upload
        4. Render sidebar
        5. Render upload widget
        6. Render chat panel
    """
    init_session_state()

    model, client, llm, bm25_indexes = load_resources()   # ← unpack 4 values

    # Merge startup indexes into session state.
    # Mid-session rebuilds (after upload) must not be overwritten.
    existing = st.session_state.get("bm25_indexes", {})
    for domain, index in bm25_indexes.items():
        if domain not in existing:
            existing[domain] = index
    st.session_state.bm25_indexes = existing

    render_sidebar(llm, client)
    render_upload_widget(model, client)
    render_chat(model, client, llm)


if __name__ == "__main__":
    main()