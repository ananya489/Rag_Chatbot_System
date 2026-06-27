import streamlit as st
import uuid

import config
import database
from rag.chunker       import load_and_chunk
from rag.embedder      import load_embedder
from rag.retriever     import get_chroma_client, ingest_chunks, retrieve
from rag.prompt_builder import build_prompt
from llm.ollama_llm    import OllamaLLM


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "RAG Chat",
    page_icon  = "🔍",
    layout     = "wide",
)


# ── Cached resource loading ───────────────────────────────────────────────────
# @st.cache_resource runs ONCE per app lifetime (not per rerun).
# This keeps the embedding model and ChromaDB client in memory
# so they aren't reloaded on every user interaction.

@st.cache_resource
def load_resources():
    """
    Load all heavy resources once and cache them for the app lifetime.

    Returns:
        Tuple of (embedding model, ChromaDB client, OllamaLLM instance)
    """
    database.init_db()

    model  = load_embedder()
    client = get_chroma_client()
    llm    = OllamaLLM()

    # Ingest documents on startup if ChromaDB is empty or new docs added
    chunks = load_and_chunk()
    if chunks:
        ingest_chunks(chunks, model, client)

    return model, client, llm


# ── Session state helpers ─────────────────────────────────────────────────────

def init_session_state():
    """
    Initialise all Streamlit session state keys on first run.
    Streamlit reruns this file on every interaction — these checks
    ensure we don't reset state that already exists.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = None     # Active chat session UUID

    if "messages" not in st.session_state:
        st.session_state.messages = []         # Current chat display messages

    if "sessions_list" not in st.session_state:
        st.session_state.sessions_list = []    # For sidebar session list


def start_new_session():
    """Create a new chat session in SQLite and set it as active."""
    session_id = str(uuid.uuid4())
    database.create_session(session_id, title="New chat")
    st.session_state.session_id = session_id
    st.session_state.messages   = []
    st.session_state.sessions_list = database.list_sessions()


def load_session(session_id: str):
    """Switch to an existing session and load its messages."""
    st.session_state.session_id = session_id
    history = database.get_history(session_id, limit=100)
    st.session_state.messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]


# ── Core RAG pipeline ─────────────────────────────────────────────────────────

def process_query(
    query:   str,
    model,
    client,
    llm:     OllamaLLM,
) -> tuple[str, list[dict]]:
    """
    Run the complete RAG pipeline for one user query.

    Steps:
      1. Load chat history from SQLite
      2. Retrieve relevant chunks from ChromaDB
      3. Build the prompt
      4. Call the LLM
      5. Save both turns to SQLite

    Args:
        query:   User's question string.
        model:   Loaded SentenceTransformer.
        client:  ChromaDB persistent client.
        llm:     OllamaLLM instance.

    Returns:
        Tuple of (answer string, list of source chunk dicts)
    """
    session_id = st.session_state.session_id

    # 1. Load recent history for context
    history = database.get_history(session_id)

    # 2. Retrieve relevant chunks
    chunks = retrieve(query, model, client)

    # 3. Build the prompt
    prompt = build_prompt(chunks, history, query)

    # 4. Generate answer (full, non-streaming for now)
    answer = llm.generate(prompt)

    # 5. Persist both turns
    database.save_message(session_id, "user",      query)
    database.save_message(session_id, "assistant", answer,
                          sources=[{"source": c["source"], "score": c["score"]}
                                   for c in chunks])

    # Update session title from first question
    sessions = database.list_sessions()
    current  = next((s for s in sessions if s["session_id"] == session_id), None)
    if current and current["title"] == "New chat":
        title = query[:50] + ("..." if len(query) > 50 else "")
        database.update_session_title(session_id, title)

    st.session_state.sessions_list = database.list_sessions()

    return answer, chunks


# ── UI ────────────────────────────────────────────────────────────────────────

def render_sidebar(model, client, llm):
    """Render the left sidebar: new chat button + session history list."""
    with st.sidebar:
        st.markdown("## RAG Chat")

        if st.button("＋ New chat", use_container_width=True, type="primary"):
            start_new_session()
            st.rerun()

        st.divider()

        # LLM health check
        if llm.health_check():
            st.success(
                f"Ollama · {config.OLLAMA_MODEL}",
                icon="✅"
            )
        else:
            st.error(
                "Ollama not reachable",
                icon="❌"
            )
            st.caption("Run: ollama serve")

        st.divider()
        st.markdown("**Past conversations**")

        sessions = st.session_state.sessions_list or database.list_sessions()

        if not sessions:
            st.caption("No conversations yet.")

        else:
            for session in sessions:

                is_active = (
                    session["session_id"]
                    == st.session_state.session_id
                )

                label = (
                    "▶ "
                    if is_active
                    else ""
                ) + session["title"]


                if st.button(
                    label,
                    key=session["session_id"],
                    use_container_width=True
                ):
                    load_session(session["session_id"])
                    st.rerun()


        st.divider()

        with st.expander("Knowledge base"):

            if client:

                from rag.retriever import get_or_create_collection

                col = get_or_create_collection(client)

                count = col.count()

                st.metric(
                    "Chunks indexed",
                    count
                )

                st.caption(
                    f"Documents: {config.DOCUMENTS_DIR}"
                )
def render_chat(model, client, llm):
    """Render the main chat area."""
    st.title("RAG Chat System")

    # No active session yet — prompt user to start one
    if not st.session_state.session_id:
        st.info("Click **＋ New chat** in the sidebar to start.")
        return

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    query = st.chat_input("Ask something about your documents...")

    if query:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate and stream assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, chunks = process_query(query, model, client, llm)

                    st.markdown(answer)

                    # Show source citations in an expander
                    if chunks:
                        with st.expander(f"Sources ({len(chunks)})"):
                            for chunk in chunks:
                                st.markdown(
                                    f"**{chunk['source']}** "
                                    f"· relevance: `{chunk['score']:.2f}`"
                                )
                                st.caption(chunk["text"][:200] + "...")

                except RuntimeError as e:
                    answer = f"Error: {e}"
                    st.error(answer)

        st.session_state.messages.append({
            "role": "assistant", "content": answer
        })


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    init_session_state()
    model, client, llm = load_resources()
    render_sidebar(model, client, llm)
    render_chat(model, client, llm)


if __name__ == "__main__":
    main()