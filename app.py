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
    retrieve,
    retrieve_all_domains,
    needs_reingestion,
    save_uploaded_file,
    ingest_uploaded_file,
    list_all_collections,
)

from rag.prompt_builder import build_prompt
from rag.query_expander import expand_query

from llm.ollama_llm import OllamaLLM


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Document Intelligence Chatbot",
    page_icon="🔍",
    layout="wide",
)


# -------------------------------------------------
# LOAD RESOURCES
# -------------------------------------------------

@st.cache_resource
def load_resources():

    database.init_db()

    print("[app] Loading embedding model")
    model = load_embedder()

    print("[app] Connecting ChromaDB")
    client = get_chroma_client()

    print("[app] Checking Ollama")
    llm = OllamaLLM()

    if not llm.health_check():
        st.warning(
            f"Ollama not running.\n\n"
            f"Run:\n"
            f"ollama serve\n\n"
            f"Then:\n"
            f"ollama pull {config.OLLAMA_MODEL}"
        )

    # Re-ingest default domain if needed
    if needs_reingestion(
        client,
        str(config.DOCUMENTS_DIR),
        domain=config.DEFAULT_DOMAIN,
    ):

        print("[app] Re-ingesting documents...")

        chunks = load_and_chunk()

        if chunks:
            ingest_chunks(
                chunks,
                model,
                client,
                domain=config.DEFAULT_DOMAIN,
            )

    else:
        print("[app] Using existing vector DB")

    return model, client, llm


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

def init_session_state():

    defaults = {

        "session_id": None,

        "messages": [],

        "sessions_list": [],

        "last_chunks": [],

        "renaming_id": None,

        "last_uploaded": None,

        "active_domain": config.DEFAULT_DOMAIN,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_new_session():

    sid = str(uuid.uuid4())

    database.create_session(
        sid,
        title="New chat",
    )

    st.session_state.session_id = sid
    st.session_state.messages = []
    st.session_state.last_chunks = []

    st.rerun()


def load_session(sid):

    st.session_state.session_id = sid

    history = database.get_history(
        sid,
        limit=100,
    )

    st.session_state.messages = [
        {
            "role": msg["role"],
            "content": msg["content"],
        }
        for msg in history
    ]


# -------------------------------------------------
# EXPORT CHAT
# -------------------------------------------------

def export_session_markdown(
    session_id: str,
    title: str,
) -> str:

    history = database.get_history(
        session_id,
        limit=1000,
    )

    lines = [
        f"# {title}",
        "",
    ]

    for msg in history:

        role = (
            "**You**"
            if msg["role"] == "user"
            else "**Assistant**"
        )

        lines.append(f"{role}: {msg['content']}")
        lines.append("")

    return "\n".join(lines)

# -------------------------------------------------
# RAG PIPELINE
# -------------------------------------------------

def process_query_stream(
    query: str,
    model,
    client,
    llm: OllamaLLM,
):

    sid = st.session_state.session_id

    domain = st.session_state.get(
        "active_domain",
        config.DEFAULT_DOMAIN,
    )

    history = database.get_history(sid)

    try:
        expanded_query = expand_query(
            query,
            history,
        )
    except Exception:
        expanded_query = query

    # -----------------------------
    # Retrieval
    # -----------------------------
    if domain == "All Domains":
        chunks = retrieve_all_domains(
            expanded_query,
            model,
            client,
        )
    else:
        chunks = retrieve(
            expanded_query,
            model,
            client,
            domain=domain,
        )

    prompt = build_prompt(
        chunks,
        history,
        query,
    )

    start = _time.time()

    collected_tokens = []

    for token in llm.stream(prompt):
        collected_tokens.append(token)
        yield token

    elapsed_ms = int((_time.time() - start) * 1000)

    answer = "".join(collected_tokens)

    database.save_message(
        sid,
        "user",
        query,
    )

    database.save_message(
        sid,
        "assistant",
        answer,
        sources=[
            {
                "source": c["source"],
                "score": c["score"],
                "page": c.get("page"),
                "domain": c.get("domain", domain),
            }
            for c in chunks
        ],
    )

    analytics.log_query(
        sid,
        elapsed_ms,
        chunks,
    )

    current = next(
        (
            s
            for s in database.list_sessions()
            if s["session_id"] == sid
        ),
        None,
    )

    if current and current["title"] == "New chat":
        database.update_session_title(
            sid,
            query[:50] + ("..." if len(query) > 50 else ""),
        )

    st.session_state.last_chunks = chunks


# -------------------------------------------------
# UPLOAD
# -------------------------------------------------

def render_upload_widget(model, client):

    with st.sidebar.expander(
        "📄 Upload Document",
        expanded=False,
    ):

        domain = st.session_state.get(
            "active_domain",
            config.DEFAULT_DOMAIN,
        )

        if domain == "All Domains":
            st.info(
                "Select a specific domain before uploading."
            )
            return

        st.caption(
            f"Uploading to **{domain}**"
        )

        uploaded_file = st.file_uploader(
            "Add document",
            type=[
                "txt",
                "md",
                "py",
                "csv",
                "html",
                "pdf",
                "docx",
            ],
            key="doc_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.caption(
                "Supported: pdf, docx, txt, md, py, csv, html"
            )
            return

        if (
            st.session_state.get("last_uploaded")
            == uploaded_file.name
        ):
            return

        try:

            with st.spinner(
                f"Indexing {uploaded_file.name}..."
            ):

                count = ingest_uploaded_file(
                    uploaded_file,
                    model,
                    client,
                    config.DOCUMENTS_DIR,
                    domain=domain,
                )

            st.session_state.last_uploaded = uploaded_file.name

            if count > 0:
                st.success(
                    f"✅ Indexed {count} chunks into '{domain}'"
                )
            else:
                st.warning(
                    "No extractable text found."
                )

        except Exception as e:
            st.error(str(e))


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

def render_sidebar(llm, client):

    with st.sidebar:

        st.title("🔍 RAG Chat")

        if st.button(
            "＋ New Chat",
            use_container_width=True,
        ):
            start_new_session()

        st.divider()

        if llm.health_check():
            st.success(
                f"Ollama : {config.OLLAMA_MODEL}"
            )
        else:
            st.error("Ollama Offline")

        # ==================================================
        # DOMAIN SELECTOR
        # ==================================================

        st.divider()

        st.markdown("### Knowledge Domain")

        options = ["All Domains"] + config.DOMAINS

        selected = st.selectbox(
            "Search In",
            options,
            index=options.index(
                st.session_state.active_domain
            )
            if st.session_state.active_domain in options
            else 0,
            label_visibility="collapsed",
        )

        if selected != st.session_state.active_domain:
            st.session_state.active_domain = selected
            st.rerun()

        if selected == "All Domains":

            domains = list_all_collections(client)

            total = sum(
                get_or_create_collection(
                    client,
                    d,
                ).count()
                for d in config.DOMAINS
            )

            st.caption(
                f"{total} chunks across {len(domains)} domains"
            )

        else:

            collection = get_or_create_collection(
                client,
                selected,
            )

            st.caption(
                f"{collection.count()} chunks in this domain"
            )

        # ==================================================
        # CHAT LIST
        # ==================================================

        st.divider()

        st.subheader("Chats")

        for session in database.list_sessions():

            active = (
                session["session_id"]
                == st.session_state.session_id
            )

            label = (
                ("▶ " if active else "")
                + session["title"]
            )

            cols = st.columns([0.7, 0.15, 0.15])

            with cols[0]:

                if st.button(
                    label,
                    key=f"open_{session['session_id']}",
                ):
                    load_session(session["session_id"])
                    st.rerun()

            with cols[1]:

                if st.button(
                    "✏️",
                    key=f"ren_{session['session_id']}",
                ):
                    st.session_state.renaming_id = session["session_id"]

            with cols[2]:

                if st.button(
                    "🗑️",
                    key=f"del_{session['session_id']}",
                ):
                    database.delete_session(
                        session["session_id"]
                    )

                    if active:
                        st.session_state.session_id = None
                        st.session_state.messages = []

                    st.rerun()

        # ==================================================
        # KNOWLEDGE BASE
        # ==================================================

        st.divider()

        with st.expander("📚 Knowledge Base"):

            if selected == "All Domains":

                total = sum(
                    get_or_create_collection(
                        client,
                        d,
                    ).count()
                    for d in config.DOMAINS
                )

                st.metric(
                    "Chunks",
                    total,
                )

            else:

                col = get_or_create_collection(
                    client,
                    selected,
                )

                st.metric(
                    "Chunks",
                    col.count(),
                )

        # ==================================================
        # ANALYTICS
        # ==================================================

        with st.expander("📊 Analytics"):

            stats = analytics.get_stats()

            st.metric(
                "Total Queries",
                stats["total_queries"],
            )

            st.metric(
                "Average Response",
                f"{stats['avg_response_ms']} ms",
            )

            if stats["top_sources"]:

                st.caption("Top Sources")

                for name, count in stats["top_sources"]:

                    st.caption(
                        f"• {name} ({count}×)"
                    )

# -------------------------------------------------
# CHAT
# -------------------------------------------------

def render_chat(model, client, llm):

    st.title("📄 Document Intelligence Chatbot")

    if not st.session_state.session_id:
        st.info("Click **New Chat** to begin.")
        return

    # -----------------------------
    # Export conversation
    # -----------------------------

    current_title = next(
        (
            s["title"]
            for s in database.list_sessions()
            if s["session_id"] == st.session_state.session_id
        ),
        "chat",
    )

    md = export_session_markdown(
        st.session_state.session_id,
        current_title,
    )

    st.download_button(
        "⬇️ Export Conversation",
        data=md,
        file_name=f"{current_title[:30].replace(' ', '_')}.md",
        mime="text/markdown",
    )

    st.divider()

    # -----------------------------
    # Previous Messages
    # -----------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # -----------------------------
    # User Input
    # -----------------------------

    query = st.chat_input(
        "Ask something about your documents..."
    )

    if not query:
        return

    # -----------------------------
    # User Message
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.write(query)

    # -----------------------------
    # Assistant Response
    # -----------------------------

    with st.chat_message("assistant"):

        try:

            answer = st.write_stream(
                process_query_stream(
                    query,
                    model,
                    client,
                    llm,
                )
            )

            chunks = st.session_state.last_chunks

            if chunks:

                with st.expander(
                    f"📚 Sources ({len(chunks)})"
                ):

                    for chunk in chunks:

                        page = ""

                        if chunk.get("page"):
                            page = f" • Page {chunk['page']}"

                        domain = chunk.get(
                            "domain",
                            st.session_state.active_domain,
                        )

                        st.markdown(
                            f"""
**{chunk['source']}**

Domain: **{domain}**

Relevance: `{chunk['score']:.2f}`

{page}
"""
                        )

                        st.caption(
                            chunk["text"][:250] + "..."
                        )

        except Exception as e:

            answer = f"Error: {e}"

            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    init_session_state()

    model, client, llm = load_resources()

    render_sidebar(
        llm,
        client,
    )

    render_upload_widget(
        model,
        client,
    )

    render_chat(
        model,
        client,
        llm,
    )


if __name__ == "__main__":
    main()