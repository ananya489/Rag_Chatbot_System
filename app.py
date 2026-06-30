import streamlit as st
import uuid

import config
import database

from rag.chunker import load_and_chunk
from rag.embedder import load_embedder

from rag.retriever import (
    get_chroma_client,
    get_or_create_collection,
    ingest_chunks,
    retrieve,
    needs_reingestion,
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
    layout="wide"
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



    # -------------------------
    # Smart ingestion
    # -------------------------

    if needs_reingestion(
        client,
        str(config.DOCUMENTS_DIR)
    ):


        print(
            "[app] Documents changed. Re-ingesting..."
        )


        chunks = load_and_chunk()


        if chunks:


            ingest_chunks(
                chunks,
                model,
                client
            )


        else:


            print(
                "[app] No documents found"
            )


    else:


        print(
            "[app] Using existing vector database"
        )



    return model, client, llm





# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

def init_session_state():


    defaults = {


        "session_id": None,

        "messages": [],

        "sessions_list": [],

        "last_chunks": []


    }



    for key,value in defaults.items():


        if key not in st.session_state:

            st.session_state[key] = value





def start_new_session():


    sid = str(uuid.uuid4())


    database.create_session(

        sid,

        title="New chat"

    )


    st.session_state.session_id = sid

    st.session_state.messages = []

    st.session_state.last_chunks = []

    st.session_state.sessions_list = database.list_sessions()





def load_session(sid):


    st.session_state.session_id = sid


    history = database.get_history(

        sid,

        limit=100

    )


    st.session_state.messages = [

        {

            "role":msg["role"],

            "content":msg["content"]

        }

        for msg in history

    ]





# -------------------------------------------------
# RAG PIPELINE
# -------------------------------------------------

def process_query_stream(

        query,

        model,

        client,

        llm

):


    sid = st.session_state.session_id



    history = database.get_history(

        sid

    )



    # Query expansion

    try:


        expanded_query = expand_query(

            query,

            history

        )


    except Exception:


        expanded_query = query




    # Retrieve

    chunks = retrieve(

        expanded_query,

        model,

        client

    )




    prompt = build_prompt(

        chunks,

        history,

        query

    )




    tokens = []



    for token in llm.stream(prompt):


        tokens.append(token)


        yield token




    answer = "".join(tokens)



    # Save user message

    database.save_message(

        sid,

        "user",

        query

    )




    # Save assistant message

    database.save_message(

        sid,

        "assistant",

        answer,

        sources=[


            {

                "source":c["source"],

                "score":c["score"],

                "page":c.get("page")

            }


            for c in chunks


        ]

    )



    st.session_state.last_chunks = chunks





    # Auto title

    sessions = database.list_sessions()



    current = next(

        (

            s for s in sessions

            if s["session_id"] == sid

        ),

        None

    )



    if current and current["title"] == "New chat":


        database.update_session_title(

            sid,

            query[:50]

        )



    st.session_state.sessions_list = database.list_sessions()






# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

def render_sidebar(llm, client):


    with st.sidebar:



        st.title(
            "🔍 RAG Chat"
        )



        if st.button(

            "＋ New Chat",

            use_container_width=True

        ):


            start_new_session()

            st.rerun()




        st.divider()




        if llm.health_check():


            st.success(

                f"Ollama : {config.OLLAMA_MODEL}"

            )


        else:


            st.error(
                "Ollama Offline"
            )





        st.divider()



        st.subheader(
            "Chats"
        )



        sessions = database.list_sessions()



        for s in sessions:


            active = (

                s["session_id"]

                ==

                st.session_state.session_id

            )



            label = (

                "▶ "

                if active

                else ""

            ) + s["title"]




            if st.button(

                label,

                key=s["session_id"],

                use_container_width=True

            ):


                load_session(

                    s["session_id"]

                )


                st.rerun()




        st.divider()



        with st.expander(
            "📚 Knowledge Base"
        ):



            collection = get_or_create_collection(

                client

            )



            total_chunks = collection.count()


            st.metric(
                "Indexed Chunks",
                total_chunks
            )


            st.caption(
                str(config.DOCUMENTS_DIR)
            )


            if total_chunks > 0:

                all_metadata = collection.get()["metadatas"]


                sources = sorted(
                    {
                        m["source"]
                        for m in all_metadata
                        if m and "source" in m
                    }
                )


                st.caption(
                    f"{len(sources)} file(s):"
                )


                for src in sources:

                    st.caption(
                        f"• {src}"
                    )

            else:

                st.caption(
                    "No documents indexed yet."
                )





        with st.expander(

            "📊 Retrieval Debug"

        ):



            chunks = st.session_state.last_chunks



            if chunks:



                for c in chunks:


                    score = max(

                        0,

                        min(

                            c["score"],

                            1

                        )

                    )


                    st.progress(

                        score,

                        text=(

                            f"{c['source']} "

                            f"{c['score']:.2f}"

                        )

                    )



            else:


                st.caption(

                    "Ask something to see retrieval"

                )







# -------------------------------------------------
# CHAT UI
# -------------------------------------------------

def render_chat(

        model,

        client,

        llm

):


    st.title(

        "Document Intelligence Chatbot"

    )



    if not st.session_state.session_id:


        st.info(

            "Click New Chat"

        )

        return





    for msg in st.session_state.messages:


        with st.chat_message(

            msg["role"]

        ):


            st.write(

                msg["content"]

            )





    query = st.chat_input(

        "Ask about documents..."

    )



    if query:



        st.session_state.messages.append(

            {

                "role":"user",

                "content":query

            }

        )



        with st.chat_message("user"):


            st.write(query)





        with st.chat_message("assistant"):


            try:


                answer = st.write_stream(

                    process_query_stream(

                        query,

                        model,

                        client,

                        llm

                    )

                )



                chunks = st.session_state.last_chunks




                if chunks:


                    with st.expander(

                        f"Sources ({len(chunks)})"

                    ):


                        for c in chunks:



                            page = ""


                            if c.get("page"):


                                page = (

                                    f" · page {c['page']}"

                                )



                            st.markdown(

                                f"""

                                **{c['source']}**

                                {page}

                                relevance:

                                `{c['score']:.2f}`

                                """

                            )



                            st.caption(

                                c["text"][:200]

                                +

                                "..."

                            )



            except Exception as e:


                answer = f"Error: {e}"

                st.error(answer)





        st.session_state.messages.append(

            {

                "role":"assistant",

                "content":answer

            }

        )

def render_upload_widget(model, client) -> None:
    """
    File uploader in the sidebar — saves and immediately indexes
    new documents without requiring an app restart.
    """
    from rag.retriever import ingest_uploaded_file

    with st.sidebar.expander("📄 Upload document", expanded=False):
        uploaded_file = st.file_uploader(
            "Add to knowledge base",
            type=["txt", "md", "py", "csv", "html", "pdf"],
            label_visibility="collapsed",
            key="doc_uploader",
        )

        if uploaded_file is None:
            st.caption("Supported: txt, md, py, csv, html, pdf")
            return

        # Avoid re-processing the same upload on every Streamlit rerun —
        # track the last processed filename in session_state
        if st.session_state.get("last_uploaded") == uploaded_file.name:
            return

        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                count = ingest_uploaded_file(
                    uploaded_file, model, client, config.DOCUMENTS_DIR
                )
                st.session_state.last_uploaded = uploaded_file.name

                if count > 0:
                    st.success(f"Indexed {count} chunks from {uploaded_file.name}")
                else:
                    st.warning(f"No extractable text found in {uploaded_file.name}")

            except ValueError as e:
                st.error(str(e))





# -------------------------------------------------
# MAIN
# -------------------------------------------------

# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():


    init_session_state()



    model, client, llm = load_resources()



    render_sidebar(
        llm,
        client
    )


    # Upload widget
    render_upload_widget(
        model,
        client
    )


    render_chat(
        model,
        client,
        llm
    )

if __name__=="__main__":

    main()