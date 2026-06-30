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
            f"Run:\nollama serve\n\n"
            f"Then:\nollama pull {config.OLLAMA_MODEL}"
        )



    if needs_reingestion(
        client,
        str(config.DOCUMENTS_DIR)
    ):


        print("[app] Re-ingesting documents")


        chunks = load_and_chunk()


        if chunks:

            ingest_chunks(
                chunks,
                model,
                client
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

        "last_uploaded": None

    }


    for k,v in defaults.items():

        if k not in st.session_state:

            st.session_state[k] = v





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
            "role":m["role"],
            "content":m["content"]
        }

        for m in history
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


    try:

        expanded_query = expand_query(
            query,
            history
        )

    except:

        expanded_query = query



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



    tokens=[]


    for token in llm.stream(prompt):

        tokens.append(token)

        yield token



    answer="".join(tokens)



    database.save_message(
        sid,
        "user",
        query
    )



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



    current = next(

        (
            s for s in database.list_sessions()
            if s["session_id"]==sid
        ),

        None

    )



    if current and current["title"]=="New chat":

        database.update_session_title(
            sid,
            query[:50]
        )






# -------------------------------------------------
# UPLOAD WIDGET
# -------------------------------------------------

def render_upload_widget(model,client):

    from rag.retriever import ingest_uploaded_file


    with st.sidebar.expander(
        "📄 Upload Document"
    ):


        file = st.file_uploader(

            "Add document",

            type=[
                "txt",
                "md",
                "py",
                "csv",
                "html",
                "pdf"
            ],

            key="upload"

        )


        if file is None:

            return



        if st.session_state.last_uploaded == file.name:

            return



        with st.spinner(
            "Indexing..."
        ):


            count = ingest_uploaded_file(

                file,

                model,

                client,

                config.DOCUMENTS_DIR

            )


            st.session_state.last_uploaded=file.name



            st.success(
                f"Indexed {count} chunks"
            )






# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

def render_sidebar(llm,client):


    with st.sidebar:


        st.title("🔍 RAG Chat")



        if st.button(
            "＋ New Chat",
            use_container_width=True
        ):

            start_new_session()

            st.rerun()



        st.divider()



        st.subheader("Chats")



        sessions = database.list_sessions()



        active_sid = st.session_state.session_id



        for session in sessions:


            active = session["session_id"] == active_sid


            cols = st.columns(
                [0.7,0.15,0.15]
            )


            with cols[0]:


                label = (

                    "▶ "

                    if active

                    else ""

                ) + session["title"]



                if st.button(
                    label,
                    key=f"open_{session['session_id']}"
                ):

                    load_session(
                        session["session_id"]
                    )

                    st.rerun()



            with cols[1]:


                if st.button(
                    "✏️",
                    key=f"ren_{session['session_id']}"
                ):

                    st.session_state.renaming_id=session["session_id"]



            with cols[2]:


                if st.button(
                    "🗑️",
                    key=f"del_{session['session_id']}"
                ):

                    database.delete_session(
                        session["session_id"]
                    )

                    if active:

                        st.session_state.session_id=None

                        st.session_state.messages=[]


                    st.rerun()



            if st.session_state.renaming_id == session["session_id"]:


                new_title = st.text_input(

                    "Rename",

                    value=session["title"],

                    key=f"title_{session['session_id']}"

                )


                if st.button(
                    "Save",
                    key=f"save_{session['session_id']}"
                ):

                    database.update_session_title(

                        session["session_id"],

                        new_title

                    )


                    st.session_state.renaming_id=None

                    st.rerun()





        st.divider()



        with st.expander(
            "📚 Knowledge Base"
        ):


            col=get_or_create_collection(client)


            count=col.count()


            st.metric(
                "Chunks",
                count
            )


            if count:


                metadata=col.get()["metadatas"]


                files=sorted(
                    {
                        m["source"]
                        for m in metadata
                    }
                )


                st.caption(
                    f"{len(files)} files"
                )


                for f in files:

                    st.caption(
                        "• "+f
                    )

            else:

                st.caption(
                    "No documents"
                )






# -------------------------------------------------
# CHAT
# -------------------------------------------------

def render_chat(model,client,llm):


    st.title(
        "Document Intelligence Chatbot"
    )



    if not st.session_state.session_id:

        st.info(
            "Click New Chat"
        )

        return



    for m in st.session_state.messages:


        with st.chat_message(
            m["role"]
        ):

            st.write(
                m["content"]
            )




    query=st.chat_input(
        "Ask about documents..."
    )


    if query:


        with st.chat_message("assistant"):


            answer=st.write_stream(

                process_query_stream(

                    query,

                    model,

                    client,

                    llm

                )

            )



        st.session_state.messages.append(

            {
                "role":"assistant",

                "content":answer
            }

        )






# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    init_session_state()


    model,client,llm = load_resources()



    render_sidebar(
        llm,
        client
    )



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