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
)
from rag.prompt_builder import build_prompt
from rag.query_expander import expand_query

from llm.ollama_llm import OllamaLLM


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Document Intelligence Chatbot",
    page_icon="🔍",
    layout="wide"
)


# ---------------- LOAD RESOURCES ----------------

@st.cache_resource
def load_resources():

    database.init_db()

    model = load_embedder()

    client = get_chroma_client()

    llm = OllamaLLM()


    if not llm.health_check():

        st.warning(
            f"Ollama not connected.\n\n"
            f"Run:\n"
            f"ollama serve\n\n"
            f"Then:\n"
            f"ollama pull {config.OLLAMA_MODEL}"
        )


    chunks = load_and_chunk()


    if chunks:

        ingest_chunks(
            chunks,
            model,
            client
        )


    return model, client, llm



# ---------------- SESSION ----------------


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
            "role":m["role"],
            "content":m["content"]
        }

        for m in history

    ]



# ---------------- RAG PIPELINE ----------------


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


    # query expansion

    try:

        expanded_query = expand_query(
            query,
            history
        )

    except Exception:

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



    answer = []


    for token in llm.stream(prompt):

        answer.append(token)

        yield token



    final_answer = "".join(answer)



    database.save_message(
        sid,
        "user",
        query
    )


    database.save_message(

        sid,

        "assistant",

        final_answer,

        sources=[

            {
                "source":c["source"],
                "score":c["score"]

            }

            for c in chunks

        ]

    )


    st.session_state.last_chunks = chunks



    # update title

    sessions = database.list_sessions()


    current = next(

        (
            s for s in sessions
            if s["session_id"] == sid
        ),

        None

    )


    if current and current["title"]=="New chat":

        database.update_session_title(

            sid,

            query[:50]

        )



    st.session_state.sessions_list = database.list_sessions()



# ---------------- SIDEBAR ----------------


def render_sidebar(llm,client):


    with st.sidebar:


        st.title("🔍 RAG Chat")



        if st.button(
            "New Chat",
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
                key=s["session_id"]
            ):

                load_session(
                    s["session_id"]
                )

                st.rerun()



        st.divider()



        with st.expander(
            "Knowledge Base"
        ):


            col = get_or_create_collection(
                client
            )


            st.metric(
                "Chunks",
                col.count()
            )



        with st.expander(
            "Retrieval Debug"
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
                        text=f"{c['source']} {c['score']:.2f}"
                    )


            else:

                st.caption(
                    "No retrieval yet"
                )



# ---------------- CHAT ----------------


def render_chat(model,client,llm):


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



            except Exception as e:


                answer = f"Error: {e}"

                st.error(answer)




        st.session_state.messages.append(

            {
                "role":"assistant",
                "content":answer
            }

        )



# ---------------- MAIN ----------------


def main():


    init_session_state()


    model,client,llm = load_resources()



    render_sidebar(
        llm,
        client
    )


    render_chat(
        model,
        client,
        llm
    )



if __name__=="__main__":

    main()