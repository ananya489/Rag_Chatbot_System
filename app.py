import streamlit as st
import uuid

import config
import database

from rag.chunker import load_and_chunk
from rag.embedder import load_embedder
from rag.retriever import (
    get_chroma_client,
    ingest_chunks,
    retrieve
)
from rag.prompt_builder import build_prompt
from rag.query_expander import expand_query
from llm.ollama_llm import OllamaLLM



# ---------------- Page Config ----------------

st.set_page_config(
    page_title="RAG Chat",
    page_icon="🔍",
    layout="wide"
)



# ---------------- Load Resources ----------------


@st.cache_resource
def load_resources():

    database.init_db()

    model = load_embedder()

    client = get_chroma_client()

    llm = OllamaLLM()


    chunks = load_and_chunk()


    if chunks:

        ingest_chunks(
            chunks,
            model,
            client
        )


    return model, client, llm





# ---------------- Session State ----------------


def init_session_state():

    if "session_id" not in st.session_state:

        st.session_state.session_id = None


    if "messages" not in st.session_state:

        st.session_state.messages = []


    if "sessions_list" not in st.session_state:

        st.session_state.sessions_list = []


    if "last_chunks" not in st.session_state:

        st.session_state.last_chunks = []





def start_new_session():

    session_id = str(uuid.uuid4())


    database.create_session(
        session_id,
        title="New chat"
    )


    st.session_state.session_id = session_id

    st.session_state.messages = []

    st.session_state.sessions_list = database.list_sessions()





def load_session(session_id):

    st.session_state.session_id = session_id


    history = database.get_history(
        session_id,
        limit=100
    )


    st.session_state.messages = [

        {
            "role": msg["role"],
            "content": msg["content"]
        }

        for msg in history

    ]






# ---------------- RAG PIPELINE ----------------


def process_query_stream(
        query,
        model,
        client,
        llm
):


    session_id = st.session_state.session_id



    # Load previous chat

    history = database.get_history(
        session_id
    )




    # -------- Query Expansion --------

    expanded_query = expand_query(
        query,
        history
    )




    # Retrieve documents

    chunks = retrieve(
        expanded_query,
        model,
        client
    )




    # Build prompt

    prompt = build_prompt(
        chunks,
        history,
        query
    )




    answer_tokens = []



    # Stream response from Ollama

    for token in llm.stream(prompt):

        answer_tokens.append(token)

        yield token




    answer = "".join(answer_tokens)





    # Save user message

    database.save_message(
        session_id,
        "user",
        query
    )




    # Save assistant message

    database.save_message(

        session_id,

        "assistant",

        answer,

        sources=[

            {
                "source": c["source"],
                "score": c["score"]
            }

            for c in chunks

        ]

    )




    # Save retrieved chunks for UI

    st.session_state.last_chunks = chunks





    # Update title

    sessions = database.list_sessions()


    current = next(

        (
            s for s in sessions
            if s["session_id"] == session_id
        ),

        None

    )



    if current and current["title"] == "New chat":


        database.update_session_title(

            session_id,

            query[:50]

        )




    st.session_state.sessions_list = database.list_sessions()






# ---------------- Sidebar ----------------


def render_sidebar(
        model,
        client,
        llm
):


    with st.sidebar:


        st.markdown(
            "## 🔍 RAG Chat"
        )




        if st.button(

            "＋ New chat",

            use_container_width=True,

            type="primary"

        ):


            start_new_session()

            st.rerun()





        st.divider()




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


            st.caption(

                "Run: ollama serve"

            )





        st.divider()



        st.markdown(
            "**Past conversations**"
        )




        sessions = (

            st.session_state.sessions_list

            or database.list_sessions()

        )




        for session in sessions:


            if st.button(

                session["title"],

                key=session["session_id"],

                use_container_width=True

            ):


                load_session(

                    session["session_id"]

                )

                st.rerun()





        st.divider()




        with st.expander(
            "Knowledge Base"
        ):


            from rag.retriever import get_or_create_collection



            col = get_or_create_collection(
                client
            )


            st.metric(

                "Chunks indexed",

                col.count()

            )



            st.caption(

                f"Documents: {config.DOCUMENTS_DIR}"

            )






# ---------------- Chat UI ----------------



def render_chat(
        model,
        client,
        llm
):


    st.title(
        "RAG Chat System"
    )




    if not st.session_state.session_id:


        st.info(
            "Click + New chat to start"
        )


        return





    # Display old messages


    for msg in st.session_state.messages:


        with st.chat_message(
            msg["role"]
        ):


            st.markdown(
                msg["content"]
            )





    query = st.chat_input(

        "Ask something about your documents..."

    )





    if query:



        st.session_state.messages.append(

            {
                "role":"user",

                "content":query
            }

        )




        with st.chat_message("user"):


            st.markdown(query)





        answer = ""





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



                        for chunk in chunks:


                            st.markdown(

                                f"""

**{chunk['source']}**

Relevance:
`{chunk['score']:.2f}`

                                """

                            )


                            st.caption(

                                chunk["text"][:200]

                                +

                                "..."

                            )




            except RuntimeError as e:


                answer = str(e)


                st.error(answer)


                st.caption(

                    "Check Ollama: ollama serve"

                )





        st.session_state.messages.append(

            {

                "role":"assistant",

                "content":answer

            }

        )






# ---------------- Main ----------------



def main():


    init_session_state()



    model, client, llm = load_resources()




    render_sidebar(

        model,

        client,

        llm

    )




    render_chat(

        model,

        client,

        llm

    )





if __name__ == "__main__":

    main()