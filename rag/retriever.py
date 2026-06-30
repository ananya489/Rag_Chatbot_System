import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

import config
from rag.embedder import embed_text, embed_batch


COLLECTION_NAME = "rag_documents"


# ---------------------------------------------------------
# Chroma Client
# ---------------------------------------------------------

def get_chroma_client() -> chromadb.PersistentClient:

    return chromadb.PersistentClient(

        path=str(config.CHROMA_PATH),

        settings=ChromaSettings(
            anonymized_telemetry=False
        )

    )



# ---------------------------------------------------------
# Collection
# ---------------------------------------------------------

def get_or_create_collection(client):

    return client.get_or_create_collection(

        name=COLLECTION_NAME,

        metadata={
            "hnsw:space": "cosine"
        }

    )



# ---------------------------------------------------------
# Document Fingerprint Cache
# ---------------------------------------------------------

def get_documents_fingerprint(
        documents_path: str
) -> str:


    path = Path(documents_path)


    if not path.exists():

        return "no_documents"



    supported = {

        ".txt",
        ".md",
        ".py",
        ".csv",
        ".html",
        ".pdf"

    }



    files = []



    for file in path.iterdir():

        if file.suffix.lower() in supported:


            files.append(

                {

                    "name": file.name,

                    "size": file.stat().st_size,

                    "mtime": round(
                        file.stat().st_mtime,
                        2
                    )

                }

            )



    files.sort(
        key=lambda x:x["name"]
    )



    if not files:

        return "empty"



    raw = json.dumps(
        files,
        sort_keys=True
    )



    return hashlib.md5(
        raw.encode()
    ).hexdigest()





def needs_reingestion(
        client,
        documents_path: str
):


    cache_file = (

        Path(config.CHROMA_PATH)

        /

        "ingestion_cache.json"

    )



    current_fp = get_documents_fingerprint(
        documents_path
    )


    old_fp = ""



    if cache_file.exists():


        try:

            with open(
                cache_file,
                "r"
            ) as f:


                data = json.load(f)


                old_fp = data.get(
                    "fingerprint",
                    ""
                )


        except Exception:

            old_fp = ""




    if current_fp == old_fp:


        print(

            f"[cache] Fingerprint match "
            f"{current_fp[:8]} "
            "- skipping ingestion"

        )


        return False




    print(

        f"[cache] Fingerprint changed "
        f"{old_fp[:8] or 'none'} "
        f"-> {current_fp[:8]}"

    )



    cache_file.parent.mkdir(
        exist_ok=True
    )



    with open(
        cache_file,
        "w"
    ) as f:


        json.dump(

            {
                "fingerprint": current_fp
            },

            f

        )



    return True





# ---------------------------------------------------------
# Ingestion
# ---------------------------------------------------------

def ingest_chunks(

        chunks:list[dict],

        model:SentenceTransformer,

        client

):


    if not chunks:


        print(
            "[retriever] No chunks found"
        )


        return 0




    collection = get_or_create_collection(
        client
    )



    existing = set(

        collection.get()["ids"]

    )



    new_chunks = [

        c for c in chunks

        if c["chunk_id"] not in existing

    ]



    if not new_chunks:


        print(
            "[retriever] No new chunks"
        )


        return 0




    texts = [

        c["text"]

        for c in new_chunks

    ]



    ids = [

        c["chunk_id"]

        for c in new_chunks

    ]



    # PDF page metadata support

    metadata = [

        {

            "source": c["source"],

            "chunk_id": c["chunk_id"],


            **(

                {
                    "page": c["page"]
                }

                if "page" in c

                else {}

            )

        }

        for c in new_chunks

    ]



    vectors = embed_batch(

        model,

        texts

    )



    collection.add(

        ids=ids,

        documents=texts,

        embeddings=vectors,

        metadatas=metadata

    )



    print(

        f"[retriever] Added {len(ids)} chunks"

    )


    return len(ids)






# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve(

        query:str,

        model,

        client,

        top_k=config.RETRIEVAL_TOP_K,

        top_n=config.RERANK_TOP_N

):


    collection = get_or_create_collection(
        client
    )



    if collection.count() == 0:


        print(
            "[retriever] Empty database"
        )


        return []




    query_vector = embed_text(

        model,

        query

    )



    results = collection.query(

        query_embeddings=[

            query_vector

        ],


        n_results=min(

            top_k,

            collection.count()

        ),


        include=[

            "documents",

            "metadatas",

            "distances"

        ]

    )



    documents = results["documents"][0]

    metadata = results["metadatas"][0]

    distances = results["distances"][0]



    chunks = []



    for doc,meta,distance in zip(

        documents,

        metadata,

        distances

    ):



        score = round(

            1 - distance,

            4

        )



        chunks.append(

            {

                "text": doc,


                "source": meta.get(

                    "source",

                    "unknown"

                ),


                "page": meta.get(

                    "page",

                    None

                ),


                "score": score

            }

        )




    chunks.sort(

        key=lambda x:x["score"],

        reverse=True

    )



    final_chunks = [

        c for c in chunks

        if c["score"] >= 0.15

    ][:top_n]



    print(

        f"[retriever] Returned "
        f"{len(final_chunks)} chunks"

    )



    return final_chunks

# ---------------------------------------------------------
# Upload Support (Phase 4)
# ---------------------------------------------------------

def save_uploaded_file(uploaded_file, documents_path: Path):

    documents_path = Path(documents_path)

    documents_path.mkdir(
        parents=True,
        exist_ok=True
    )


    destination = documents_path / uploaded_file.name


    if destination.exists():

        raise ValueError(
            f"{uploaded_file.name} already exists."
        )


    with open(destination, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )


    print(
        f"[upload] Saved {destination.name}"
    )


    return destination





def ingest_uploaded_file(
        uploaded_file,
        model,
        client,
        documents_path
):

    from rag.chunker import (
        load_text_document,
        load_pdf,
        clean_text,
        split_text
    )


    saved_file = save_uploaded_file(
        uploaded_file,
        documents_path
    )


    ext = saved_file.suffix.lower()



    if ext == ".pdf":

        raw_docs = load_pdf(
            saved_file
        )


    else:

        doc = load_text_document(
            saved_file
        )

        raw_docs = [doc] if doc else []




    if not raw_docs:

        print(
            "[upload] No content extracted"
        )

        return 0




    chunks = []



    for doc in raw_docs:


        text = clean_text(
            doc["text"]
        )


        pieces = split_text(
            text
        )


        for idx,piece in enumerate(pieces):


            if ext == ".pdf":


                page = doc.get(
                    "page",
                    1
                )


                chunk_id = (
                    f"{doc['source']}_"
                    f"p{page}_"
                    f"{idx}"
                )


                chunks.append(

                    {
                        "text":piece,

                        "source":doc["source"],

                        "chunk_id":chunk_id,

                        "page":page
                    }

                )


            else:


                chunks.append(

                    {
                        "text":piece,

                        "source":doc["source"],

                        "chunk_id":
                        f"{doc['source']}_{idx}"
                    }

                )




    count = ingest_chunks(
        chunks,
        model,
        client
    )



    print(
        f"[upload] Added {count} chunks"
    )


    return count