from rag.chunker import load_and_chunk
from rag.embedder import load_embedder
from rag.retriever import get_chroma_client, ingest_chunks, retrieve


# Load
model = load_embedder()
client = get_chroma_client()
chunks = load_and_chunk()

# Ingest
ingest_chunks(chunks, model, client)

# Retrieve
results = retrieve(
    "What is machine learning?",
    model,
    client
)

print()

for r in results:
    print(f"Score: {r['score']} | Source: {r['source']}")
    print(f"Text:  {r['text'][:120]}")
    print()