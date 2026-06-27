from rag.chunker import load_and_chunk

chunks = load_and_chunk()

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(f"Source:   {chunk['source']}")
    print(f"ID:       {chunk['chunk_id']}")
    print(f"Length:   {len(chunk['text'])} chars")
    print(f"Preview:  {chunk['text'][:80]}...")