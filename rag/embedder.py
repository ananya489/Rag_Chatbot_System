from sentence_transformers import SentenceTransformer
import config


def load_embedder() -> SentenceTransformer:
    """
    Load the sentence-transformer embedding model.

    This is called once and cached by Streamlit's @st.cache_resource.
    The model weights (~90MB) are downloaded on first use and cached
    locally by sentence-transformers in ~/.cache/huggingface/

    Returns:
        A loaded SentenceTransformer model ready to encode text.
    """
    print(f"[embedder] Loading model: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    print(f"[embedder] Model ready. Output dimension: {config.EMBEDDING_DIMENSION}")
    return model


def embed_text(model: SentenceTransformer, text: str) -> list[float]:
    """
    Embed a single string into a 384-dimensional vector.

    Used at query time: the user's question is embedded with this
    same model before searching ChromaDB.

    Args:
        model: The loaded SentenceTransformer instance.
        text:  Any string — a query, a chunk, a sentence.

    Returns:
        A list of 384 floats representing the text's meaning.
    """
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def embed_batch(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
    Embed multiple strings in one efficient batch call.

    Used at ingestion time: all chunks are embedded together,
    which is significantly faster than embedding one at a time
    because the model processes them in parallel on the GPU/CPU.

    Args:
        model: The loaded SentenceTransformer instance.
        texts: List of strings to embed.

    Returns:
        List of 384-float vectors, one per input string.
    """
    print(f"[embedder] Embedding batch of {len(texts)} texts...")
    vectors = model.encode(
        texts,
        batch_size=32,           # Process 32 chunks at a time
        show_progress_bar=True,  # Visible progress for large document sets
        convert_to_numpy=True,
    )
    print(f"[embedder] Done. Shape: {vectors.shape}")
    return vectors.tolist()