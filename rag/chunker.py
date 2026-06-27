import os
import re
from pathlib import Path
import config


# ── Supported file types ──────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".csv", ".html"}


# ── Text loading ──────────────────────────────────────────────────────────────

def load_document(file_path: Path) -> str:
    """
    Read a single file and return its text content.
    Only plain-text formats are supported right now.
    PDF support can be added later by installing pypdf and
    adding an elif branch for .pdf extension.

    Args:
        file_path: Absolute path to the file.

    Returns:
        The file's text content as a single string.

    Raises:
        ValueError: If the file extension is not supported.
        IOError:    If the file cannot be read.
    """
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_all_documents(documents_path: Path) -> list[dict]:
    """
    Walk the documents directory and load every supported file.

    Returns a list of dicts:
        [{"text": "...", "source": "filename.txt"}, ...]

    Silently skips unsupported files so you can drop any files
    into data/documents/ without the app crashing.
    """
    documents = []
    documents_path = Path(documents_path)

    if not documents_path.exists():
        print(f"[chunker] Documents directory not found: {documents_path}")
        return []

    for file_path in sorted(documents_path.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            text = load_document(file_path)
            if text.strip():               # Skip empty files
                documents.append({
                    "text":   text,
                    "source": file_path.name,
                })
                print(f"[chunker] Loaded: {file_path.name} ({len(text)} chars)")
        except Exception as e:
            print(f"[chunker] Skipping {file_path.name}: {e}")

    return documents


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize whitespace without destroying meaningful line breaks.

    Steps:
    1. Replace Windows line endings with Unix (\r\n → \n)
    2. Collapse 3+ consecutive blank lines into 2 (preserve paragraph spacing)
    3. Replace tabs with spaces
    4. Strip trailing whitespace from each line
    5. Strip leading/trailing whitespace from the whole document
    """
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\t", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ── Core splitting logic ──────────────────────────────────────────────────────

def split_text(
    text: str,
    chunk_size: int   = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """
    Recursively split text into chunks of at most chunk_size characters,
    with chunk_overlap characters of context shared between adjacent chunks.

    Split priority (most natural boundary first):
      1. Double newline  (paragraph boundary)
      2. Single newline  (line boundary)
      3. Period + space  (sentence boundary)
      4. Space           (word boundary)
      5. Hard cut        (last resort — avoids infinite loops on dense text)

    Args:
        text:          The document text to split.
        chunk_size:    Max characters per chunk.
        chunk_overlap: Characters shared between adjacent chunks.

    Returns:
        List of chunk strings, each at most chunk_size characters.
    """
    # Base case — text already fits in one chunk
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Separators in priority order — most natural break first
    separators = ["\n\n", "\n", ". ", " ", ""]

    for separator in separators:
        if separator == "":
            # Last resort: hard-cut at chunk_size
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i : i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

        if separator not in text:
            continue                   # Try the next separator

        # Split on this separator and recombine into size-respecting chunks
        parts   = text.split(separator)
        chunks  = []
        current = ""

        for part in parts:
            candidate = current + separator + part if current else part

            if len(candidate) <= chunk_size:
                # Still fits — keep accumulating
                current = candidate
            else:
                # Doesn't fit — save current chunk, start a new one
                if current.strip():
                    chunks.append(current.strip())

                # Carry overlap from the end of the saved chunk
                if chunk_overlap > 0 and current:
                    overlap_text = current[-chunk_overlap:]
                    current = overlap_text + separator + part
                else:
                    current = part

        # Don't forget the last accumulated chunk
        if current.strip():
            chunks.append(current.strip())

        # If splitting on this separator produced useful chunks, return them
        if len(chunks) > 1:
            return chunks

    return [text]


# ── Public interface ──────────────────────────────────────────────────────────

def load_and_chunk(
    documents_path: Path | None = None,
    chunk_size: int              = config.CHUNK_SIZE,
    chunk_overlap: int           = config.CHUNK_OVERLAP,
) -> list[dict]:
    """
    Full ingestion pipeline: load all documents → clean → split → tag.

    This is the only function the rest of the app calls.

    Returns a list of chunk dicts:
        [
            {
                "text":     "actual chunk content",
                "source":   "filename.txt",
                "chunk_id": "filename.txt_0",
            },
            ...
        ]

    Args:
        documents_path: Directory containing source documents.
                        Defaults to config.DOCUMENTS_DIR.
        chunk_size:     Max characters per chunk.
        chunk_overlap:  Overlap between adjacent chunks.
    """
    path = Path(documents_path or config.DOCUMENTS_DIR)
    raw_documents = load_all_documents(path)

    if not raw_documents:
        print("[chunker] No documents found. Add files to data/documents/")
        return []

    all_chunks = []

    for doc in raw_documents:
        cleaned = clean_text(doc["text"])
        pieces  = split_text(cleaned, chunk_size, chunk_overlap)

        for idx, piece in enumerate(pieces):
            all_chunks.append({
                "text":     piece,
                "source":   doc["source"],
                "chunk_id": f"{doc['source']}_{idx}",
            })

    print(
        f"[chunker] {len(raw_documents)} document(s) → {len(all_chunks)} chunks"
    )
    return all_chunks