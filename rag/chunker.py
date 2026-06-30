import os
import re
from pathlib import Path

import config


# ── Supported file types ──────────────────────────────────────────────────────

TEXT_EXTENSIONS = {".txt", ".md", ".py", ".csv", ".html"}
PDF_EXTENSION   = ".pdf"
ALL_EXTENSIONS  = TEXT_EXTENSIONS | {PDF_EXTENSION}


# ── PDF loader ────────────────────────────────────────────────────────────────

def load_pdf(file_path: Path) -> list[dict]:
    """
    Extract text from a PDF file page by page using pypdf.

    Each page becomes a separate entry in the returned list so that
    page numbers can be preserved in chunk metadata. Pages that yield
    no text (e.g. scanned images without OCR) are silently skipped.

    Args:
        file_path: Absolute Path to a .pdf file.

    Returns:
        List of page dicts:
            [{"text": "...", "source": "book.pdf", "page": 1}, ...]
        Returns an empty list if the PDF cannot be read.

    Note:
        pypdf can only extract text from born-digital PDFs (exported
        from Word, LaTeX, etc.). Scanned PDFs require an OCR step
        (e.g. pytesseract) which is outside the scope of this phase.
    """
    try:
        from pypdf import PdfReader      # Late import — only used for PDFs
    except ImportError:
        print("[chunker] pypdf not installed. Run: pip install pypdf==4.3.1")
        return []

    pages = []
    try:
        reader     = PdfReader(str(file_path))
        page_count = len(reader.pages)

        print(f"[chunker] Loaded PDF: {file_path.name} | pages: {page_count}")

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
                text = text.strip()

                if not text:
                    # Scanned page or image-only — no extractable text
                    continue

                pages.append({
                    "text":   text,
                    "source": file_path.name,
                    "page":   page_num,
                })

            except Exception as page_err:
                # One bad page shouldn't stop the rest of the document
                print(
                    f"[chunker] Skipping page {page_num} of "
                    f"{file_path.name}: {page_err}"
                )

        if not pages:
            print(
                f"[chunker] No extractable text in {file_path.name}. "
                "It may be a scanned PDF — OCR support is not yet implemented."
            )

    except Exception as e:
        # Corrupted file, encrypted PDF, or unsupported format
        print(f"[chunker] Failed loading {file_path.name}: {e}")

    return pages


# ── Text document loader ──────────────────────────────────────────────────────

def load_text_document(file_path: Path) -> dict | None:
    """
    Read a single plain-text file and return its content.

    Args:
        file_path: Absolute Path to a supported text file.

    Returns:
        Dict with keys text, source — or None if the file is empty
        or cannot be read.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        if not text:
            return None

        print(f"[chunker] Loaded: {file_path.name} | chars: {len(text)}")
        return {"text": text, "source": file_path.name}

    except Exception as e:
        print(f"[chunker] Skipping {file_path.name}: {e}")
        return None


# ── Directory walker ──────────────────────────────────────────────────────────

def load_all_documents(documents_path: Path) -> list[dict]:
    """
    Walk the documents directory and load every supported file.

    PDFs are loaded page-by-page (each page becomes one entry).
    Text files are loaded as a single entry.

    Args:
        documents_path: Path to the documents directory.

    Returns:
        List of raw document dicts ready for cleaning and chunking:
            Text: {"text": "...", "source": "notes.md"}
            PDF:  {"text": "...", "source": "book.pdf", "page": 3}
    """
    documents_path = Path(documents_path)

    if not documents_path.exists():
        print(f"[chunker] Documents directory not found: {documents_path}")
        return []

    all_docs: list[dict] = []

    for file_path in sorted(documents_path.iterdir()):
        ext = file_path.suffix.lower()

        if ext not in ALL_EXTENSIONS:
            continue                          # Silently skip unsupported types

        if ext == PDF_EXTENSION:
            pages = load_pdf(file_path)
            all_docs.extend(pages)            # One entry per non-empty page

        elif ext in TEXT_EXTENSIONS:
            doc = load_text_document(file_path)
            if doc:
                all_docs.append(doc)

    return all_docs


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize whitespace in extracted text.

    PDFs often produce excessive whitespace, mid-word hyphenation from
    line breaks, and inconsistent newlines. This normaliser:
      1. Normalises line endings (Windows → Unix)
      2. Collapses 3+ consecutive blank lines to 2 (preserve paragraphs)
      3. Replaces tabs with spaces
      4. Strips trailing whitespace from each line
      5. Strips leading/trailing whitespace from the whole string
      6. Removes soft hyphens that PDFs insert at line breaks
         e.g. "repre-\nsentation" → "representation"

    Args:
        text: Raw text string from any document type.

    Returns:
        Cleaned text string.
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Rejoin words hyphenated across PDF line breaks
    # "repre-\nsentation" → "representation"
    text = re.sub(r"-\n(\S)", r"\1", text)

    text = re.sub(r"\n{3,}", "\n\n", text)   # Collapse excessive blank lines
    text = text.replace("\t", " ")

    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ── Recursive text splitter ───────────────────────────────────────────────────

def split_text(
    text: str,
    chunk_size:    int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """
    Recursively split text into chunks of at most chunk_size characters
    with chunk_overlap characters of shared context between adjacent chunks.

    Split priority (most natural boundary first):
      1. Double newline  — paragraph boundary
      2. Single newline  — line boundary
      3. Period + space  — sentence boundary
      4. Space           — word boundary
      5. Hard cut        — last resort, avoids infinite loops

    Args:
        text:          The text to split.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Characters shared between adjacent chunks.

    Returns:
        List of chunk strings, each at most chunk_size characters.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " ", ""]

    for separator in separators:
        if separator == "":
            # Last resort — hard-cut at chunk_size
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i : i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

        if separator not in text:
            continue

        parts   = text.split(separator)
        chunks  = []
        current = ""

        for part in parts:
            candidate = current + separator + part if current else part

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())

                if chunk_overlap > 0 and current:
                    current = current[-chunk_overlap:] + separator + part
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        if len(chunks) > 1:
            return chunks

    return [text]


# ── Public interface ──────────────────────────────────────────────────────────

def load_and_chunk(
    documents_path: Path | None  = None,
    chunk_size:     int          = config.CHUNK_SIZE,
    chunk_overlap:  int          = config.CHUNK_OVERLAP,
) -> list[dict]:
    """
    Full ingestion pipeline: load → clean → split → tag with metadata.

    This is the only function called by the rest of the application.
    The output format is identical regardless of source file type —
    retriever.py and ChromaDB receive the same structure for PDFs and
    text files alike.

    Output format per chunk:
        {
            "text":     "actual chunk content",
            "source":   "filename.pdf",          # or .txt, .md, etc.
            "chunk_id": "filename.pdf_p3_0",     # unique ID
            "page":     3,                        # PDF only — absent for text
        }

    chunk_id format:
        Text files: "{source}_{chunk_index}"         e.g. "notes.md_4"
        PDF files:  "{source}_p{page}_{chunk_index}" e.g. "book.pdf_p12_2"

    The "page" key is only present for PDF-sourced chunks. retriever.py
    stores whatever metadata dict is passed — no changes needed there.

    Args:
        documents_path: Directory containing source documents.
                        Defaults to config.DOCUMENTS_DIR.
        chunk_size:     Max characters per chunk.
        chunk_overlap:  Overlap between adjacent chunks.

    Returns:
        List of chunk dicts ready for embed_batch() and ChromaDB ingest.
    """
    path     = Path(documents_path or config.DOCUMENTS_DIR)
    raw_docs = load_all_documents(path)

    if not raw_docs:
        print("[chunker] No documents found. Add files to data/documents/")
        return []

    all_chunks: list[dict] = []

    for doc in raw_docs:
        cleaned = clean_text(doc["text"])
        pieces  = split_text(cleaned, chunk_size, chunk_overlap)
        is_pdf  = doc["source"].lower().endswith(".pdf")

        for idx, piece in enumerate(pieces):
            if is_pdf:
                page      = doc.get("page", 1)
                chunk_id  = f"{doc['source']}_p{page}_{idx}"
                chunk     = {
                    "text":     piece,
                    "source":   doc["source"],
                    "chunk_id": chunk_id,
                    "page":     page,          # Preserved for citations
                }
            else:
                chunk_id  = f"{doc['source']}_{idx}"
                chunk     = {
                    "text":     piece,
                    "source":   doc["source"],
                    "chunk_id": chunk_id,
                }

            all_chunks.append(chunk)

    doc_count = len({d["source"] for d in raw_docs})   # Unique files
    print(f"[chunker] {doc_count} document(s) → {len(all_chunks)} chunks")

    return all_chunks