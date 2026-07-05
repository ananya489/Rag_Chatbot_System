import re
from pathlib import Path

import config
from logger import get_logger

log = get_logger(__name__)
# ── Supported file types ──────────────────────────────────────────────────────

TEXT_EXTENSIONS = {".txt", ".md", ".py", ".csv", ".html"}
PDF_EXTENSION   = ".pdf"
DOCX_EXTENSION  = ".docx"
ALL_EXTENSIONS  = TEXT_EXTENSIONS | {PDF_EXTENSION, DOCX_EXTENSION}


# ── DOCX loader ───────────────────────────────────────────────────────────────

def load_docx(file_path: Path) -> dict | None:
    """
    Extract text from a .docx Word document using python-docx.

    Reads all paragraphs in document order and joins them with newlines.
    Empty paragraphs (section breaks, blank lines in the doc) are preserved
    as single newlines so the chunker's paragraph-aware splitting still works.

    Tables are not extracted in this implementation — only paragraph text.
    Table support can be added by iterating doc.tables if needed.

    Args:
        file_path: Absolute Path to a .docx file.

    Returns:
        Dict with keys text and source, or None if unreadable or empty.
    """
    try:
        from docx import Document   # Late import — only used for .docx files
    except ImportError:
        log.error("python-docx not installed. Run: pip install python-docx==1.1.2")
        return None

    try:
        doc   = Document(str(file_path))
        lines = []

        for paragraph in doc.paragraphs:
            # Strip trailing whitespace but preserve intentionally blank paragraphs
            # as empty strings — join will produce \n\n which the splitter treats
            # as a paragraph boundary.
            lines.append(paragraph.text.rstrip())

        text = "\n".join(lines).strip()

        if not text:
            log.warning("No extractable text in %s (empty document).", file_path.name)
            return None

        log.info("Loaded DOCX: %s | chars: %d", file_path.name, len(text))
        return {"text": text, "source": file_path.name}

    except Exception as e:
        log.error("Failed loading %s: %s", file_path.name, str(e))
        return None


# ── PDF loader ────────────────────────────────────────────────────────────────

def load_pdf(file_path: Path) -> list[dict]:
    """
    Extract text from a PDF file page by page using pypdf.

    Each non-empty page becomes a separate entry so page numbers
    can be preserved in chunk metadata.

    Args:
        file_path: Absolute Path to a .pdf file.

    Returns:
        List of page dicts: [{"text": "...", "source": "book.pdf", "page": 1}, ...]
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        log.error("pypdf not installed. Run: pip install pypdf==4.3.1")
        return []

    pages = []
    try:
        reader     = PdfReader(str(file_path))
        page_count = len(reader.pages)
        log.info("Loaded PDF: %s | pages: %d", file_path.name, page_count)
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({
                        "text":   text,
                        "source": file_path.name,
                        "page":   page_num,
                    })
            except Exception as page_err:
                log.warning(
    "Skipping page %d of %s: %s",
    page_num,
    file_path.name,
    str(page_err),
)

        if not pages:
            log.warning(
    "No extractable text in %s. It may be a scanned PDF — OCR is not yet supported.",
    file_path.name,
)

    except Exception as e:
        log.error("Failed loading %s: %s", file_path.name, str(e))
    return pages


# ── Text document loader ──────────────────────────────────────────────────────

def load_text_document(file_path: Path) -> dict | None:
    """
    Read a plain-text file (.txt, .md, .py, .csv, .html).

    Args:
        file_path: Absolute Path to a supported text file.

    Returns:
        Dict with keys text and source, or None if empty or unreadable.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        if not text:
            return None

        log.info("Loaded: %s | chars: %d", file_path.name, len(text))
        return {"text": text, "source": file_path.name}

    except Exception as e:
        log.warning("Skipping %s: %s", file_path.name, str(e))
        return None


# ── Directory walker ──────────────────────────────────────────────────────────

def load_all_documents(documents_path: Path) -> list[dict]:
    """
    Walk the documents directory and load every supported file.

    Dispatch:
        .pdf  → load_pdf()          (returns list — one entry per page)
        .docx → load_docx()         (returns single dict)
        text  → load_text_document() (returns single dict)

    Args:
        documents_path: Path to the documents directory.

    Returns:
        List of raw document dicts for cleaning and chunking.
    """
    documents_path = Path(documents_path)

    if not documents_path.exists():
        log.warning("Documents directory not found: %s", documents_path)
        return []

    all_docs: list[dict] = []

    for file_path in sorted(documents_path.iterdir()):
        ext = file_path.suffix.lower()

        if ext not in ALL_EXTENSIONS:
            continue

        if ext == PDF_EXTENSION:
            pages = load_pdf(file_path)
            all_docs.extend(pages)

        elif ext == DOCX_EXTENSION:
            doc = load_docx(file_path)
            if doc:
                all_docs.append(doc)

        elif ext in TEXT_EXTENSIONS:
            doc = load_text_document(file_path)
            if doc:
                all_docs.append(doc)

    return all_docs


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize whitespace in extracted text.

    Handles PDF hyphenation artifacts, Windows line endings,
    excessive blank lines, and trailing whitespace.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(\S)", r"\1", text)        # Rejoin PDF hyphenated words
    text = re.sub(r"\n{3,}", "\n\n", text)         # Collapse excessive blank lines
    text = text.replace("\t", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()
# ── Test helper ───────────────────────────────────────────────────────────────

def tokenize_for_test(text: str) -> list[str]:
    """
    Helper used only by unit tests.

    Tokenizes text into lowercase alphanumeric tokens.
    """
    if not text:
        return []

    return re.findall(r"\b[a-z0-9]+\b", text.lower())

# ── Recursive text splitter ───────────────────────────────────────────────────

def split_text(
    text: str,
    chunk_size:    int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """
    Recursively split text into chunks of at most chunk_size characters.

    Split priority: paragraph → line → sentence → word → hard cut.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for separator in ["\n\n", "\n", ". ", " ", ""]:
        if separator == "":
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i : i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

        if separator not in text:
            continue

        parts, chunks, current = text.split(separator), [], ""

        for part in parts:
            candidate = current + separator + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = (current[-chunk_overlap:] + separator + part
                           if chunk_overlap > 0 and current else part)

        if current.strip():
            chunks.append(current.strip())

        if len(chunks) > 1:
            return chunks

    return [text]


# ── Public interface ──────────────────────────────────────────────────────────

def load_and_chunk(
    documents_path: Path | None = None,
    chunk_size:     int         = config.CHUNK_SIZE,
    chunk_overlap:  int         = config.CHUNK_OVERLAP,
) -> list[dict]:
    """
    Full ingestion pipeline: load → clean → split → tag with metadata.

    Output format per chunk:
        Text / DOCX: {"text": "...", "source": "notes.md",  "chunk_id": "notes.md_4"}
        PDF:         {"text": "...", "source": "book.pdf",  "chunk_id": "book.pdf_p3_1", "page": 3}

    This is the only function called externally.
    """
    path     = Path(documents_path or config.DOCUMENTS_DIR)
    raw_docs = load_all_documents(path)

    if not raw_docs:
        log.warning("No documents found. Add files to data/documents/")
        return []

    all_chunks: list[dict] = []

    for doc in raw_docs:
        cleaned  = clean_text(doc["text"])
        pieces   = split_text(cleaned, chunk_size, chunk_overlap)
        is_pdf   = doc["source"].lower().endswith(".pdf")

        for idx, piece in enumerate(pieces):
            if is_pdf:
                page = doc.get("page", 1)
                all_chunks.append({
                    "text":     piece,
                    "source":   doc["source"],
                    "chunk_id": f"{doc['source']}_p{page}_{idx}",
                    "page":     page,
                })
            else:
                all_chunks.append({
                    "text":     piece,
                    "source":   doc["source"],
                    "chunk_id": f"{doc['source']}_{idx}",
                })

    doc_count = len({d["source"] for d in raw_docs})
    log.info("%d document(s) → %d chunks", doc_count, len(all_chunks))
    return all_chunks