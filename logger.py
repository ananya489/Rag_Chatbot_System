"""
logger.py — Centralized logging configuration for the RAG Chat System.

Every module imports get_logger() from here instead of using print().
This gives us:
    - Severity levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - Timestamps on every log line
    - Simultaneous output to terminal (colored) and log file
    - Single place to change log level for the entire application

Usage in any module:
    from logger import get_logger
    log = get_logger(__name__)

    log.info("Loaded PDF: %s | pages: %d", filename, page_count)
    log.warning("No extractable text in %s", filename)
    log.error("Failed loading %s: %s", filename, str(e))
"""

import logging
import sys
from pathlib import Path


# ── Log file location ─────────────────────────────────────────────────────────

LOG_DIR  = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "rag_chat.log"

# Create the logs directory if it doesn't exist
LOG_DIR.mkdir(exist_ok=True)


# ── Format ────────────────────────────────────────────────────────────────────

# Terminal format — concise, readable during development
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# File format — includes more detail for post-mortem debugging
FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ── Root logger setup (runs once at import time) ──────────────────────────────

def _configure_root_logger() -> None:
    """
    Configure the root logger with two handlers:
        1. StreamHandler → terminal (stdout)
        2. FileHandler   → logs/rag_chat.log (rotating not needed at this scale)

    Called once when this module is first imported. Subsequent calls to
    get_logger() return child loggers that inherit this configuration.
    """
    root = logging.getLogger("rag_chat")

    # Don't reconfigure if handlers already attached (Streamlit reruns)
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)    # Capture everything; handlers filter

    # ── Console handler ───────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)   # INFO+ to terminal
    console_handler.setFormatter(
        logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT)
    )

    # ── File handler ──────────────────────────────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)     # DEBUG+ to file
    file_handler.setFormatter(
        logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
    )

    root.addHandler(console_handler)
    root.addHandler(file_handler)


_configure_root_logger()


# ── Public interface ──────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a named child logger under the 'rag_chat' namespace.

    Args:
        name: Typically __name__ from the calling module.
               'rag.chunker' → logger name 'rag_chat.rag.chunker'

    Returns:
        A configured Logger instance. Inherits handlers from root logger.

    Example:
        from logger import get_logger
        log = get_logger(__name__)
        log.info("Loaded: %s", filename)
    """
    return logging.getLogger(f"rag_chat.{name}")