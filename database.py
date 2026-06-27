import sqlite3
import time
import json
import contextlib
from pathlib import Path
import config


@contextlib.contextmanager
def get_connection():
    """
    Open a fresh SQLite connection, yield it, commit on success,
    rollback on error, always close. Thread-safe for Streamlit reruns.
    """
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New chat',
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content     TEXT NOT NULL,
                sources     TEXT NOT NULL DEFAULT '[]',
                timestamp   INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
        """)


def create_session(session_id: str, title: str = "New chat") -> dict:
    """Insert a new session. session_id is a UUID from the caller."""
    now = int(time.time())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
    return {"session_id": session_id, "title": title, "created_at": now}


def list_sessions() -> list[dict]:
    """All sessions, most recently updated first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT session_id, title, created_at, updated_at "
            "FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def update_session_title(session_id: str, title: str) -> None:
    """Rename a session — called after the first user message."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (title, int(time.time()), session_id),
        )


def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: list | None = None,
) -> None:
    """
    Insert one message. Called twice per turn:
      save_message(id, 'user', question)
      save_message(id, 'assistant', answer, sources=[...])
    """
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, sources, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(sources or []), int(time.time())),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (int(time.time()), session_id),
        )


def get_history(session_id: str, limit: int | None = None) -> list[dict]:
    """
    Last N messages for a session, oldest-first.
    Format matches what any LLM chat API expects:
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    n = limit or config.HISTORY_LIMIT
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, n),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]