import sqlite3
import time
import json
import contextlib
import config


# -------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------

@contextlib.contextmanager
def get_connection():
    """
    Open SQLite connection.
    Commit on success.
    Rollback on error.
    Close always.
    """

    conn = sqlite3.connect(
        str(config.DB_PATH)
    )

    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()



# -------------------------------------------------
# INITIALIZE DATABASE
# -------------------------------------------------

def init_db() -> None:
    """
    Create all required tables.
    Safe to call every app startup.
    """

    with get_connection() as conn:

        conn.executescript("""

        CREATE TABLE IF NOT EXISTS sessions (

            session_id TEXT PRIMARY KEY,

            title TEXT NOT NULL DEFAULT 'New chat',

            created_at INTEGER NOT NULL,

            updated_at INTEGER NOT NULL

        );


        CREATE TABLE IF NOT EXISTS messages (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            session_id TEXT NOT NULL,

            role TEXT NOT NULL
                CHECK(role IN ('user','assistant')),

            content TEXT NOT NULL,

            sources TEXT NOT NULL DEFAULT '[]',

            timestamp INTEGER NOT NULL,

            FOREIGN KEY(session_id)
            REFERENCES sessions(session_id)

        );


        CREATE TABLE IF NOT EXISTS analytics (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            session_id TEXT NOT NULL,

            response_time_ms INTEGER NOT NULL,

            source_count INTEGER NOT NULL,

            timestamp INTEGER NOT NULL

        );

        """)



# -------------------------------------------------
# SESSION FUNCTIONS
# -------------------------------------------------

def create_session(
    session_id: str,
    title: str = "New chat"
):

    now = int(time.time())


    with get_connection() as conn:

        conn.execute(

            """
            INSERT INTO sessions
            (
                session_id,
                title,
                created_at,
                updated_at
            )

            VALUES (?, ?, ?, ?)

            """,

            (
                session_id,
                title,
                now,
                now
            )

        )



def list_sessions():

    with get_connection() as conn:

        rows = conn.execute(

            """
            SELECT *
            FROM sessions
            ORDER BY updated_at DESC

            """

        ).fetchall()


    return [

        dict(row)

        for row in rows

    ]



def update_session_title(
    session_id,
    title
):

    with get_connection() as conn:

        conn.execute(

            """
            UPDATE sessions

            SET title=?,
                updated_at=?

            WHERE session_id=?

            """,

            (
                title,
                int(time.time()),
                session_id
            )

        )



def delete_session(session_id):

    with get_connection() as conn:

        conn.execute(

            "DELETE FROM messages WHERE session_id=?",

            (session_id,)

        )


        conn.execute(

            "DELETE FROM sessions WHERE session_id=?",

            (session_id,)

        )



# -------------------------------------------------
# MESSAGE FUNCTIONS
# -------------------------------------------------

def save_message(
    session_id,
    role,
    content,
    sources=None
):

    with get_connection() as conn:

        conn.execute(

            """
            INSERT INTO messages
            (
                session_id,
                role,
                content,
                sources,
                timestamp
            )

            VALUES (?, ?, ?, ?, ?)

            """,

            (
                session_id,
                role,
                content,
                json.dumps(sources or []),
                int(time.time())
            )

        )


        conn.execute(

            """
            UPDATE sessions

            SET updated_at=?

            WHERE session_id=?

            """,

            (
                int(time.time()),
                session_id
            )

        )



def get_history(
    session_id,
    limit=None
):

    n = limit or config.HISTORY_LIMIT


    with get_connection() as conn:

        rows = conn.execute(

            """
            SELECT role,content

            FROM messages

            WHERE session_id=?

            ORDER BY timestamp DESC

            LIMIT ?

            """,

            (
                session_id,
                n
            )

        ).fetchall()



    return [

        {
            "role": row["role"],
            "content": row["content"]
        }

        for row in reversed(rows)

    ]