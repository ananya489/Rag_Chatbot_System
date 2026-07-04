import time
import database
import json


def create_analytics_table():

    with database.get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                response_time_ms INTEGER NOT NULL,
                source_count INTEGER NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)



def log_query(session_id: str, response_time_ms: int, sources: list[dict]) -> None:

    create_analytics_table()

    with database.get_connection() as conn:

        conn.execute(
            """
            INSERT INTO analytics
            (
                session_id,
                response_time_ms,
                source_count,
                timestamp
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                response_time_ms,
                len(sources),
                int(time.time())
            )
        )



def get_stats():

    create_analytics_table()


    with database.get_connection() as conn:


        row = conn.execute(
            """
            SELECT 
            COUNT(*) as total,
            AVG(response_time_ms) as avg_ms
            FROM analytics
            """
        ).fetchone()



        top_sources = conn.execute(
            """
            SELECT sources
            FROM messages
            WHERE role='assistant'
            """
        ).fetchall()



    source_counts = {}


    for r in top_sources:

        try:

            for s in json.loads(r["sources"]):

                name = s.get(
                    "source",
                    "unknown"
                )

                source_counts[name] = (
                    source_counts.get(name,0)+1
                )

        except:

            pass



    top = sorted(
        source_counts.items(),
        key=lambda x:-x[1]
    )[:5]



    return {

        "total_queries": row["total"] or 0,

        "avg_response_ms": round(
            row["avg_ms"] or 0
        ),

        "top_sources": top
    }