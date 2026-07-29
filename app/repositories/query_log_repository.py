from datetime import datetime

from app.repositories.vector_repository import connect_to_db


class QueryLogRepository:

    def __init__(self, conn=None):
        self.conn = conn or connect_to_db()

    def create_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dev.query_logs (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    lang TEXT,
                    abstained BOOLEAN NOT NULL,
                    top_rerank_score REAL,
                    sources_count INTEGER,
                    latency_ms INTEGER,
                    created_at TIMESTAMP NOT NULL DEFAULT now()
                );
            """)
        self.conn.commit()

    def log_query(
        self,
        question: str,
        lang: str | None,
        abstained: bool,
        top_rerank_score: float | None,
        sources_count: int,
        latency_ms: int,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO dev.query_logs
                    (question, lang, abstained, top_rerank_score, sources_count, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (
                question,
                lang,
                abstained,
                float(top_rerank_score) if top_rerank_score is not None else None,
                sources_count,
                latency_ms,
            ))
        self.conn.commit()