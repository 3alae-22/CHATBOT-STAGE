import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import DATABASE_URL


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30)
)
def connect_to_db():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        register_vector(conn)
        return conn
    except psycopg2.Error as e:
        print(f"Database connection failed: {e}")
        raise


class VectorRepository:

    def __init__(self, conn=None):
        self.conn = conn or connect_to_db()

    def create_table(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE SCHEMA IF NOT EXISTS dev;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dev.chunks(
                    id SERIAL PRIMARY KEY,
                    pdf_name TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    lang TEXT NOT NULL,
                    embedding VECTOR(1024)
                );
            """)
        self.conn.commit()

    def already_ingested(self, pdf_name: str, page_num: int) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM dev.chunks
                WHERE pdf_name = %s AND page_num = %s
            """, (pdf_name, page_num))
            return cur.fetchone()[0] > 0

    def search_similar(self, embedding: np.ndarray, k: int = 5, lang: str | None = None):
        with self.conn.cursor() as cur:
            if lang:
                cur.execute("""
                    SELECT id, pdf_name, page_num, chunk_text,
                        embedding <=> %s::vector AS distance
                    FROM dev.chunks
                    WHERE lang = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (embedding, lang, embedding, k))
            else:
                cur.execute("""
                    SELECT id, pdf_name, page_num, chunk_text,
                        embedding <=> %s::vector AS distance
                    FROM dev.chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (embedding, embedding, k))
            return cur.fetchall()

    def insert_chunks(self, chunks: list[dict], embeddings: list) -> int:
        rows = [
            (
                chunks[i]["pdf_name"],
                chunks[i]["page_num"],
                chunks[i]["chunk"],
                chunks[i]["lang"],
                np.array(embeddings[i], dtype=np.float32),
            )
            for i in range(len(chunks))
        ]
        with self.conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO dev.chunks(pdf_name, page_num, chunk_text, lang, embedding)
                VALUES (%s, %s, %s, %s, %s);
            """, rows)
        self.conn.commit()
        return len(rows)