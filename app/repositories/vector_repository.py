import json

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
                    contextualized_text TEXT,
                    lang TEXT NOT NULL,
                    embedding VECTOR(1024)
                );
            """)
            cur.execute("""
                ALTER TABLE dev.chunks
                ADD COLUMN IF NOT EXISTS contextualized_text TEXT;
            """)
            cur.execute("""
                UPDATE dev.chunks
                SET contextualized_text = chunk_text
                WHERE contextualized_text IS NULL;
            """)
            cur.execute("""
                ALTER TABLE dev.chunks
                ADD COLUMN IF NOT EXISTS source_method TEXT NOT NULL DEFAULT 'native';
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dev.extracted_tables (
                    id SERIAL PRIMARY KEY,
                    chunk_id INTEGER NOT NULL REFERENCES dev.chunks(id) ON DELETE CASCADE,
                    table_json JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT now()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_extracted_tables_chunk_id
                ON dev.extracted_tables(chunk_id);
            """)
        self.conn.commit()

    def already_ingested(self, pdf_name: str, page_num: int) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM dev.chunks
                WHERE pdf_name = %s AND page_num = %s
            """, (pdf_name, page_num))
            return cur.fetchone()[0] > 0

    def search_similar(self, embedding: np.ndarray, k: int = 5, lang: str | None = None) -> list[tuple]:
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

    def search_lexical(self, query: str, lang: str, k: int = 10) -> list[tuple]:
        ts_config = "french" if lang == "fr" else "simple"
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, pdf_name, page_num, chunk_text,
                    ts_rank(to_tsvector(%s, contextualized_text), plainto_tsquery(%s, %s)) AS rank
                FROM dev.chunks
                WHERE lang = %s
                AND to_tsvector(%s, contextualized_text) @@ plainto_tsquery(%s, %s)
                ORDER BY rank DESC
                LIMIT %s;
            """, (ts_config, ts_config, query, lang, ts_config, ts_config, query, k))
            return cur.fetchall()

    def insert_chunks(self, chunks: list[dict], embeddings: list) -> int:
        rows = [
            (
                chunks[i]["pdf_name"],
                chunks[i]["page_num"],
                chunks[i]["chunk"],
                chunks[i]["contextualized_text"],
                chunks[i]["lang"],
                np.array(embeddings[i], dtype=np.float32),
            )
            for i in range(len(chunks))
        ]
        with self.conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO dev.chunks(pdf_name, page_num, chunk_text, contextualized_text, lang, embedding)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, rows)
        self.conn.commit()
        return len(rows)

    def insert_chunk_with_source(
        self,
        pdf_name: str,
        page_num: int,
        chunk_text: str,
        contextualized_text: str,
        lang: str,
        embedding,
        source_method: str = "native",
    ) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO dev.chunks(pdf_name, page_num, chunk_text, contextualized_text, lang, embedding, source_method)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (pdf_name, page_num, chunk_text, contextualized_text, lang,
                  np.array(embedding, dtype=np.float32), source_method))
            chunk_id = cur.fetchone()[0]
        self.conn.commit()
        return chunk_id

    def insert_extracted_table(self, chunk_id: int, table_json: list[dict]) -> int:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO dev.extracted_tables(chunk_id, table_json)
                VALUES (%s, %s)
                RETURNING id;
            """, (chunk_id, json.dumps(table_json, ensure_ascii=False)))
            table_id = cur.fetchone()[0]
        self.conn.commit()
        return table_id

    def get_table_for_chunk(self, chunk_id: int) -> list[dict] | None:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT table_json FROM dev.extracted_tables WHERE chunk_id = %s;
            """, (chunk_id,))
            row = cur.fetchone()
        return row[0] if row else None