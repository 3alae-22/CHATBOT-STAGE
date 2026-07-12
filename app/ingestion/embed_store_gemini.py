import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from pypdf import PdfReader
from pathlib import Path
from chunk import split_text
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from tenacity import retry, stop_after_attempt, wait_exponential
import re
import os
import time

env_path = Path(r"..\..\.env")
load_dotenv(env_path)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def connect_to_db():
    print("Connecting to PostgreSQL database...")
    try:
        database_url = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(database_url)
        return conn
    except psycopg2.Error as e:
        print(f"Database connection failed: {e}")
        raise

def create_table(conn):
    print("Creating table if not exists...")
    try:
        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("CREATE SCHEMA IF NOT EXISTS dev;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dev.chunks(
                    id SERIAL PRIMARY KEY,
                    pdf_name TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    lang TEXT NOT NULL,
                    embedding VECTOR(3072)
                );
            """)
        conn.commit()
        print("Table created.")
    except psycopg2.Error as e:
        print(f"Failed to create table: {e}")
        raise

def already_ingested(conn, pdf_name, page_num):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*) FROM dev.chunks
            WHERE pdf_name = %s AND page_num = %s
        """, (pdf_name, page_num))
        return cursor.fetchone()[0] > 0

def detect_lang(text):
    if re.search(r"[a-zA-Zà-üÀ-Ü]", text):
        return "fr"
    else:
        return "ar"

def embed_with_retry(client, text, max_retries=5):
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            return result.embeddings[0].values
        except genai_errors.ClientError as e:
            if e.code == 429:
                wait = 60
                print(f"Rate limited. Waiting {wait}s before retry ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded max retries due to rate limiting.")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def insert_records(conn, chunks, embeddings):
    try:
        with conn.cursor() as cursor:

            rows = [
                (
                    chunks[i]["pdf_name"],
                    chunks[i]["page_num"],
                    chunks[i]["chunk"],
                    chunks[i]["lang"],
                    np.array(embeddings[i], dtype=np.float32)
                )
                for i in range(len(chunks))
            ]
            cursor.executemany("""
                INSERT INTO dev.chunks(
                    pdf_name, page_num, chunk_text, lang, embedding
                ) VALUES (%s, %s, %s, %s, %s);
            """, rows)
        conn.commit()
        return len(rows)
    except psycopg2.Error as e:
        print(f"Error inserting data: {e}")
        raise

if __name__ == '__main__':

    conn = connect_to_db()
    create_table(conn)
    register_vector(conn)

    pdf_names = ["Decret_marches_publics_n_2_22_431_du_09_03_2023_Fr marchés publics",
                "قانون الوظيفة العمومية"
                ]
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        for pdf_name in pdf_names:
            nb_pages = len(PdfReader(Path(f"../../data/raw/{pdf_name}.pdf")).pages)

            for page_num in range(nb_pages):
                if already_ingested(conn, pdf_name, page_num):
                    continue

                chunks = split_text(pdf_name, page_num)
                for chunk in chunks:
                    chunk["lang"] = detect_lang(chunk["chunk"])

                embeddings = []
                for chunk in chunks:
                    embeddings.append(embed_with_retry(client, chunk["chunk"]))
                    time.sleep(0.7)

                insert_records(conn, chunks, embeddings)

    finally:
        conn.close()