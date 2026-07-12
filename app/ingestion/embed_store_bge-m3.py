from pathlib import Path
from pypdf import PdfReader

from app.repositories.vector_repository import connect_to_db, VectorRepository
from app.ingestion.chunk import split_text
from app.core.embeddings import embed_batch, detect_lang


if __name__ == '__main__':
    conn = connect_to_db()
    repo = VectorRepository(conn)
    repo.create_table()

    pdf_names = [
        "Decret_marches_publics_n_2_22_431_du_09_03_2023_Fr marchés publics",
        "قانون الوظيفة العمومية",
    ]

    try:
        for pdf_name in pdf_names:
            nb_pages = len(PdfReader(Path(f"../../data/raw/{pdf_name}.pdf")).pages)

            for page_num in range(nb_pages):
                if repo.already_ingested(pdf_name, page_num):
                    continue

                chunks = split_text(pdf_name, page_num)
                for chunk in chunks:
                    chunk["lang"] = detect_lang(chunk["chunk"])

                texts = [c["chunk"] for c in chunks]
                embeddings = embed_batch(texts)

                repo.insert_chunks(chunks, embeddings)
    finally:
        conn.close()