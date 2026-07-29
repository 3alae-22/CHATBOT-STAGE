from pathlib import Path
from pypdf import PdfReader

from app.repositories.vector_repository import connect_to_db, VectorRepository
from app.ingestion.chunk import split_text
from app.ingestion.contextualize import generate_doc_summary, contextualize_chunk
from app.core.embeddings import embed_batch, detect_lang


if __name__ == '__main__':
    conn = connect_to_db()
    repo = VectorRepository(conn)
    repo.create_table()

    pdf_names = [
        "Decret_marches_publics_n_2_22_431_du_09_03_2023_Fr marchés publics",
        "قانون الوظيفة العمومية",
        "ExtraitsDahir_N_1-58-008_13092024_Fr sanctions disciplinaires"
    ]

    try:
        for pdf_name in pdf_names:
            pdf_path = Path(f"../../data/raw/{pdf_name}.pdf")
            reader = PdfReader(pdf_path)
            nb_pages = len(reader.pages)

            first_pages_text = "\n".join(
                reader.pages[i].extract_text() or "" for i in range(min(2, nb_pages))
            )
            doc_lang = detect_lang(first_pages_text)
            doc_summary = generate_doc_summary(first_pages_text, lang=doc_lang)
            print(f"[{pdf_name}] Résumé généré ({doc_lang}) : {doc_summary[:80]}...")

            for page_num in range(nb_pages):
                if repo.already_ingested(pdf_name, page_num):
                    continue

                chunks = split_text(pdf_name, page_num)
                for chunk in chunks:
                    chunk["lang"] = detect_lang(chunk["chunk"])
                    chunk["contextualized_text"] = contextualize_chunk(
                        chunk["chunk"], doc_summary, lang=chunk["lang"]
                    )

                texts_to_embed = [c["contextualized_text"] for c in chunks]
                embeddings = embed_batch(texts_to_embed)

                repo.insert_chunks(chunks, embeddings)
    finally:
        conn.close()