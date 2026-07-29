import re
from pathlib import Path

import fitz
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.genai.errors import ClientError

from app.ingestion.detect_scan import classify_document, PageInfo
from app.ingestion.ocr.gemini_vision_engine import GeminiVisionEngine
from app.ingestion.contextualize import generate_doc_summary, contextualize_chunk
from app.repositories.vector_repository import VectorRepository
from app.core.embeddings import embed_batch, detect_lang


CHUNK_SIZE = 300
CHUNK_OVERLAP = 0
LATIN_CHAR_THRESHOLD = 15


def _is_quota_exhausted(exc: Exception) -> bool:
    return isinstance(exc, ClientError) and "RESOURCE_EXHAUSTED" in str(exc)


def _render_page_to_image_bytes(pdf_path: Path, page_number: int, dpi: int = 200) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=dpi)
    image_bytes = pix.tobytes("png")
    doc.close()
    return image_bytes


def _extract_native_text(pdf_path: Path, page_number: int) -> str:
    reader = PdfReader(pdf_path)
    return reader.pages[page_number].extract_text()


def _split_ocr_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)


def _looks_corrupted(text: str, expected_lang: str) -> bool:
    if expected_lang == "ar":
        latin_chars = len(re.findall(r"[a-zA-Z]", text))
        return latin_chars > LATIN_CHAR_THRESHOLD
    return False


def process_page(pdf_path: Path, page_info: PageInfo, engine: GeminiVisionEngine, lang_prompt: str) -> dict:
    classification = page_info["classification"]
    page_number = page_info["page_number"]

    if classification == "numerique":
        text = _extract_native_text(pdf_path, page_number)
        return {"chunks": _split_ocr_text(text), "source_method": "native", "table_json": None}

    image_bytes = _render_page_to_image_bytes(pdf_path, page_number)

    if classification == "scanne_texte":
        text = engine.extract_text(image_bytes, lang=lang_prompt)
        return {"chunks": _split_ocr_text(text), "source_method": "ocr_text", "table_json": None}

    if classification == "scanne_tableau":
        result = engine.extract_table(image_bytes, lang=lang_prompt)
        return {"chunks": [result["markdown"]], "source_method": "ocr_table", "table_json": result["json_data"]}

    raise ValueError(f"Classification inconnue : {classification}")


def process_document(pdf_path: Path, lang_prompt: str) -> None:
    engine = GeminiVisionEngine()
    repo = VectorRepository()
    pdf_name = pdf_path.name

    pages_info = classify_document(pdf_path)

    page_results = {}
    quota_hit_during_extraction = False

    for page_info in pages_info:
        page_num = page_info["page_number"]

        if repo.already_ingested(pdf_name, page_num):
            continue

        try:
            result = process_page(pdf_path, page_info, engine, lang_prompt)
        except Exception as e:
            if _is_quota_exhausted(e):
                print(
                    f"Quota Gemini épuisé à la page {page_num} de {pdf_name} pendant l'extraction. "
                    f"Arrêt propre. Relance plus tard pour continuer."
                )
                quota_hit_during_extraction = True
                break
            raise

        if not any(c.strip() for c in result["chunks"]):
            continue

        page_results[page_num] = result

    if not page_results:
        if quota_hit_during_extraction:
            print(f"Aucune nouvelle page traitée pour {pdf_name} avant l'épuisement du quota.")
        return

    first_page_nums = sorted(page_results.keys())[:2]
    first_pages_text = "\n".join(
        "\n".join(page_results[p]["chunks"]) for p in first_page_nums
    )
    doc_lang = detect_lang(first_pages_text)
    doc_summary = generate_doc_summary(first_pages_text, lang=doc_lang)

    for page_num, result in page_results.items():
        for chunk_text in result["chunks"]:
            if not chunk_text.strip():
                continue

            chunk_lang = detect_lang(chunk_text)
            contextualized_text = contextualize_chunk(chunk_text, doc_summary, lang=chunk_lang)

            if _looks_corrupted(contextualized_text, chunk_lang):
                contextualized_text = chunk_text

            embedding = embed_batch([contextualized_text])[0]

            chunk_id = repo.insert_chunk_with_source(
                pdf_name=pdf_name,
                page_num=page_num,
                chunk_text=chunk_text,
                contextualized_text=contextualized_text,
                lang=chunk_lang,
                embedding=embedding,
                source_method=result["source_method"],
            )

            if result["table_json"] is not None:
                repo.insert_extracted_table(chunk_id, result["table_json"])

    if quota_hit_during_extraction:
        print(f"{len(page_results)} page(s) insérée(s) pour {pdf_name} avant l'arrêt. Relance possible plus tard.")
    else:
        print(f"{pdf_name} traité intégralement ({len(page_results)} page(s) insérée(s)).")