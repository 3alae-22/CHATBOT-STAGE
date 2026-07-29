import time
from pathlib import Path
from typing import Literal, TypedDict

from pypdf import PdfReader
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt
import httpx

from app.core.config import GEMINI_API_KEY


TEXT_LENGTH_THRESHOLD = 30
MIN_SECONDS_BETWEEN_CALLS = 13

PageClassification = Literal["numerique", "scanne_texte", "scanne_tableau"]


class PageInfo(TypedDict):
    page_number: int
    classification: PageClassification
    text_length: int


def _is_quota_exhausted(exc: Exception) -> bool:
    return isinstance(exc, ClientError) and "RESOURCE_EXHAUSTED" in str(exc)


def classify_page_text_or_scan(page, text_threshold: int = TEXT_LENGTH_THRESHOLD) -> tuple[Literal["numerique", "scanne"], int]:
    text = page.extract_text()
    text_len = len(text.strip())
    classification = "numerique" if text_len >= text_threshold else "scanne"
    return classification, text_len


def _render_page_to_image_bytes(pdf_path: Path, page_number: int) -> bytes:
    import fitz

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=200)
    image_bytes = pix.tobytes("png")
    doc.close()
    return image_bytes


@retry(
    retry=retry_if_exception_type((APIError, httpx.TransportError)),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(5),
)
def _call_gemini_vision(client: genai.Client, image_bytes: bytes, prompt: str):
    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )


def classify_scan_content(client: genai.Client, pdf_path: Path, page_number: int) -> Literal["texte", "tableau"]:
    image_bytes = _render_page_to_image_bytes(pdf_path, page_number)

    prompt = (
        "Look at this scanned document page image. "
        "Answer with exactly one word: 'tableau' if the page's main content "
        "is a table (rows/columns of structured data), or 'texte' if it is "
        "primarily running text (even if it contains a stamp, logo, or header image). "
        "Answer with only one of these two words, nothing else."
    )

    response = _call_gemini_vision(client, image_bytes, prompt)
    time.sleep(MIN_SECONDS_BETWEEN_CALLS)

    answer = response.text.strip().lower()
    if "tableau" in answer:
        return "tableau"
    return "texte"


def classify_document(pdf_path: Path) -> list[PageInfo]:
    reader = PdfReader(pdf_path)
    results: list[PageInfo] = []
    client = genai.Client(api_key=GEMINI_API_KEY)

    for page_number, page in enumerate(reader.pages):
        base_classification, text_len = classify_page_text_or_scan(page)

        if base_classification == "numerique":
            final_classification: PageClassification = "numerique"
        else:
            try:
                content_type = classify_scan_content(client, pdf_path, page_number)
                final_classification = "scanne_tableau" if content_type == "tableau" else "scanne_texte"
            except Exception as e:
                if _is_quota_exhausted(e):
                    print(
                        f"Quota Gemini épuisé à la page {page_number} de {pdf_path.name}. "
                        f"Pages classifiées jusqu'ici : {len(results)}/{len(reader.pages)}. Arrêt propre."
                    )
                    break
                raise

        results.append({
            "page_number": page_number,
            "classification": final_classification,
            "text_length": text_len,
        })

    return results


def summarize_document(results: list[PageInfo]) -> dict:
    total = len(results)
    counts = {"numerique": 0, "scanne_texte": 0, "scanne_tableau": 0}
    for r in results:
        counts[r["classification"]] += 1
    return {"total_pages": total, **counts}


if __name__ == "__main__":
    raw_dir = Path("../../data/raw")
    for pdf_path in raw_dir.glob("*.pdf"):
        results = classify_document(pdf_path)
        summary = summarize_document(results)
        print(f"{pdf_path.name}: {summary}")