import json
import time
from pathlib import Path
from typing import TypedDict

from google import genai
from google.genai import types
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

from app.core.config import GEMINI_API_KEY
import httpx


MIN_SECONDS_BETWEEN_CALLS = 13

MODEL_NAME = "gemini-2.5-flash"


class TableExtractionResult(TypedDict):
    markdown: str
    json_data: list[dict]


@retry(
    retry=retry_if_exception_type((APIError, httpx.TransportError)),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(5),
)
def _call_gemini_vision(client: genai.Client, image_bytes: bytes, prompt: str):
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )


class GeminiVisionEngine:

    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.client = genai.Client(api_key=api_key)

    def extract_text(self, image_bytes: bytes, lang: str) -> str:
        prompt = (
            f"Transcribe all the text visible in this scanned document page image. "
            f"The text is in {lang}. "
            "Output ONLY the transcribed text, preserving line breaks and reading order. "
            "Do NOT translate, summarize, or add any commentary. "
            "Ignore stamps, logos, and signatures — transcribe only the running text content."
        )

        response = _call_gemini_vision(self.client, image_bytes, prompt)
        time.sleep(MIN_SECONDS_BETWEEN_CALLS)

        return response.text.strip()

    def extract_table(self, image_bytes: bytes, lang: str) -> TableExtractionResult:
        prompt = (
            f"Look at this scanned document page image, which contains a table. "
            f"The table content is in {lang}. "
            "Reconstruct the table faithfully, respecting rows and columns as they visually appear. "
            "Respond ONLY with a JSON object with exactly two keys:\n"
            '- "markdown": the table as a Markdown table string\n'
            '- "rows": a JSON array of objects, one per data row, using the table\'s '
            "column headers as keys\n"
            "Do NOT include any text outside this JSON object, no preamble, no code fences."
        )

        response = _call_gemini_vision(self.client, image_bytes, prompt)
        time.sleep(MIN_SECONDS_BETWEEN_CALLS)

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Réponse Gemini non parsable en JSON pour l'extraction de tableau : {e}\n"
                f"Réponse brute : {raw[:500]}"
            ) from e

        return TableExtractionResult(
            markdown=parsed.get("markdown", ""),
            json_data=parsed.get("rows", []),
        )


if __name__ == "__main__":
    import fitz 

    def render_page_to_image_bytes(pdf_path: Path, page_number: int, dpi: int = 200) -> bytes:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=dpi)
        image_bytes = pix.tobytes("png")
        doc.close()
        return image_bytes

    engine = GeminiVisionEngine()

    # Test 1: text
    # pdf_path = Path("../../../data/raw/Circulaire_N_26-2012_18092024التغيب غير المشروع عن العمل.pdf")
    # image_bytes = render_page_to_image_bytes(pdf_path, page_number=1)
    # text = engine.extract_text(image_bytes, lang="arabic")
    # print("extract_text")
    # print(text)

    # Test 2: tableau
    pdf_path_table = Path("../../../data/raw/حالات الاستيداع المؤقت.pdf")
    image_bytes_table = render_page_to_image_bytes(pdf_path_table, page_number=0)
    table_result = engine.extract_table(image_bytes_table, lang="arabic")
    print("extract_table")
    print(table_result["markdown"])
    print(table_result["json_data"])