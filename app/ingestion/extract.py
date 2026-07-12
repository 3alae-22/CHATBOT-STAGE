from pypdf import PdfReader
from pathlib import Path
import os

def extract_page(pdf_name, page_num):
    pdf_path = Path(f"../../data/raw/{pdf_name}.pdf")
    reader = PdfReader(pdf_path)
    page = reader.pages[page_num]
    text = page.extract_text()
    return {"text": text, "pdf_name": pdf_name, "page_num": page_num}

if __name__ == "__main__":
    print(len(PdfReader(Path(f"../../data/raw/قانون الوظيفة العمومية.pdf")).pages))
    print(extract_page("قانون الوظيفة العمومية",11))