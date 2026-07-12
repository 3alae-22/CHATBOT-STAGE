
from extract import extract_page
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(pdf_name, page_num):
    page = extract_page(pdf_name, page_num)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0
    )
    chunks = text_splitter.split_text(page["text"])
    return [
        {"chunk": chunk, "pdf_name": page["pdf_name"], "page_num": page["page_num"]}
        for chunk in chunks
    ]

if __name__ == "__main__":
    print(split_text("Decret_marches_publics_n_2_22_431_du_09_03_2023_Fr marchés publics", 11))