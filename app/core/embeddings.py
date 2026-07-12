import numpy as np
import ollama
import re

EMBED_MODEL = "bge-m3"


def embed_query(text: str) -> np.ndarray:
    response = ollama.embed(model=EMBED_MODEL, input=[text])
    return np.array(response["embeddings"][0], dtype=np.float32)


def embed_batch(texts: list[str]) -> list:
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return response["embeddings"]


def detect_lang(text: str) -> str:
    if re.search(r"[a-zA-Zà-üÀ-Ü]", text):
        return "fr"
    return "ar"