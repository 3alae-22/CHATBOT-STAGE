from app.repositories.vector_repository import VectorRepository
from app.core.embeddings import embed_query


class RetrievalService:

    def __init__(self, vector_repository: VectorRepository):
        self.vector_repository = vector_repository

    def search(self, query: str, k: int = 5, lang: str | None = None) -> list:
        embedding = embed_query(query)
        return self.vector_repository.search_similar(embedding, k=k, lang=lang)
