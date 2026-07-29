from app.repositories.vector_repository import VectorRepository
from app.core.embeddings import embed_query
from app.retrieval.hybrid_search import fuse_rrf
from app.retrieval.reranker import Reranker


class RetrievalService:

    def __init__(self, vector_repository: VectorRepository, reranker: Reranker):
        self.vector_repository = vector_repository
        self.reranker = reranker

    def retrieve_candidates(self, query: str, lang: str, k: int = 20) -> list[tuple]:
        embedding = embed_query(query)
        dense_results = self.vector_repository.search_similar(embedding=embedding, k=k, lang=lang)
        lexical_results = self.vector_repository.search_lexical(query=query, lang=lang, k=k)
        return fuse_rrf(dense_results, lexical_results, k=60)

    def rerank_candidates(self, query: str, candidates: list[tuple], top_n: int = 6) -> list[tuple]:
        candidate_texts = [row[3] for row in candidates]
        reranked = self.reranker.rerank(query=query, candidates=candidate_texts, top_n=top_n)
        text_to_row = {row[3]: row for row in candidates}
        return [text_to_row[text] for text, score in reranked]

    def rerank_candidates_with_scores(self, query: str, candidates: list[tuple], top_n: int = 6) -> list[tuple[tuple, float]]:
        candidate_texts = [row[3] for row in candidates]
        reranked = self.reranker.rerank(query=query, candidates=candidate_texts, top_n=top_n)
        text_to_row = {row[3]: row for row in candidates}
        return [(text_to_row[text], score) for text, score in reranked]

    def search(self, query: str, lang: str, k: int = 20, rerank_top_n: int = 6) -> list[tuple]:
        candidates = self.retrieve_candidates(query, lang, k)
        return self.rerank_candidates(query, candidates, rerank_top_n)