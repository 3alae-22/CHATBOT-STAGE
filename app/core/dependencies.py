from functools import lru_cache

from google import genai
from fastapi import Depends

from app.core.config import GEMINI_API_KEY
from app.repositories.vector_repository import connect_to_db, VectorRepository
from app.retrieval.reranker import Reranker
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.graph.build_graph import build_graph


def get_vector_repository():
    conn = connect_to_db()
    try:
        yield VectorRepository(conn)
    finally:
        conn.close()


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()


def get_retrieval_service(
    repo: VectorRepository = Depends(get_vector_repository),
    reranker: Reranker = Depends(get_reranker),
) -> RetrievalService:
    return RetrievalService(repo, reranker)


@lru_cache
def get_llm_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def get_generation_service(
    llm_client=Depends(get_llm_client),
) -> GenerationService:
    return GenerationService(llm_client=llm_client)


def get_graph(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
    llm_client=Depends(get_llm_client),
):
    return build_graph(retrieval_service, generation_service, llm_client)