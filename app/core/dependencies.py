from google import genai
from fastapi import Depends

from app.core.config import GEMINI_API_KEY
from app.repositories.vector_repository import connect_to_db, VectorRepository
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService


def get_vector_repository():
    
    conn = connect_to_db()
    try:
        yield VectorRepository(conn)
    finally:
        conn.close()


def get_retrieval_service(
    repo: VectorRepository = Depends(get_vector_repository),
) -> RetrievalService:
    return RetrievalService(repo)


def get_llm_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def get_generation_service(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    llm_client=Depends(get_llm_client),
) -> GenerationService:
    return GenerationService(
        llm_client=llm_client,
        retrieval_service=retrieval_service,
    )