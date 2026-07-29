from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.core.dependencies import get_graph

import time
from app.repositories.query_log_repository import QueryLogRepository

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    k: int = 5


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    abstained: bool = False


def get_query_log_repository():
    return QueryLogRepository()

@router.post("/ask", response_model=AskResponse)
def ask_endpoint(
    request: AskRequest,
    graph=Depends(get_graph),
    query_log_repo: QueryLogRepository = Depends(get_query_log_repository),
):
    start = time.perf_counter()
    try:
        result = graph.invoke({
            "question": request.question, "lang": None, "translated_question": "",
            "translated_lang": "", "k": request.k, "chunks": [], "top_rerank_score": 0.0,
            "answer": {}, "is_grounded": False, "abstained": False,
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lors de la génération : {e}")

    latency_ms = int((time.perf_counter() - start) * 1000)

    query_log_repo.log_query(
        question=request.question,
        lang=result.get("lang"),
        abstained=result.get("abstained", False),
        top_rerank_score=result.get("top_rerank_score"),
        sources_count=len(result["answer"]["sources"]),
        latency_ms=latency_ms,
    )

    return AskResponse(
        answer=result["answer"]["answer"],
        sources=result["answer"]["sources"],
        abstained=result.get("abstained", False),
    )