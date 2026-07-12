from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.services.generation_service import GenerationService
from app.core.dependencies import get_generation_service

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    k: int = 5


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(
    request: AskRequest,
    generation_service: GenerationService = Depends(get_generation_service),
):
    try:
        result = generation_service.ask(request.question, k=request.k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lors de la génération : {e}")

    return AskResponse(answer=result["answer"], sources=result["sources"])