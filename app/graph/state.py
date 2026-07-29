from typing import TypedDict


class GraphState(TypedDict):
    question: str
    lang: str
    translated_question: str
    translated_lang: str
    k: int
    chunks: list[tuple]
    top_rerank_score: float
    answer: dict
    is_grounded: bool
    abstained: bool