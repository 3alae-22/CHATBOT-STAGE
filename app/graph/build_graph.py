from langgraph.graph import StateGraph, END
from app.graph.state import GraphState
from app.graph.nodes import (
    make_rewrite_query_node,
    make_retrieve_node,
    make_generate_node,
    make_check_grounding_node,
    make_abstain_node,
    RETRIEVAL_SCORE_THRESHOLD,
)
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService


def _route_after_retrieve(state: GraphState) -> str:
    """Score de retrieval trop faible → abstention directe, sans générer."""
    if state["top_rerank_score"] < RETRIEVAL_SCORE_THRESHOLD:
        return "abstain"
    return "generate"


def _route_after_grounding_check(state: GraphState) -> str:
    """Réponse générée mais non ancrée dans les sources → abstention."""
    if not state["is_grounded"]:
        return "abstain"
    return "end"


def build_graph(retrieval_service: RetrievalService, generation_service: GenerationService, llm_client):
    graph = StateGraph(GraphState)

    graph.add_node("rewrite_query", make_rewrite_query_node(llm_client))
    graph.add_node("retrieve", make_retrieve_node(retrieval_service))
    graph.add_node("generate", make_generate_node(generation_service))
    graph.add_node("check_grounding", make_check_grounding_node(llm_client))
    graph.add_node("abstain", make_abstain_node())

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"abstain": "abstain", "generate": "generate"},
    )

    graph.add_edge("generate", "check_grounding")

    graph.add_conditional_edges(
        "check_grounding",
        _route_after_grounding_check,
        {"abstain": "abstain", "end": END},
    )

    graph.add_edge("abstain", END)

    return graph.compile()