from google.genai import types

from app.graph.state import GraphState
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.core.embeddings import detect_lang


REFORMULATION_INSTRUCTION = """
You translate a user's question into {target_lang_name} for the purpose of
document retrieval — not for natural conversational translation.

Rules:
- Output ONLY the translated/reformulated question, nothing else.
- Favor vocabulary likely to appear in official Moroccan administrative/legal
  texts in the target language (formal legal register), even if it diverges
  slightly from a literal word-for-word translation.
- Do not add explanations, quotes marks, or notes.
"""

GROUNDING_CHECK_INSTRUCTION = """
You verify whether a generated answer is fully supported by the provided
source excerpts, for an administrative/legal RAG system where hallucination
has real consequences.

Rules:
- Answer with EXACTLY one word: "OUI" if every factual claim in the answer
  is explicitly supported by the excerpts, or "NON" if the answer contains
  any claim not present in the excerpts, or if the excerpts don't actually
  address the question.
- Be strict: partial support or plausible-sounding inference counts as "NON".
- Output ONLY "OUI" or "NON", nothing else.
"""

RETRIEVAL_SCORE_THRESHOLD = 0.05

ABSTENTION_MESSAGE_FR = "Je ne trouve pas cette information dans les documents fournis."
ABSTENTION_MESSAGE_AR = "لا أجد هذه المعلومة في الوثائق المتوفرة."


def make_rewrite_query_node(llm_client, model: str = "gemini-2.5-flash"):
    def rewrite_query_node(state: GraphState) -> dict:
        source_lang = state.get("lang") or detect_lang(state["question"])
        target_lang = "ar" if source_lang == "fr" else "fr"
        target_lang_name = "Arabic" if target_lang == "ar" else "French"

        response = llm_client.models.generate_content(
            model=model,
            contents=state["question"],
            config=types.GenerateContentConfig(
                system_instruction=REFORMULATION_INSTRUCTION.format(target_lang_name=target_lang_name)
            ),
        )
        return {
            "lang": source_lang,
            "translated_question": response.text.strip(),
            "translated_lang": target_lang,
        }
    return rewrite_query_node


def make_retrieve_node(retrieval_service: RetrievalService):
    def retrieve_node(state: GraphState) -> dict:
        candidates_original = retrieval_service.retrieve_candidates(
            query=state["question"], lang=state["lang"], k=state["k"]
        )
        candidates_translated = retrieval_service.retrieve_candidates(
            query=state["translated_question"], lang=state["translated_lang"], k=state["k"]
        )
        combined = {row[0]: row for row in candidates_original}
        for row in candidates_translated:
            combined.setdefault(row[0], row)
        pool = list(combined.values())

        reranked_with_scores = retrieval_service.rerank_candidates_with_scores(
            query=state["question"], candidates=pool, top_n=6
        )
        top_score = reranked_with_scores[0][1] if reranked_with_scores else 0.0
        chunks = [row for row, score in reranked_with_scores]

        return {"chunks": chunks, "top_rerank_score": top_score}
    return retrieve_node


def make_generate_node(generation_service: GenerationService):
    def generate_node(state: GraphState) -> dict:
        result = generation_service.ask(question=state["question"], search_results=state["chunks"])
        return {"answer": result}
    return generate_node


def make_check_grounding_node(llm_client, model: str = "gemini-2.5-flash"):
    def check_grounding_node(state: GraphState) -> dict:
        answer_text = state["answer"]["answer"]
        context = "\n\n".join(row[3] for row in state["chunks"])

        prompt = f"Excerpts:\n{context}\n\nGenerated answer:\n{answer_text}"

        response = llm_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=GROUNDING_CHECK_INSTRUCTION),
        )
        is_grounded = response.text.strip().upper().startswith("OUI")
        return {"is_grounded": is_grounded}
    return check_grounding_node


def make_abstain_node():
    def abstain_node(state: GraphState) -> dict:
        message = ABSTENTION_MESSAGE_AR if state["lang"] == "ar" else ABSTENTION_MESSAGE_FR
        return {
            "answer": {"answer": message, "sources": []},
            "abstained": True,
        }
    return abstain_node