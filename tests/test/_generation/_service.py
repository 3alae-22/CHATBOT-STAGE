import pytest
from unittest.mock import MagicMock

from app.services.generation_service import (
    GenerationService,
    USER_PROMPT_TEMPLATE,
)


def test_generate_uses_instance_model_and_instruction(mock_gemini_client):

    service = GenerationService(
        llm_client=mock_gemini_client,
        retrieval_service=MagicMock(),
        model="gemini-2.5-pro",   
        instruction="CUSTOM INSTRUCTION",
    )

    service.generate("un prompt")

    _, kwargs = mock_gemini_client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-pro"
    assert kwargs["config"].system_instruction == "CUSTOM INSTRUCTION"


def test_build_prompt_formats_question_and_context(fake_search_results):

    mock_llm_client = MagicMock()
    mock_retrieval_service = MagicMock()

    service = GenerationService(
        llm_client=mock_llm_client,
        retrieval_service=mock_retrieval_service,
    )

    question = "C'est quoi le marché public ?"
    result = service.build_prompt(question, fake_search_results)

    expected_context_lines = []
    for i, row in enumerate(fake_search_results, start=1):
        _, pdf_name, page_num, chunk_text, distance = row
        expected_context_lines.append(f"[Source {i} — {pdf_name}, page {page_num}]")
        expected_context_lines.append(chunk_text.strip())
        expected_context_lines.append("")
    expected_context = "\n".join(expected_context_lines).strip()

    expected_prompt = USER_PROMPT_TEMPLATE.format(
        question=question, context=expected_context
    ).strip()

    assert result == expected_prompt


def test_ask_returns_sources_with_correct_shape(fake_search_results, mock_gemini_client):
    from app.services.generation_service import GenerationService

    mock_retrieval = MagicMock()
    mock_retrieval.search.return_value = fake_search_results

    service = GenerationService(
        llm_client=mock_gemini_client,
        retrieval_service=mock_retrieval,
    )

    result = service.ask("C'est quoi le marché public ?", k=3)

    assert result["sources"] == [
        {"pdf_name": "Decret_marches_publics_n_2_22_431", "page_num": 13, "chunk_text": "Le marché public est un contrat administratif."},
        {"pdf_name": "Decret_marches_publics_n_2_22_431", "page_num": 29, "chunk_text": "Les marchés publics sont soumis au décret 2.22.431."},
        {"pdf_name": "qanoun_wadifa_omoumia", "page_num": 19, "chunk_text": "La province de Benslimane organise les appels d'offres."},
    ]