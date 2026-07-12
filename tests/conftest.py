import pytest
from unittest.mock import MagicMock


@pytest.fixture
def fake_search_results():

    return [
        (1, "Decret_marches_publics_n_2_22_431", 13,
         "Le marché public est un contrat administratif.", 0.12),
        (2, "Decret_marches_publics_n_2_22_431", 29,
         "Les marchés publics sont soumis au décret 2.22.431.", 0.18),
        (3, "qanoun_wadifa_omoumia", 19,
         "La province de Benslimane organise les appels d'offres.", 0.25),
    ]


@pytest.fixture
def fake_chunks_to_insert():
    return [
        {"pdf_name": "pdf_1", "page_num": 13,
         "chunk": "Le marché public est un contrat administratif.", "lang": "fr"},
        {"pdf_name": "pdf_4", "page_num": 29,
         "chunk": "Les marchés publics sont soumis au décret 2.22.431.", "lang": "fr"},
        {"pdf_name": "pdf_3", "page_num": 19,
         "chunk": "La province de Benslimane organise les appels d'offres.", "lang": "fr"},
    ]


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    client.models.generate_content.return_value.text = (
        "Le marché public est un contrat administratif conclu entre "
        "une administration publique et un prestataire (Source 1, page 13). "
        "Ces marchés sont régis par le décret n° 2.22.431 (Source 2, page 29)."
    )
    return client