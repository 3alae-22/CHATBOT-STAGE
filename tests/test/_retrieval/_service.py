import pytest
from unittest.mock import MagicMock

from app.repositories.vector_repository import VectorRepository
from app.services.retrieval_service import RetrievalService


def test_search_delegates_to_repository_with_correct_k_and_lang_filter(fake_search_results):

    mock_repo = MagicMock(spec=VectorRepository)
    mock_repo.search_similar.return_value = fake_search_results

    retrieval_service = RetrievalService(mock_repo)

    result = retrieval_service.search("C'est quoi le marché public ?", k=3, lang="ar")

    assert result == fake_search_results

    mock_repo.search_similar.assert_called_once()
    _, kwargs = mock_repo.search_similar.call_args
    assert kwargs["k"] == 3
    assert kwargs["lang"] == "ar"
