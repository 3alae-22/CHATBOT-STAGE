from unittest.mock import MagicMock, patch
import numpy as np

from app.repositories.vector_repository import VectorRepository
from app.services.retrieval_service import RetrievalService


@patch("app.services.retrieval_service.embed_query")
def test_search_delegates_to_repository_with_correct_k_and_lang_filter(
    mock_embed_query, fake_search_results
):
    mock_embed_query.return_value = np.zeros(1024, dtype=np.float32)

    mock_repo = MagicMock(spec=VectorRepository)
    mock_repo.search_similar.return_value = fake_search_results

    retrieval_service = RetrievalService(mock_repo)
    result = retrieval_service.search("C'est quoi le marché public ?", k=3, lang="ar")

    assert result == fake_search_results

    mock_repo.search_similar.assert_called_once()
    _, kwargs = mock_repo.search_similar.call_args
    assert kwargs["k"] == 3
    assert kwargs["lang"] == "ar"

    mock_embed_query.assert_called_once_with("C'est quoi le marché public ?")