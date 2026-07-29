

def fuse_rrf(
    dense_results: list[tuple],
    lexical_results: list[tuple],
    k: int = 60,
) -> list[tuple]:
    """
        Fuses dense and lexical retrieval results via Reciprocal Rank Fusion (RRF).

        Fusion by RANK, not by raw score: cosine distance (dense, lower = better)
        and ts_rank (lexical, higher = better) are not on the same scale nor in
        the same direction. Comparing them directly would be a silent error.
        RRF ignores the original score and only looks at position within each list.

        Formula: score(doc) = sum, over the lists in which the doc appears,
        of 1 / (k + rank), rank starting at 1. k=60 by default (standard value
        from the original RRF paper).

        Args:
            dense_results: tuples (id, pdf_name, page_num, chunk_text, distance),
                already sorted by decreasing dense relevance (increasing distance).
            lexical_results: tuples (id, pdf_name, page_num, chunk_text, rank),
                already sorted by decreasing lexical relevance.
            k: RRF constant.

        Returns:
            List of tuples (id, pdf_name, page_num, chunk_text, rrf_score),
            sorted by decreasing rrf_score.
    """
    scores: dict[int, float] = {}
    docs: dict[int, tuple] = {}

    for rank, row in enumerate(dense_results, start=1):
        doc_id = row[0]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = row

    for rank, row in enumerate(lexical_results, start=1):
        doc_id = row[0]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs.setdefault(doc_id, row)

    fused = [
        (*docs[doc_id][:4], scores[doc_id])
        for doc_id in scores
    ]
    fused.sort(key=lambda t: t[-1], reverse=True)
    return fused