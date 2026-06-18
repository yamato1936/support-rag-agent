from app.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_returns_top_k_docs():
    retriever = HybridRetriever(alpha=0.5)
    results = retriever.retrieve(
        "My deposit has not arrived. What should I check?",
        top_k=3,
    )

    assert len(results) == 3
    assert "doc_id" in results[0]
    assert "score" in results[0]
    assert "bm25_score" in results[0]
    assert "dense_score" in results[0]
    assert "bm25_norm" in results[0]
    assert "dense_norm" in results[0]


def test_hybrid_retriever_validates_alpha():
    try:
        HybridRetriever(alpha=1.5)
    except ValueError:
        return

    raise AssertionError("HybridRetriever should reject alpha > 1.0")
