from app.dense_retriever import DenseRetriever


def test_dense_retriever_returns_top_k_docs():
    retriever = DenseRetriever()
    results = retriever.retrieve(
        "My deposit has not arrived. What should I check?",
        top_k=3,
    )

    assert len(results) == 3
    assert "doc_id" in results[0]
    assert "score" in results[0]
    assert "snippet" in results[0]