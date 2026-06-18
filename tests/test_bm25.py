from app.bm25 import tokenize
from app.retriever import SupportRetriever


def test_tokenize_normalizes_plural():
    tokens = tokenize("Withdrawals are delayed")
    assert "withdrawal" in tokens
    assert "delay" in tokens


def test_tokenize_normalizes_long_time_phrase():
    tokens = tokenize("Withdrawal is taking a long time")
    assert "withdrawal" in tokens
    assert "delay" in tokens


def test_retrieve_deposit_issue():
    retriever = SupportRetriever()
    results = retriever.retrieve(
        "My USDT deposit has not arrived. What should I check?",
        top_k=5,
    )
    doc_ids = [r["doc_id"] for r in results]
    assert "deposit_not_credited" in doc_ids


def test_retrieve_withdrawal_delay_rank1():
    retriever = SupportRetriever()
    results = retriever.retrieve(
        "Why is my withdrawal taking a long time?",
        top_k=5,
    )
    assert results[0]["doc_id"] == "withdrawal_delay"