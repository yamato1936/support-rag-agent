from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_bm25_endpoint():
    response = client.post(
        "/query",
        json={
            "question": "My deposit has not arrived. What should I check?",
            "top_k": 3,
            "retriever": "bm25",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retriever"] == "bm25"
    assert len(data["retrieved_docs"]) == 3
    assert "deposit_not_credited" in data["citations"]


def test_query_dense_endpoint():
    response = client.post(
        "/query",
        json={
            "question": "I changed my login credentials and now transfers are blocked.",
            "top_k": 5,
            "retriever": "dense",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retriever"] == "dense"
    assert len(data["retrieved_docs"]) == 5


def test_query_hybrid_endpoint():
    response = client.post(
        "/query",
        json={
            "question": "I changed my login credentials and now transfers are blocked.",
            "top_k": 5,
            "retriever": "hybrid",
            "alpha": 0.3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["retriever"] == "hybrid"
    assert data["alpha"] == 0.3
    assert len(data["retrieved_docs"]) == 5
    assert "bm25_score" in data["retrieved_docs"][0]
    assert "dense_score" in data["retrieved_docs"][0]


def test_answer_endpoint_returns_grounded_answer():
    response = client.post(
        "/answer",
        json={
            "question": "I changed my login credentials and now transfers are blocked.",
            "top_k": 5,
            "retriever": "hybrid",
            "alpha": 0.3,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["retriever"] == "hybrid"
    assert data["alpha"] == 0.3
    assert data["is_supported"] is True
    assert len(data["citations"]) > 0
    assert len(data["retrieved_docs"]) == 5
    assert "answer" in data

    # The answer should include multiple grounded documents, not only rank 1.
    assert "password_reset" in data["citations"]
    assert "Password Reset" in data["answer"]
