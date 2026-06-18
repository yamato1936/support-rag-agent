from app.generator import GroundedAnswerGenerator


def test_generator_returns_supported_answer():
    generator = GroundedAnswerGenerator()

    docs = [
        {
            "doc_id": "password_reset",
            "title": "Password Reset",
            "score": 0.9,
            "snippet": "After resetting your password, withdrawals may be temporarily restricted for security reasons.",
        }
    ]

    result = generator.generate(
        question="I changed my login credentials and now transfers are blocked.",
        retrieved_docs=docs,
    )

    assert result["is_supported"] is True
    assert "password_reset" in result["citations"]
    assert "Password Reset" in result["answer"]
    assert "citation: password_reset" in result["answer"]


def test_generator_uses_multiple_documents():
    generator = GroundedAnswerGenerator(max_answer_docs=3)

    docs = [
        {
            "doc_id": "deposit_not_credited",
            "title": "Deposit Not Credited",
            "score": 0.7,
            "snippet": "If your crypto deposit has not arrived, check blockchain confirmations.",
        },
        {
            "doc_id": "password_reset",
            "title": "Password Reset",
            "score": 0.69,
            "snippet": "After resetting your password, withdrawals may be temporarily restricted.",
        },
        {
            "doc_id": "account_security",
            "title": "Account Security",
            "score": 0.59,
            "snippet": "Enable two-factor authentication and use a strong password.",
        },
    ]

    result = generator.generate(
        question="I changed my login credentials and now transfers are blocked.",
        retrieved_docs=docs,
    )

    assert result["is_supported"] is True

    # It should not only use the first document.
    assert "deposit_not_credited" in result["citations"]
    assert "password_reset" in result["citations"]
    assert "account_security" in result["citations"]

    assert "Deposit Not Credited" in result["answer"]
    assert "Password Reset" in result["answer"]
    assert "Account Security" in result["answer"]


def test_generator_limits_number_of_answer_documents():
    generator = GroundedAnswerGenerator(max_answer_docs=2)

    docs = [
        {
            "doc_id": "doc_1",
            "title": "Document 1",
            "score": 0.9,
            "snippet": "Snippet 1.",
        },
        {
            "doc_id": "doc_2",
            "title": "Document 2",
            "score": 0.8,
            "snippet": "Snippet 2.",
        },
        {
            "doc_id": "doc_3",
            "title": "Document 3",
            "score": 0.7,
            "snippet": "Snippet 3.",
        },
    ]

    result = generator.generate(
        question="test question",
        retrieved_docs=docs,
    )

    assert result["is_supported"] is True
    assert result["citations"] == ["doc_1", "doc_2"]
    assert "Document 1" in result["answer"]
    assert "Document 2" in result["answer"]
    assert "Document 3" not in result["answer"]


def test_generator_refuses_without_docs():
    generator = GroundedAnswerGenerator()

    result = generator.generate(
        question="What is the weather tomorrow?",
        retrieved_docs=[],
    )

    assert result["is_supported"] is False
    assert result["citations"] == []
    assert "could not find relevant support documents" in result["answer"]
