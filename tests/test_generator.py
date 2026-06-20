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
            "score": 0.9,
            "snippet": "If your deposit is not credited, check confirmations and contact support.",
        },
        {
            "doc_id": "password_reset",
            "title": "Password Reset",
            "score": 0.8,
            "snippet": "After resetting your password, withdrawals may be temporarily restricted for security reasons.",
        },
        {
            "doc_id": "account_security",
            "title": "Account Security",
            "score": 0.7,
            "snippet": "Account security checks may temporarily block transfers after credential changes.",
        },
    ]

    result = generator.generate(
        question="I changed my login credentials and now transfers are blocked.",
        retrieved_docs=docs,
    )

    assert result["is_supported"] is True

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
            "title": "Deposit not credited",
            "score": 0.9,
            "snippet": "If your deposit is not credited, check confirmations and contact support.",
        },
        {
            "doc_id": "doc_2",
            "title": "Account verification",
            "score": 0.8,
            "snippet": "Account verification may be required before some transactions are processed.",
        },
        {
            "doc_id": "doc_3",
            "title": "Wallet address",
            "score": 0.7,
            "snippet": "Always confirm the wallet address before making a transfer.",
        },
    ]

    result = generator.generate(
        question="My deposit was not credited to my account.",
        retrieved_docs=docs,
    )

    assert result["is_supported"] is True
    assert result["citations"] == ["doc_1", "doc_2"]
    assert "doc_3" not in result["citations"]


def test_generator_refuses_without_docs():
    generator = GroundedAnswerGenerator()

    result = generator.generate(
        question="What is the weather tomorrow?",
        retrieved_docs=[],
    )

    assert result["is_supported"] is False
    assert result["citations"] == []
    assert "could not find relevant support documents" in result["answer"]


def test_generator_refuses_out_of_domain_question_with_retrieved_docs():
    generator = GroundedAnswerGenerator()

    docs = [
        {
            "doc_id": "deposit_not_credited",
            "title": "Deposit Not Credited",
            "score": 0.9,
            "snippet": "If your deposit is not credited, check confirmations and contact support.",
        }
    ]

    result = generator.generate(
        question="Who is the CEO of Apple?",
        retrieved_docs=docs,
    )

    assert result["is_supported"] is False
    assert result["citations"] == []
    assert "outside the support-document domain" in result["reason"]
