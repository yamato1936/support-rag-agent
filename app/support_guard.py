import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SupportDecision:
    is_supported: bool
    reason: str


OUT_OF_DOMAIN_TERMS = {
    "weather",
    "temperature",
    "forecast",
    "ceo",
    "president",
    "prime",
    "minister",
    "apple",
    "google",
    "openai",
    "stock",
    "bitcoin",
    "football",
    "baseball",
    "recipe",
    "movie",
    "capital",
}

SUPPORT_DOMAIN_TERMS = {
    "account",
    "login",
    "password",
    "credential",
    "credentials",
    "reset",
    "transfer",
    "withdraw",
    "withdrawal",
    "deposit",
    "kyc",
    "verification",
    "2fa",
    "authentication",
    "fee",
    "fees",
    "blocked",
    "suspended",
    "transaction",
    "wallet",
    "address",
    "credited",
    "support",
    "ticket",
    "funds",
    "locked",
    "restricted",
}

SYNONYMS = {
    "credentials": {"password", "login", "account"},
    "credential": {"password", "login", "account"},
    "blocked": {"suspended", "restricted", "locked"},
    "money": {"funds", "deposit", "withdrawal", "transfer"},
    "send": {"transfer", "withdrawal"},
    "receive": {"deposit", "credited"},
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokens(text: str) -> set[str]:
    return set(_normalize(text).split())


def _doc_to_text(doc: Any) -> str:
    keys = ("doc_id", "id", "title", "snippet", "content", "text", "body")

    if isinstance(doc, Mapping):
        return " ".join(str(doc[key]) for key in keys if doc.get(key))

    return " ".join(
        str(getattr(doc, key))
        for key in keys
        if getattr(doc, key, None)
    )


def _expand_query_tokens(query_tokens: set[str]) -> set[str]:
    expanded = set(query_tokens)

    for token in query_tokens:
        expanded.update(SYNONYMS.get(token, set()))

    return expanded


def assess_support(question: str, retrieved_docs: Sequence[Any]) -> SupportDecision:
    question_tokens = _tokens(question)

    if not question_tokens:
        return SupportDecision(False, "Empty question.")

    if question_tokens & OUT_OF_DOMAIN_TERMS:
        return SupportDecision(False, "Question is outside the support-document domain.")

    if not retrieved_docs:
        return SupportDecision(False, "No retrieved support documents.")

    expanded_query_tokens = _expand_query_tokens(question_tokens)

    if expanded_query_tokens & SUPPORT_DOMAIN_TERMS:
        return SupportDecision(True, "Question contains support-domain intent.")

    docs_text = " ".join(_doc_to_text(doc) for doc in retrieved_docs)
    docs_tokens = _tokens(docs_text)

    overlap = expanded_query_tokens & docs_tokens

    if len(overlap) >= 2:
        return SupportDecision(
            True,
            "Question has sufficient lexical overlap with retrieved support documents.",
        )

    return SupportDecision(False, "No relevant support-document signal found.")