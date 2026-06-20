from typing import Dict, List
from app.support_guard import assess_support


class GroundedAnswerGenerator:
    """
    Deterministic citation-grounded answer generator.

    This version does not call an external LLM.
    It builds an answer only from retrieved support documents.

    Design goal:
    - avoid hallucination
    - avoid over-trusting rank 1
    - refuse unsupported / out-of-domain questions
    - expose citations clearly
    - prepare for later LLM-based generation
    """

    def __init__(
        self,
        min_relevance_score: float = 0.0,
        max_answer_docs: int = 2,
    ):
        self.min_relevance_score = min_relevance_score
        self.max_answer_docs = max_answer_docs

    def generate(self, question: str, retrieved_docs: List[Dict]) -> Dict:
        if not retrieved_docs:
            return self._unsupported_response(
                "I could not find relevant support documents for this question."
            )

        useful_docs = self._filter_useful_docs(retrieved_docs)

        if not useful_docs:
            return self._unsupported_response(
                "I found some documents, but none were relevant enough to answer safely."
            )

        support_decision = assess_support(question, useful_docs)

        if not support_decision.is_supported:
            return self._unsupported_response(support_decision.reason)

        answer_docs = useful_docs[: self.max_answer_docs]
        citations = [doc["doc_id"] for doc in answer_docs]

        answer = self._build_grounded_answer(answer_docs, citations)

        return {
            "answer": answer,
            "citations": citations,
            "is_supported": True,
            "reason": "Answered using retrieved support documents.",
        }

    def _unsupported_response(self, reason: str) -> Dict:
        return {
            "answer": (
                f"{reason} I do not have enough information in the support "
                "documents to answer this question safely."
            ),
            "citations": [],
            "is_supported": False,
            "reason": reason,
        }

    def _filter_useful_docs(self, retrieved_docs: List[Dict]) -> List[Dict]:
        useful_docs = []

        for doc in retrieved_docs:
            score = doc.get("score", 0.0)

            if score is not None and score >= self.min_relevance_score:
                useful_docs.append(doc)

        return useful_docs

    def _build_grounded_answer(self, docs: List[Dict], citations: List[str]) -> str:
        parts = []

        parts.append(
            "I found the following relevant support guidance based only on the retrieved documents. "
        )

        for index, doc in enumerate(docs, start=1):
            title = doc["title"]
            snippet = doc["snippet"]
            doc_id = doc["doc_id"]

            parts.append(
                f"{index}. {title}: {snippet} "
                f"[citation: {doc_id}] "
            )

        parts.append(
            "If these documents do not match the exact situation, the question should be escalated to support. "
        )

        parts.append(f"Citations: {', '.join(citations)}.")

        return "".join(parts)
