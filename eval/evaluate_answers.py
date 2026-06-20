import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from app.main import app


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def citation_recall(citations: List[str], gold_doc_ids: List[str]) -> float:
    if not gold_doc_ids:
        return 1.0 if not citations else 0.0

    return len(set(citations) & set(gold_doc_ids)) / len(set(gold_doc_ids))


def citation_precision(citations: List[str], gold_doc_ids: List[str]) -> float:
    if not citations:
        return 1.0 if not gold_doc_ids else 0.0

    return len(set(citations) & set(gold_doc_ids)) / len(set(citations))


def citation_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def evaluate(
    eval_path: Path,
    retriever: str,
    top_k: int,
    alpha: float,
) -> Dict[str, Any]:
    client = TestClient(app)
    rows = load_jsonl(eval_path)

    results = []

    label_correct = 0

    supported_total = 0
    supported_citation_recall_sum = 0.0
    supported_citation_precision_sum = 0.0
    supported_citation_f1_sum = 0.0

    unsupported_total = 0
    unsupported_correct = 0
    unsupported_citation_violations = 0

    for row in rows:
        question = row["question"]
        expected_is_supported = row["expected_is_supported"]
        gold_doc_ids = row["gold_doc_ids"]

        payload = {
            "question": question,
            "top_k": top_k,
            "retriever": retriever,
            "alpha": alpha,
        }

        response = client.post("/answer", json=payload)
        response.raise_for_status()

        answer = response.json()

        predicted_is_supported = answer["is_supported"]
        citations = answer.get("citations", [])

        is_label_correct = predicted_is_supported == expected_is_supported
        label_correct += int(is_label_correct)

        row_result = {
            "question": question,
            "expected_is_supported": expected_is_supported,
            "predicted_is_supported": predicted_is_supported,
            "gold_doc_ids": gold_doc_ids,
            "citations": citations,
            "label_correct": is_label_correct,
            "citation_recall": None,
            "citation_precision": None,
            "citation_f1": None,
            "reason": answer.get("reason"),
        }

        if expected_is_supported:
            supported_total += 1

            recall = citation_recall(citations, gold_doc_ids)
            precision = citation_precision(citations, gold_doc_ids)
            f1 = citation_f1(precision, recall)

            supported_citation_recall_sum += recall
            supported_citation_precision_sum += precision
            supported_citation_f1_sum += f1

            row_result["citation_recall"] = recall
            row_result["citation_precision"] = precision
            row_result["citation_f1"] = f1
        else:
            unsupported_total += 1

            if not predicted_is_supported:
                unsupported_correct += 1

            if citations:
                unsupported_citation_violations += 1

        results.append(row_result)

    total = len(rows)

    return {
        "eval_path": str(eval_path),
        "retriever": retriever,
        "top_k": top_k,
        "alpha": alpha,
        "total": total,
        "answerability_accuracy": label_correct / total if total else 0.0,
        "supported_count": supported_total,
        "unsupported_count": unsupported_total,
        "supported_citation_recall": (
            supported_citation_recall_sum / supported_total
            if supported_total
            else 0.0
        ),
        "supported_citation_precision": (
            supported_citation_precision_sum / supported_total
            if supported_total
            else 0.0
        ),
        "supported_citation_f1": (
            supported_citation_f1_sum / supported_total
            if supported_total
            else 0.0
        ),
        "unsupported_refusal_accuracy": (
            unsupported_correct / unsupported_total
            if unsupported_total
            else 0.0
        ),
        "unsupported_citation_violations": unsupported_citation_violations,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-path",
        type=Path,
        default=Path("data/eval_answer_questions.jsonl"),
    )
    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.3)

    args = parser.parse_args()

    metrics = evaluate(
        eval_path=args.eval_path,
        retriever=args.retriever,
        top_k=args.top_k,
        alpha=args.alpha,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()