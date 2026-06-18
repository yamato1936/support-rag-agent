import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from app.retriever import SupportRetriever
from app.dense_retriever import DenseRetriever


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate_retriever(
    name: str,
    retriever,
    eval_path: str,
    top_k: int = 5,
) -> Tuple[float, float, float]:
    questions = load_jsonl(eval_path)

    hits = 0
    reciprocal_ranks = []
    latencies = []

    print("=" * 80)
    print(f"Evaluating: {name}")
    print(f"Eval file: {eval_path}")
    print("=" * 80)

    for row in questions:
        question = row["question"]
        gold_doc_ids = set(row["gold_doc_ids"])

        start = time.perf_counter()
        results = retriever.retrieve(question, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [r["doc_id"] for r in results]

        hit = bool(gold_doc_ids.intersection(retrieved_ids))
        hits += int(hit)

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in gold_doc_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        print(f"Q: {question}")
        print(f"Gold: {list(gold_doc_ids)}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Hit: {hit}")
        print(f"Latency: {latency_ms:.2f} ms")
        print("-" * 80)

    recall_at_k = hits / len(questions)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    avg_latency = sum(latencies) / len(latencies)

    print(f"\n{name} RESULT")
    print(f"Recall@{top_k}: {recall_at_k:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print()

    return recall_at_k, mrr, avg_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-path",
        default="data/eval_questions.jsonl",
        help="Path to evaluation questions JSONL file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved documents.",
    )
    args = parser.parse_args()

    bm25 = SupportRetriever()
    dense = DenseRetriever()

    bm25_metrics = evaluate_retriever(
        "BM25 v0.2",
        bm25,
        eval_path=args.eval_path,
        top_k=args.top_k,
    )
    dense_metrics = evaluate_retriever(
        "Dense Retrieval",
        dense,
        eval_path=args.eval_path,
        top_k=args.top_k,
    )

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Eval file: {args.eval_path}")
    print(f"Top-k: {args.top_k}")
    print()
    print("| Method | Recall@5 | MRR | Avg Latency |")
    print("|---|---:|---:|---:|")
    print(
        f"| BM25 v0.2 | {bm25_metrics[0]:.4f} | {bm25_metrics[1]:.4f} | {bm25_metrics[2]:.2f} ms |"
    )
    print(
        f"| Dense Retrieval | {dense_metrics[0]:.4f} | {dense_metrics[1]:.4f} | {dense_metrics[2]:.2f} ms |"
    )


if __name__ == "__main__":
    main()
