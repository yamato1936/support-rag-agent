import json
import time
from pathlib import Path
from typing import List, Dict

from app.retriever import SupportRetriever


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate(top_k: int = 5):
    retriever = SupportRetriever()
    questions = load_jsonl("data/eval_questions.jsonl")

    hits = 0
    reciprocal_ranks = []
    latencies = []

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

        print("=" * 80)
        print(f"Q: {question}")
        print(f"Gold: {list(gold_doc_ids)}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Hit: {hit}")
        print(f"Latency: {latency_ms:.2f} ms")

    recall_at_k = hits / len(questions)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    avg_latency = sum(latencies) / len(latencies)

    print("\nRESULT")
    print(f"Recall@{top_k}: {recall_at_k:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"Avg Latency: {avg_latency:.2f} ms")


if __name__ == "__main__":
    evaluate(top_k=5)