import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        docs_path: str = "data/raw_docs_sample.jsonl",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.docs = self._load_jsonl(docs_path)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.search_texts = [
            f"{doc.get('title', '')}. {doc.get('text', '')}"
            for doc in self.docs
        ]

        self.doc_embeddings = self._encode(self.search_texts)

    def _load_jsonl(self, path: str) -> List[Dict[str, str]]:
        docs = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        return docs

    def _encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        start = time.perf_counter()

        query_embedding = self._encode([query])[0]

        # cosine similarity because embeddings are L2-normalized
        scores = self.doc_embeddings @ query_embedding

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        latency_ms = (time.perf_counter() - start) * 1000

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            doc = self.docs[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "score": round(float(scores[idx]), 4),
                    "snippet": doc["text"][:240],
                    "latency_ms": round(latency_ms, 2),
                }
            )

        return results