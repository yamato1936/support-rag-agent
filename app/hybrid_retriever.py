from typing import Dict, List

import numpy as np

from app.retriever import SupportRetriever
from app.dense_retriever import DenseRetriever


class HybridRetriever:
    """
    Hybrid retriever combining BM25 lexical scores and dense embedding scores.

    Score:
        hybrid_score = alpha * normalized_bm25_score
                     + (1 - alpha) * normalized_dense_score

    alpha:
        - alpha = 1.0: BM25 only
        - alpha = 0.0: Dense only
        - alpha = 0.5: equal weighting
    """

    def __init__(
        self,
        docs_path: str = "data/raw_docs_sample.jsonl",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        alpha: float = 0.5,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        self.alpha = alpha
        self.bm25_retriever = SupportRetriever(docs_path=docs_path)
        self.dense_retriever = DenseRetriever(
            docs_path=docs_path,
            model_name=model_name,
        )

        self.docs = self.bm25_retriever.docs

    def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))

        if max_score - min_score < 1e-8:
            return np.zeros_like(scores, dtype=np.float32)

        return ((scores - min_score) / (max_score - min_score)).astype(np.float32)

    def _bm25_scores(self, query: str) -> np.ndarray:
        scores = [
            self.bm25_retriever.bm25.score(query, doc_index)
            for doc_index in range(len(self.docs))
        ]
        return np.array(scores, dtype=np.float32)

    def _dense_scores(self, query: str) -> np.ndarray:
        query_embedding = self.dense_retriever._encode([query])[0]

        # Cosine similarity because DenseRetriever uses normalized embeddings.
        scores = self.dense_retriever.doc_embeddings @ query_embedding
        return scores.astype(np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        bm25_scores = self._bm25_scores(query)
        dense_scores = self._dense_scores(query)

        bm25_norm = self._min_max_normalize(bm25_scores)
        dense_norm = self._min_max_normalize(dense_scores)

        hybrid_scores = self.alpha * bm25_norm + (1.0 - self.alpha) * dense_norm

        ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            doc = self.docs[int(idx)]

            results.append(
                {
                    "rank": rank,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "score": round(float(hybrid_scores[idx]), 4),
                    "bm25_score": round(float(bm25_scores[idx]), 4),
                    "dense_score": round(float(dense_scores[idx]), 4),
                    "bm25_norm": round(float(bm25_norm[idx]), 4),
                    "dense_norm": round(float(dense_norm[idx]), 4),
                    "alpha": self.alpha,
                    "snippet": doc["text"][:240],
                }
            )

        return results
