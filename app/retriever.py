import json
from pathlib import Path
from typing import List, Dict

from app.bm25 import BM25


class SupportRetriever:
    def __init__(self, docs_path: str = "data/raw_docs_sample.jsonl"):
        self.docs = self._load_jsonl(docs_path)
        self.bm25 = BM25(self.docs)

    def _load_jsonl(self, path: str) -> List[Dict[str, str]]:
        docs = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        return docs

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        results = self.bm25.search(query, top_k=top_k)

        return [
            {
                "rank": rank + 1,
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "score": round(score, 4),
                "snippet": doc["text"][:240],
            }
            for rank, (doc, score) in enumerate(results)
        ]