import math
import re
from collections import Counter
from typing import List, Dict, Tuple


STOPWORDS = {
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did",
    "i", "my", "me", "you", "your",
    "what", "why", "how", "when", "where", "which",
    "to", "of", "in", "on", "for", "with", "at", "by", "from",
    "and", "or", "but",
    "has", "have", "had",
    "not", "can", "could", "should", "would",
    "it", "this", "that",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("taking a long time", "delay")
    text = text.replace("takes a long time", "delay")
    text = text.replace("long time", "delay")
    text = text.replace("not arrived", "not credited")
    text = text.replace("not received", "not credited")
    return text

def normalize_token(token: str) -> str:
    token = token.lower()

    # plural normalization: withdrawals -> withdrawal
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and len(token) > 3:
        token = token[:-1]

    # simple verb normalization: delayed -> delay, taking -> tak
    if token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    elif token.endswith("ing") and len(token) > 5:
        token = token[:-3]

    return token


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    raw_tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    tokens = []

    for token in raw_tokens:
        token = normalize_token(token)
        if token and token not in STOPWORDS:
            tokens.append(token)

    return tokens


class BM25:
    def __init__(self, documents: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b

        # Include title because support documents often encode the core intent in the title.
        self.search_texts = [f"{doc.get('title', '')} {doc.get('title', '')} {doc.get('title', '')} {doc.get('text', '')}" for doc in documents]

        self.tokenized_docs = [tokenize(text) for text in self.search_texts]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_len = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

        self.term_freqs = [Counter(tokens) for tokens in self.tokenized_docs]
        self.doc_freq = Counter()

        for tokens in self.tokenized_docs:
            for term in set(tokens):
                self.doc_freq[term] += 1

        self.num_docs = len(documents)

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_index: int) -> float:
        query_terms = tokenize(query)
        tf = self.term_freqs[doc_index]
        doc_len = self.doc_lengths[doc_index]

        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue

            term_freq = tf[term]
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_len
            )
            score += self.idf(term) * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, str], float]]:
        scored = []
        for i, doc in enumerate(self.documents):
            scored.append((doc, self.score(query, i)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]