# support-rag-agent

A domain-agnostic support RAG/Agent framework with retrieval evaluation, ranking improvements, citation-based retrieval, and failure-case analysis.

This project is designed as a production-oriented NLP system for FAQ, Help Center, and API documentation search.
The current version implements a BM25 retrieval baseline with preprocessing, title boosting, synonym normalization, FastAPI serving, and retrieval evaluation.

---

## 1. Goal

Most support chatbots are shown only as demos, but real NLP systems need measurable retrieval quality, latency, and failure analysis.

This project builds an evaluated support question-answering foundation:

```text
User Question
    ↓
Query Normalization
    ↓
BM25 Retriever
    ↓
Top-k Support Documents
    ↓
Citations
    ↓
Evaluation
```

The current version focuses on retrieval quality before adding LLM generation.

---

## 2. Current Features

* BM25 lexical retrieval
* Stopword removal
* Simple token normalization
* Title-based search
* Title boosting
* Synonym / phrase normalization
* FastAPI `/query` endpoint
* Citation-style retrieved document output
* Recall@5 and MRR evaluation
* Latency measurement
* Failure-case analysis
* Unit tests for retrieval behavior

---

## 3. Why This Project Matters

This project is not just an LLM wrapper.

It focuses on the algorithmic components behind support QA systems:

* how documents are matched to user questions
* how retrieval quality is evaluated
* how ranking failures are analyzed
* how lightweight normalization improves retrieval
* how latency changes after preprocessing
* how the system can later support dense retrieval, reranking, and tool use

---

## 4. Job Requirements Mapping

| Requirement            | Evidence in this repo                                              |
| ---------------------- | ------------------------------------------------------------------ |
| Python coding          | Modular Python implementation with FastAPI                         |
| NLP fundamentals       | Tokenization, normalization, retrieval, ranking                    |
| Semantic matching      | Query-document matching using BM25                                 |
| Q&A system             | Support document retrieval for user questions                      |
| Workflow development   | Query → retrieval → citations → evaluation                         |
| Prompt / LLM readiness | Designed to support citation-grounded generation in later versions |
| Agent readiness        | Roadmap includes tool use and multi-turn state                     |
| Result analysis        | Evaluation script and failure analysis report                      |
| Backend/API            | FastAPI endpoint for retrieval                                     |
| Testing                | Pytest tests for tokenizer and retriever behavior                  |

---

## 5. Architecture

```text
support-rag-agent/
├── app/
│   ├── bm25.py
│   ├── retriever.py
│   └── main.py
├── data/
│   ├── raw_docs_sample.jsonl
│   └── eval_questions.jsonl
├── eval/
│   └── evaluate_retrieval.py
├── reports/
│   └── results.md
├── tests/
│   └── test_bm25.py
├── requirements.txt
└── README.md
```

---

## 6. Algorithmic Design

Let:

* `D = {d_1, ..., d_N}` be support documents
* `q` be a user question
* `s(q, d_i)` be the relevance score between query `q` and document `d_i`
* `top_k(q)` be the top-k retrieved documents
* `G(q)` be the set of gold relevant document IDs

The objective is:

```text
maximize Recall@k
maximize MRR
minimize latency
```

The current baseline uses BM25 lexical retrieval.

For each query-document pair, BM25 scores documents based on term frequency, inverse document frequency, and document length normalization.

---

## 7. Retrieval Improvements

### v0: BM25 baseline

Initial implementation:

* tokenized document text only
* no stopword removal
* no title search
* no phrase normalization

### v0.1: BM25 + normalization + title search

Improvements:

* added stopword removal
* added simple plural / suffix normalization
* included document title in searchable text

### v0.2: BM25 + title boost + synonym normalization

Improvements:

* repeated document titles in searchable text
* normalized support-related phrases:

  * `taking a long time` → `delay`
  * `long time` → `delay`
  * `not arrived` → `not credited`
  * `not received` → `not credited`

---

## 8. Evaluation Results

Dataset:

* documents: 10
* evaluation questions: 5
* top_k: 5

| Version | Method                                     | Recall@5 |    MRR | Avg Latency |
| ------- | ------------------------------------------ | -------: | -----: | ----------: |
| v0      | BM25 baseline                              |   0.8000 | 0.8000 |     0.04 ms |
| v0.1    | BM25 + normalization + title search        |   1.0000 | 0.8667 |     0.09 ms |
| v0.2    | BM25 + title boost + synonym normalization |   1.0000 | 1.0000 |     0.09 ms |

### Important Limitation

The current evaluation set is intentionally small.
The high score does not prove generalization yet.

The next step is to increase the number of documents and questions, then compare BM25 with dense retrieval and reranking.

---

## 9. Example API Usage

Start the API server:

```bash
uvicorn app.main:app --reload
```

Send a query:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"My deposit has not arrived. What should I check?","top_k":5}'
```

Example response:

```json
{
  "question": "My deposit has not arrived. What should I check?",
  "retrieved_docs": [
    {
      "rank": 1,
      "doc_id": "deposit_not_credited",
      "title": "Deposit Not Credited",
      "score": 10.242,
      "snippet": "If your crypto deposit has not arrived..."
    }
  ],
  "citations": [
    "deposit_not_credited"
  ]
}
```

---

## 10. Run Evaluation

```bash
python -m eval.evaluate_retrieval
```

Example output:

```text
RESULT
Recall@5: 1.0000
MRR: 1.0000
Avg Latency: 0.09 ms
```

---

## 11. Run Tests

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

---

## 12. Data Format

### Support documents

`data/raw_docs_sample.jsonl`

```json
{
  "doc_id": "deposit_not_credited",
  "title": "Deposit Not Credited",
  "text": "If your crypto deposit has not arrived, check whether the transaction has enough blockchain confirmations..."
}
```

### Evaluation questions

`data/eval_questions.jsonl`

```json
{
  "question": "My USDT deposit has not arrived. What should I check?",
  "gold_doc_ids": ["deposit_not_credited", "network_selection"]
}
```

---

## 13. Failure Analysis

One initial failure case was:

```text
Query:
Why is my withdrawal taking a long time?

Gold document:
withdrawal_delay

v0 retrieved:
order_not_filled, api_key_permission, kyc_verification, funding_fee, deposit_not_credited
```

Cause:

* `withdrawal` and `withdrawals` were treated as different tokens
* document titles were not included in searchable text
* lexical BM25 could not connect `taking a long time` with `delay`

Fix:

* added token normalization
* included and boosted document titles
* added phrase normalization

Result:

```text
v0.2 rank:
withdrawal_delay at rank 1
```

---

## 14. Roadmap

### v1: Dense Retrieval

* add sentence-transformer embeddings
* add FAISS vector search
* compare BM25 vs dense retrieval
* evaluate Recall@3 / Recall@5 / Recall@10
* measure latency

### v2: Reranking

* add cross-encoder reranker
* compare retrieval-only vs reranked results
* measure quality-latency tradeoff

### v3: Citation-Grounded Answer Generation

* generate answers using retrieved documents
* include citations
* evaluate hallucination rate
* evaluate answer correctness

### v4: Intent Classification and Entity Extraction

* classify user support intent
* extract entities such as product, asset, network, and action
* compare rule-based baseline with ML-based classifier

### v5: Agent Workflow

* add tool use / function calling
* add multi-turn dialogue state
* add fallback behavior for unsupported questions

---

## 15. Current Status

The current version is a retrieval baseline.

It proves:

* the API works
* retrieval can be evaluated
* failure cases can be diagnosed
* simple NLP preprocessing improves ranking quality
* the system is ready for dense retrieval and reranking experiments

This project will be extended into a full support RAG/Agent system with measurable retrieval quality, grounded answer generation, and tool use.
