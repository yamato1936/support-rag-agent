# support-rag-agent

An evaluated support RAG/Agent system for customer-support question answering.

This project implements and evaluates:

* BM25 lexical retrieval
* Dense retrieval with sentence-transformers
* Hybrid retrieval with score fusion
* FastAPI `/query` and `/answer` endpoints
* deterministic citation-grounded answer generation
* unsupported-question refusal
* answer-level evaluation for citation quality and refusal behavior

The goal is not to build a simple chatbot demo.

The goal is to expose the algorithmic and evaluation components behind production-oriented support QA systems: retrieval quality, ranking failures, paraphrase robustness, latency tradeoffs, citation grounding, and unsupported-question handling.

---

## 1. Why This Project Matters

Most RAG demos only show that a chatbot can return an answer.

Real support QA systems need stronger guarantees:

* Can the retriever find the correct support document?
* Does dense retrieval help on paraphrased questions?
* Does hybrid retrieval actually improve ranking?
* Does the answer endpoint cite the right documents?
* Does the system refuse unsupported questions?
* Does citation recall hide over-citation?
* How do quality and latency change across retrieval methods?

This repository is built around those questions.

---

## 2. Current Features

### Retrieval

* BM25 retriever
* stopword removal
* token normalization
* title search
* title boosting
* synonym / phrase normalization
* dense retriever using `sentence-transformers/all-MiniLM-L6-v2`
* cosine similarity with NumPy exact search
* hybrid retriever with min-max score normalization
* alpha interpolation for BM25 / dense score fusion

### API

* `GET /health`
* `POST /query`
* `POST /answer`

The `/query` endpoint supports:

* `bm25`
* `dense`
* `hybrid`

The `/answer` endpoint supports:

* retrieval-backed answer generation
* multi-document citations
* unsupported-question refusal
* deterministic grounded answers without external LLM calls

### Evaluation

* retrieval evaluation with Recall@5, MRR, and latency
* original support-question evaluation set
* paraphrase-heavy evaluation set
* answer endpoint evaluation
* answerability accuracy
* citation recall
* citation precision
* citation F1
* unsupported refusal accuracy
* unsupported citation violation count

### Testing

* BM25 tests
* dense retriever tests
* hybrid retriever tests
* API tests
* grounded answer generator tests
* unsupported-question refusal tests

Latest local test result:

```text
17 passed, 1 warning
```

The remaining warning is a FastAPI / Starlette / httpx deprecation warning and does not affect functionality.

---

## 3. Architecture

```text
support-rag-agent/
├── app/
│   ├── __init__.py
│   ├── bm25.py
│   ├── dense_retriever.py
│   ├── generator.py
│   ├── hybrid_retriever.py
│   ├── main.py
│   ├── retriever.py
│   └── support_guard.py
├── data/
│   ├── eval_answer_questions.jsonl
│   ├── eval_questions.jsonl
│   ├── eval_questions_paraphrase.jsonl
│   └── raw_docs_sample.jsonl
├── eval/
│   ├── compare_retrievers.py
│   ├── evaluate_answers.py
│   └── evaluate_retrieval.py
├── reports/
│   └── results.md
├── tests/
│   ├── test_api.py
│   ├── test_bm25.py
│   ├── test_dense_retriever.py
│   ├── test_generator.py
│   └── test_hybrid_retriever.py
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 4. System Flow

### Retrieval flow

```text
User Question
    ↓
Query normalization
    ↓
Retriever selection
    ├── BM25
    ├── Dense
    └── Hybrid
    ↓
Top-k support documents
    ↓
Retrieved document citations
```

### Answer flow

```text
User Question
    ↓
Retrieve top-k documents
    ↓
Filter useful documents
    ↓
Check support-domain answerability
    ↓
If supported:
    generate deterministic citation-grounded answer
If unsupported:
    refuse without citations
```

---

## 5. Algorithmic Design

Let:

* `D = {d_1, ..., d_N}` be the support document set
* `q` be a user question
* `s(q, d_i)` be the relevance score between query `q` and document `d_i`
* `top_k(q)` be the top-k retrieved documents
* `G(q)` be the gold relevant document IDs

The retrieval objective is:

```text
maximize Recall@k
maximize MRR
minimize latency
```

For answer generation, the objective becomes:

```text
maximize answerability accuracy
maximize citation recall
maximize citation precision
maximize citation F1
maximize unsupported refusal accuracy
minimize unsupported citation violations
```

This separates two different problems:

1. retrieving useful documents
2. deciding whether and how to answer from those documents

---

## 6. Implemented Versions

### v0: BM25 Baseline

Initial lexical retriever:

* document text only
* no stopword removal
* no title search
* no title boosting
* no synonym normalization

### v0.1: BM25 + Normalization + Title Search

Improvements:

* stopword removal
* simple plural / suffix normalization
* document title included in searchable text

### v0.2: BM25 + Title Boost + Synonym Normalization

Improvements:

* repeated document titles in searchable text
* added phrase normalization:

```text
taking a long time -> delay
takes a long time -> delay
long time -> delay
not arrived -> not credited
not received -> not credited
```

### v1: Dense Retrieval

Added dense retrieval using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Implementation:

* encode query and documents as dense vectors
* use cosine similarity
* use NumPy exact search
* no FAISS yet

### v1.1: Paraphrase-Heavy Evaluation

Added a harder evaluation set with weaker lexical overlap.

Example query:

```text
I changed my login credentials and now transfers are blocked.
```

This query is semantically related to password reset and account security, but BM25 struggles because of lexical mismatch.

### v2: Hybrid Retrieval

Added hybrid score fusion:

```text
hybrid_score = alpha * normalized_bm25_score + (1 - alpha) * normalized_dense_score
```

Evaluated:

* `alpha=0.3`
* `alpha=0.5`
* `alpha=0.7`

### v2.1: Retriever Selection API

The `/query` endpoint now supports retriever selection:

```json
{
  "question": "I changed my login credentials and now transfers are blocked.",
  "top_k": 5,
  "retriever": "hybrid",
  "alpha": 0.3
}
```

### v3: Grounded Answer Endpoint

Added `/answer`.

The answer generator:

* uses retrieved documents only
* does not call an external LLM
* includes citations
* uses multiple documents instead of blindly trusting rank 1

### v4: Unsupported-Question Refusal

Added deterministic refusal behavior.

Unsupported examples:

```text
What is the weather tomorrow?
Who is the CEO of Apple?
What is the current Bitcoin price?
```

The system returns:

```json
{
  "is_supported": false,
  "citations": []
}
```

### v5: Answer Endpoint Evaluation

Added endpoint-level answer evaluation.

Metrics:

* answerability accuracy
* supported citation recall
* supported citation precision
* supported citation F1
* unsupported refusal accuracy
* unsupported citation violations

### v5.1: Reduce Over-Citation

The evaluation showed that citation recall alone was insufficient.

The system often included the correct gold citation, but also cited extra irrelevant documents.

To reduce over-citation, the default number of answer documents was reduced:

```text
max_answer_docs = 3
```

to:

```text
max_answer_docs = 2
```

This improved citation precision and citation F1.

---

## 7. Retrieval Evaluation Results

### Original Evaluation Set

Dataset:

* support documents: 10
* evaluation questions: 5
* retrieval depth: top_k = 5

| Method           | Recall@5 |    MRR | Avg Latency |
| ---------------- | -------: | -----: | ----------: |
| BM25 v0.2        |   1.0000 | 1.0000 |    ~0.18 ms |
| Dense Retrieval  |   1.0000 | 1.0000 |   ~17.89 ms |
| Hybrid alpha=0.5 |   1.0000 | 1.0000 |   ~17.56 ms |

Interpretation:

All methods perform perfectly on the original keyword-heavy evaluation set.

BM25 is the best practical choice for this simple setting because it is much faster.

### Paraphrase-Heavy Evaluation Set

| Method           | Recall@5 |    MRR |   Avg Latency |
| ---------------- | -------: | -----: | ------------: |
| BM25 v0.2        |   0.9000 | 0.6500 | ~0.12-0.20 ms |
| Dense Retrieval  |   1.0000 | 0.8333 |     ~16-22 ms |
| Hybrid alpha=0.3 |   1.0000 | 0.8500 |     ~18.40 ms |
| Hybrid alpha=0.5 |   1.0000 | 0.6833 |     ~17.37 ms |
| Hybrid alpha=0.7 |   1.0000 | 0.7167 |     ~15.79 ms |

Key finding:

Hybrid retrieval is not automatically better.

For this small paraphrase-heavy dataset, dense-heavy hybrid retrieval with `alpha=0.3` produced the best MRR.

BM25-heavy hybrid retrieval degraded ranking quality.

---

## 8. Answer Endpoint Evaluation Results

Answer evaluation dataset:

* supported questions: 6
* unsupported questions: 4
* total questions: 10

### v5.0: Answer Evaluation

Hybrid alpha=0.3:

| Metric                          |  Value |
| ------------------------------- | -----: |
| Answerability accuracy          | 1.0000 |
| Supported citation recall       | 1.0000 |
| Supported citation precision    | 0.4444 |
| Supported citation F1           | 0.6000 |
| Unsupported refusal accuracy    | 1.0000 |
| Unsupported citation violations |      0 |

Interpretation:

The answer endpoint correctly decides whether to answer or refuse.

However, citation precision is low because the generator over-cites.

### v5.1: Over-Citation Reduction

Hybrid alpha=0.3:

| Version | Citation Recall | Citation Precision | Citation F1 |
| ------- | --------------: | -----------------: | ----------: |
| v5.0    |          1.0000 |             0.4444 |      0.6000 |
| v5.1    |          0.9167 |             0.5833 |      0.6944 |

Stable metrics after v5.1:

| Metric                          |  Value |
| ------------------------------- | -----: |
| Answerability accuracy          | 1.0000 |
| Unsupported refusal accuracy    | 1.0000 |
| Unsupported citation violations |      0 |

Key finding:

Citation recall alone is not enough.

A RAG system can include the correct source document while also citing irrelevant documents.

Citation precision and citation F1 are necessary to detect over-citation.

---

## 9. Example API Usage

### Start the API server

```bash
uvicorn app.main:app --reload
```

### `/query`

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "I changed my login credentials and now transfers are blocked.",
    "top_k": 5,
    "retriever": "hybrid",
    "alpha": 0.3
  }'
```

Example response shape:

```json
{
  "question": "I changed my login credentials and now transfers are blocked.",
  "retriever": "hybrid",
  "retrieved_docs": [
    {
      "rank": 1,
      "doc_id": "deposit_not_credited",
      "title": "Deposit Not Credited",
      "score": 1.0,
      "snippet": "If your crypto deposit has not arrived..."
    },
    {
      "rank": 2,
      "doc_id": "password_reset",
      "title": "Password Reset",
      "score": 0.91,
      "snippet": "After resetting your password..."
    }
  ],
  "citations": [
    "deposit_not_credited",
    "password_reset"
  ]
}
```

### `/answer`

```bash
curl -X POST "http://127.0.0.1:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "I changed my login credentials and now transfers are blocked.",
    "top_k": 5,
    "retriever": "hybrid",
    "alpha": 0.3
  }'
```

Example supported response:

```json
{
  "answer": "I found the following relevant support guidance based only on the retrieved documents...",
  "citations": [
    "password_reset",
    "account_security"
  ],
  "is_supported": true,
  "reason": "Answered using retrieved support documents."
}
```

Example unsupported response:

```bash
curl -X POST "http://127.0.0.1:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who is the CEO of Apple?",
    "top_k": 5,
    "retriever": "hybrid",
    "alpha": 0.3
  }'
```

```json
{
  "answer": "Question is outside the support-document domain. I do not have enough information in the support documents to answer this question safely.",
  "citations": [],
  "is_supported": false,
  "reason": "Question is outside the support-document domain."
}
```

---

## 10. Run Evaluation

### Retrieval evaluation

```bash
python -m eval.evaluate_retrieval
```

### Retriever comparison

```bash
python -m eval.compare_retrievers
```

### Answer endpoint evaluation

```bash
python -m eval.evaluate_answers --retriever hybrid --alpha 0.3 --top-k 5
```

Compare multiple retrievers:

```bash
python -m eval.evaluate_answers --retriever bm25 --top-k 5
python -m eval.evaluate_answers --retriever dense --top-k 5
python -m eval.evaluate_answers --retriever hybrid --alpha 0.3 --top-k 5
python -m eval.evaluate_answers --retriever hybrid --alpha 0.5 --top-k 5
python -m eval.evaluate_answers --retriever hybrid --alpha 0.7 --top-k 5
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

Expected result:

```text
17 passed, 1 warning
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

### Retrieval evaluation questions

`data/eval_questions.jsonl`

```json
{
  "question": "My USDT deposit has not arrived. What should I check?",
  "gold_doc_ids": ["deposit_not_credited", "network_selection"]
}
```

### Paraphrase-heavy retrieval questions

`data/eval_questions_paraphrase.jsonl`

```json
{
  "question": "I changed my login credentials and now transfers are blocked.",
  "gold_doc_ids": ["password_reset"]
}
```

### Answer evaluation questions

`data/eval_answer_questions.jsonl`

```json
{
  "question": "Who is the CEO of Apple?",
  "expected_is_supported": false,
  "gold_doc_ids": []
}
```

---

## 13. Failure Analysis Examples

### BM25 lexical mismatch

Query:

```text
I changed my login credentials and now transfers are blocked.
```

Gold document:

```text
password_reset
```

BM25 failed to retrieve the gold document in the top 5 on the paraphrase-heavy evaluation set.

Reason:

* query uses `login credentials`
* document uses `password reset`
* query uses `transfers are blocked`
* document uses `withdrawal restrictions`

Dense retrieval and dense-heavy hybrid retrieval handled this case better.

### Over-citation in grounded answers

v5 showed that the answer endpoint often included the correct citation but also included irrelevant citations.

This is why citation recall was high but citation precision was low.

v5.1 reduced over-citation by limiting the default number of answer documents.

---

## 14. Job Requirements Mapping

| Requirement          | Evidence in this repo                                                   |
| -------------------- | ----------------------------------------------------------------------- |
| Python coding        | Modular Python implementation with FastAPI and pytest                   |
| NLP fundamentals     | Tokenization, normalization, BM25, dense retrieval, ranking             |
| Semantic matching    | Dense retrieval and paraphrase-heavy evaluation                         |
| Algorithm evaluation | Recall@5, MRR, latency, citation precision/recall/F1                    |
| RAG / Q&A system     | `/query` and `/answer` endpoints                                        |
| Grounded generation  | Deterministic citation-grounded answer generator                        |
| Safety / refusal     | Unsupported-question refusal guard                                      |
| Failure analysis     | BM25 failure cases, hybrid alpha comparison, over-citation analysis     |
| Backend/API          | FastAPI endpoints with request models                                   |
| Testing              | 17 pytest tests across retrievers, API, generator, and refusal behavior |

---

## 15. Limitations

Current limitations:

* dataset is intentionally small
* support documents are synthetic examples
* dense retrieval uses exact NumPy search instead of FAISS
* answer generation is deterministic and extractive
* unsupported-question guard is rule-based
* no external LLM generation yet
* no cross-encoder reranker yet
* no production-scale indexing yet
* no CI pipeline yet

These are deliberate constraints for the first public version.

The focus is on clear evaluation and failure analysis rather than adding uncontrolled complexity.

---

## 16. Next Steps

The first public version is complete.

Possible future improvements:

1. Add a larger and more diverse support document set.
2. Add FAISS for scalable dense retrieval.
3. Add a cross-encoder reranker.
4. Add score-threshold citation pruning.
5. Add sentence-level citation selection.
6. Add LLM-based answer generation behind the existing evaluation framework.
7. Add CI for automated tests.
8. Add Docker for reproducible local setup.
9. Add multi-turn support workflow / agent state.

---

## 17. Current Status

Implemented:

* BM25 retrieval baseline
* dense retrieval
* hybrid retrieval
* retriever selection API
* grounded answer endpoint
* multi-document citation answers
* unsupported-question refusal
* retrieval evaluation
* answer endpoint evaluation
* over-citation analysis and improvement
* pytest test suite

Latest test result:

```text
17 passed, 1 warning
```

Current best summary:

```text
The system can retrieve support documents, compare retrieval strategies, generate deterministic citation-grounded answers, refuse unsupported questions, and evaluate answer quality using citation recall, precision, and F1.
```

For detailed experiment logs, see:

```text
reports/results.md
```
