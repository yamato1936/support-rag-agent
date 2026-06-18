# Retrieval Evaluation Results

This report summarizes the retrieval evaluation process for support-rag-agent.

The goal is to show not only the final retrieval score, but also how retrieval failures were detected, analyzed, and improved.

---

## 1. Evaluation Setup

Dataset:

| Item | Count |
|---|---:|
| Support documents | 10 |
| Evaluation questions | 5 |
| Retrieval depth | top_k = 5 |

Input files:

- data/raw_docs_sample.jsonl
- data/eval_questions.jsonl

Evaluation commands:

    python -m eval.evaluate_retrieval
    python -m eval.compare_retrievers
    pytest

---

## 2. Metrics

Recall@5:

Recall@5 measures whether at least one gold document appears in the top 5 retrieved documents.

Formula:

    Recall@5 = number of questions with at least one gold document in top 5 / number of questions

MRR:

MRR measures how highly the first relevant document is ranked.

Formula:

    MRR = average(1 / rank of first relevant document)

Average latency:

Average latency measures the mean retrieval time per query.

---

## 3. Summary Results

| Version | Method | Recall@5 | MRR | Avg Latency |
|---|---|---:|---:|---:|
| v0 | BM25 baseline | 0.8000 | 0.8000 | 0.04 ms |
| v0.1 | BM25 + normalization + title search | 1.0000 | 0.8667 | 0.09 ms |
| v0.2 | BM25 + title boost + synonym normalization | 1.0000 | 1.0000 | 0.09 ms |
| v1 | Dense Retrieval | 1.0000 | 1.0000 | 14.39 ms |

Note:

Latency values vary slightly depending on the script and runtime environment. In the BM25-only evaluation, BM25 v0.2 averaged around 0.09 ms. In the BM25 vs Dense comparison script, BM25 v0.2 averaged around 0.15 ms.

---

## 4. v0: BM25 Baseline

Method:

The initial version used a simple BM25 lexical retriever.

Characteristics:

- document text only
- no stopword removal
- no document title search
- no title boosting
- no synonym or phrase normalization

Result:

- Recall@5: 0.8000
- MRR: 0.8000
- Avg Latency: 0.04 ms

Failure case:

- Query: Why is my withdrawal taking a long time?
- Gold document: withdrawal_delay
- Retrieved documents:
  - order_not_filled
  - api_key_permission
  - kyc_verification
  - funding_fee
  - deposit_not_credited
- Hit: False

Error analysis:

The failure was caused by lexical mismatch and weak preprocessing.

Main causes:

1. withdrawal and withdrawals were treated as different tokens.
2. Stopwords such as why, is, and my added noise.
3. The document title Withdrawal Delay was not included in searchable text.
4. BM25 could not directly connect taking a long time with delay.

This showed that the baseline could retrieve documents for direct keyword overlap, but failed when the query used a semantically similar expression.

---

## 5. v0.1: BM25 + Normalization + Title Search

Changes:

- added stopword removal
- added simple token normalization
- added plural normalization
- added suffix normalization
- included document title in searchable text

Result:

- Recall@5: 1.0000
- MRR: 0.8667
- Avg Latency: 0.09 ms

Improvement:

The previous failure case became a hit.

Failure case after v0.1:

- Query: Why is my withdrawal taking a long time?
- Gold document: withdrawal_delay
- Retrieved documents:
  - funding_fee
  - kyc_verification
  - withdrawal_delay
  - api_key_permission
  - password_reset
- Hit: True

Remaining issue:

Although Recall@5 improved to 1.0000, the gold document was ranked at position 3.

This lowered MRR to 0.8667.

The remaining ranking issue was caused by lexical mismatch between taking a long time and delay.

---

## 6. v0.2: BM25 + Title Boost + Synonym Normalization

Changes:

v0.2 added two improvements.

1. Title boosting

Support document titles often contain the core intent.

Examples:

- Withdrawal Delay
- Deposit Not Credited
- API Key Permission

To increase the weight of title terms, document titles were repeated in the searchable text.

2. Phrase and synonym normalization

The following phrase mappings were added:

- taking a long time -> delay
- takes a long time -> delay
- long time -> delay
- not arrived -> not credited
- not received -> not credited

These rules help BM25 handle common support-query paraphrases without using dense embeddings yet.

Result:

- Recall@5: 1.0000
- MRR: 1.0000
- Avg Latency: 0.09 ms

Full evaluation result:

Question 1:

- Query: My USDT deposit has not arrived. What should I check?
- Gold: deposit_not_credited, network_selection
- Retrieved:
  - deposit_not_credited
  - network_selection
  - withdrawal_delay
  - api_key_permission
  - kyc_verification
- Hit: True

Question 2:

- Query: Why is my withdrawal taking a long time?
- Gold: withdrawal_delay
- Retrieved:
  - withdrawal_delay
  - api_key_permission
  - password_reset
  - network_selection
  - deposit_not_credited
- Hit: True

Question 3:

- Query: What permissions can an API key have?
- Gold: api_key_permission
- Retrieved:
  - api_key_permission
  - deposit_not_credited
  - withdrawal_delay
  - kyc_verification
  - password_reset
- Hit: True

Question 4:

- Query: Why was my limit order not executed?
- Gold: order_not_filled
- Retrieved:
  - order_not_filled
  - deposit_not_credited
  - withdrawal_delay
  - api_key_permission
  - kyc_verification
- Hit: True

Question 5:

- Query: I lost access to my authenticator app. What should I do?
- Gold: two_factor_auth
- Retrieved:
  - two_factor_auth
  - deposit_not_credited
  - withdrawal_delay
  - api_key_permission
  - kyc_verification
- Hit: True

Final result:

- Recall@5: 1.0000
- MRR: 1.0000
- Avg Latency: 0.09 ms

---

## 7. v1: Dense Retrieval

Method:

v1 adds dense retrieval using sentence-transformers/all-MiniLM-L6-v2.

Documents and queries are encoded into dense vectors. Retrieval is performed using cosine similarity.

Implementation details:

- Model: sentence-transformers/all-MiniLM-L6-v2
- Similarity: cosine similarity
- Search method: exact NumPy matrix multiplication
- Index: no FAISS yet
- Document text: title + body

Comparison result:

| Method | Recall@5 | MRR | Avg Latency |
|---|---:|---:|---:|
| BM25 v0.2 | 1.0000 | 1.0000 | 0.15 ms |
| Dense Retrieval | 1.0000 | 1.0000 | 14.39 ms |

Interpretation:

Both BM25 v0.2 and Dense Retrieval achieved perfect Recall@5 and MRR on the current small evaluation set.

However, BM25 is significantly faster on this dataset.

Latency comparison:

- BM25 v0.2 latency: 0.15 ms
- Dense retrieval latency: 14.39 ms

This suggests that BM25 remains a strong baseline for small, keyword-heavy support document retrieval.

Dense retrieval did not improve quality on the current dataset, but it is expected to become more useful when:

- user queries contain broader paraphrases
- document wording differs from user wording
- the document set becomes larger and more diverse
- semantic similarity matters more than exact keyword overlap

Engineering note:

The first dense retrieval run downloaded the embedding model from Hugging Face. After the model is cached locally, repeated runs are faster and do not require downloading model weights again.

---

## 8. Test Results

Pytest result:

- tests collected: 5
- tests passed: 5
- result: 5 passed

Test command:

    pytest

Observed output:

    5 passed in 11.98s

Note:

Dense retriever tests load a sentence-transformer model and may take several seconds.

---

## 9. Key Findings

Main findings:

1. BM25 is a strong baseline for keyword-heavy support search.
2. Stopword removal and token normalization improved Recall@5.
3. Title search improved intent matching.
4. Title boosting improved ranking quality.
5. Phrase normalization fixed common support-query paraphrases.
6. Dense retrieval matched BM25 quality on the current dataset, but was much slower.
7. Recall@5 alone is insufficient; MRR is needed to evaluate ranking quality.
8. A small evaluation set can produce perfect scores, so the result should not be overinterpreted.

The most important process is:

    baseline
    -> failure detected
    -> error analyzed
    -> preprocessing improved
    -> ranking improved
    -> evaluation repeated
    -> dense retrieval compared

---

## 10. Limitations

Current limitations:

1. The evaluation set is very small.
2. The document set contains only 10 support documents.
3. The synonym rules are manually written.
4. The current retriever is evaluated only on simple support questions.
5. There is no paraphrase-heavy evaluation set yet.
6. There is no hybrid retrieval yet.
7. There is no reranker yet.
8. There is no LLM answer generation yet.
9. There is no hallucination evaluation yet.
10. There is no production-scale indexing yet.

The current version is best understood as a clean retrieval baseline, not a complete production RAG system.

---

## 11. Next Experiments

### v1.1: Paraphrase-heavy evaluation set

Add questions with weaker lexical overlap.

Examples:

- The money I transferred is missing from my account.
- My payout is still pending after several hours.
- Can my program trade automatically with my account?
- I cannot access my one-time password device anymore.
- My buy order stayed open and never matched.

Purpose:

- stress-test BM25
- evaluate when dense retrieval helps
- expose semantic matching failures

### v2: Hybrid Retrieval

Combine BM25 and dense retrieval.

Possible scoring formula:

    hybrid_score = alpha * bm25_score + (1 - alpha) * dense_score

Tune alpha on the evaluation set.

### v3: Reranking

Add a cross-encoder reranker.

Compare:

- BM25 only
- Dense only
- Hybrid retrieval
- Hybrid retrieval + reranker

### v4: Citation-Grounded Answer Generation

Generate answers only from retrieved documents.

Evaluate:

- answer correctness
- citation accuracy
- hallucination rate
- unsupported question refusal

### v5: Agent Workflow

Add tool use and multi-turn logic.

Possible tools:

- search_docs(query)
- get_doc_by_id(doc_id)
- classify_intent(question)
- extract_entities(question)

---

## 12. Current Status

Implemented:

- BM25 retriever
- Dense retriever
- FastAPI /query endpoint
- retrieval evaluation script
- retriever comparison script
- stopword removal
- token normalization
- title search
- title boosting
- phrase normalization
- pytest tests

Current best retrieval score:

- Recall@5: 1.0000
- MRR: 1.0000

Current tests:

- 5 passed

Next milestone:

Implement a paraphrase-heavy evaluation set and compare BM25 v0.2 against Dense Retrieval again.

---

## 13. v1.1: Paraphrase-heavy Evaluation

### Purpose

The original evaluation set was too keyword-heavy.

Both BM25 v0.2 and Dense Retrieval achieved perfect scores on the original 5-question evaluation set, so it was difficult to understand when dense retrieval becomes useful.

To stress-test semantic matching, a paraphrase-heavy evaluation set was added.

Evaluation file:

- data/eval_questions_paraphrase.jsonl

This set includes queries where the user wording differs from the document wording.

Examples:

- The money I transferred is missing from my account.
- My payout is still pending after several hours.
- Can my program place trades automatically using my account?
- I cannot access my one-time password device anymore.
- I changed my login credentials and now transfers are blocked.

### Result

| Method | Recall@5 | MRR | Avg Latency |
|---|---:|---:|---:|
| BM25 v0.2 | 0.9000 | 0.6500 | 0.15 ms |
| Dense Retrieval | 1.0000 | 0.8333 | 16.67 ms |

### Interpretation

Dense Retrieval outperformed BM25 on the paraphrase-heavy evaluation set.

BM25 remained much faster, but its ranking quality dropped significantly:

- BM25 Recall@5 dropped from 1.0000 to 0.9000
- BM25 MRR dropped from 1.0000 to 0.6500
- Dense Retrieval kept Recall@5 at 1.0000
- Dense Retrieval achieved higher MRR than BM25 on paraphrase-heavy queries

This confirms the expected tradeoff:

- BM25 is strong for keyword-heavy support search.
- Dense Retrieval is stronger when user queries use semantically similar but lexically different wording.
- Dense Retrieval is much slower without indexing or caching optimization.

### BM25 Failure Case

Query:

- I changed my login credentials and now transfers are blocked.

Gold document:

- password_reset

BM25 retrieved:

- account_security
- deposit_not_credited
- withdrawal_delay
- api_key_permission
- kyc_verification

Hit:

- False

Analysis:

BM25 failed because the query used different wording from the document.

The gold document describes password reset and temporary withdrawal restrictions, but the query used:

- changed my login credentials
- transfers are blocked

These phrases are semantically related to password reset and withdrawal restriction, but they do not overlap enough lexically.

### Dense Retrieval Behavior

Dense Retrieval retrieved the correct gold document in the top 5:

- deposit_not_credited
- password_reset
- withdrawal_delay
- network_selection
- account_security

Hit:

- True

The gold document was ranked at position 2.

This shows that dense embeddings captured semantic similarity better than BM25 for this paraphrased query.

### Key Finding

The value of dense retrieval does not appear on simple keyword-based support questions.

It appears when:

- the user describes the issue indirectly
- the query uses synonyms or paraphrases
- the document and query have weak lexical overlap
- semantic similarity matters more than exact keyword overlap

### Current Tradeoff

| Retriever | Strength | Weakness |
|---|---|---|
| BM25 v0.2 | Very fast and strong on keyword queries | Sensitive to wording mismatch |
| Dense Retrieval | Better on paraphrase-heavy queries | Much slower without indexing/caching |

### Next Step

The next milestone is Hybrid Retrieval.

Hybrid retrieval should combine the strengths of both methods:

- BM25 for fast lexical precision
- Dense Retrieval for semantic robustness

Possible scoring formula:

    hybrid_score = alpha * bm25_score + (1 - alpha) * dense_score

The next experiment should evaluate:

- BM25 only
- Dense only
- Hybrid Retrieval

on both:

- original evaluation set
- paraphrase-heavy evaluation set
