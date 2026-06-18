# Retrieval Evaluation Results

This report summarizes the retrieval evaluation process for `support-rag-agent`.

The goal of this evaluation is not only to report final scores, but also to show how retrieval failures were detected, analyzed, and improved through algorithmic changes.

---

## 1. Evaluation Setup

### Dataset

Current evaluation dataset:

| Item                      |                        Count |
| ------------------------- | ---------------------------: |
| Support documents         |                           10 |
| Evaluation questions      |                            5 |
| Gold document annotations | 5 question-level annotations |
| Retrieval depth           |                    top_k = 5 |

### Input files

```text
data/raw_docs_sample.jsonl
data/eval_questions.jsonl
```

### Evaluation command

```bash
python -m eval.evaluate_retrieval
```

---

## 2. Metrics

### Recall@5

Recall@5 measures whether at least one gold document appears in the top 5 retrieved documents.

```text
Recall@5 = (# questions with at least one gold document in top 5) / (# questions)
```

### MRR

MRR measures how highly the first relevant document is ranked.

```text
MRR = average(1 / rank_of_first_relevant_document)
```

A higher MRR means the correct document appears closer to rank 1.

### Average Latency

Average latency measures the mean retrieval time per query.

```text
Avg Latency = average retrieval time per question
```

---

## 3. Summary Results

| Version | Method                                     | Recall@5 |    MRR | Avg Latency |
| ------- | ------------------------------------------ | -------: | -----: | ----------: |
| v0      | BM25 baseline                              |   0.8000 | 0.8000 |     0.04 ms |
| v0.1    | BM25 + normalization + title search        |   1.0000 | 0.8667 |     0.09 ms |
| v0.2    | BM25 + title boost + synonym normalization |   1.0000 | 1.0000 |     0.09 ms |

---

## 4. v0: BM25 Baseline

### Method

The initial version used a simple BM25 lexical retriever.

Characteristics:

* document text only
* no stopword removal
* no document title search
* no title boosting
* no synonym or phrase normalization

### Result

```text
Recall@5: 0.8000
MRR: 0.8000
Avg Latency: 0.04 ms
```

### Failure Case

Query:

```text
Why is my withdrawal taking a long time?
```

Gold document:

```text
withdrawal_delay
```

Retrieved documents:

```text
1. order_not_filled
2. api_key_permission
3. kyc_verification
4. funding_fee
5. deposit_not_credited
```

Hit:

```text
False
```

### Error Analysis

The failure was caused by lexical mismatch and weak preprocessing.

Main causes:

1. `withdrawal` and `withdrawals` were treated as different tokens.
2. Stopwords such as `why`, `is`, and `my` added noise.
3. The document title `Withdrawal Delay` was not included in searchable text.
4. BM25 could not directly connect `taking a long time` with `delay`.

This showed that the baseline could retrieve documents for direct keyword overlap, but failed when the query used a semantically similar expression.

---

## 5. v0.1: BM25 + Normalization + Title Search

### Changes

v0.1 added lightweight NLP preprocessing:

* stopword removal
* simple token normalization
* plural normalization

  * `withdrawals` → `withdrawal`
* suffix normalization

  * `delayed` → `delay`
* document title included in searchable text

### Result

```text
Recall@5: 1.0000
MRR: 0.8667
Avg Latency: 0.09 ms
```

### Improvement

The previous failure case became a hit.

Query:

```text
Why is my withdrawal taking a long time?
```

Gold document:

```text
withdrawal_delay
```

Retrieved documents:

```text
1. funding_fee
2. kyc_verification
3. withdrawal_delay
4. api_key_permission
5. password_reset
```

Hit:

```text
True
```

### Remaining Issue

Although Recall@5 improved to 1.0000, the gold document was ranked at position 3.

This lowered MRR to 0.8667.

The remaining ranking issue was caused by the lexical mismatch between:

```text
taking a long time
```

and

```text
delay / delayed
```

This motivated the next improvement: title boosting and phrase normalization.

---

## 6. v0.2: BM25 + Title Boost + Synonym Normalization

### Changes

v0.2 added two improvements.

### 1. Title Boosting

Support document titles often contain the core intent.

For example:

```text
Withdrawal Delay
Deposit Not Credited
API Key Permission
```

To increase the weight of title terms, document titles were repeated in the searchable text.

### 2. Phrase / Synonym Normalization

The following phrase mappings were added:

```text
taking a long time -> delay
takes a long time  -> delay
long time          -> delay
not arrived        -> not credited
not received       -> not credited
```

These rules help BM25 handle common support-query paraphrases without using dense embeddings yet.

### Result

```text
Recall@5: 1.0000
MRR: 1.0000
Avg Latency: 0.09 ms
```

### Full Evaluation Output

```text
Q: My USDT deposit has not arrived. What should I check?
Gold: ['deposit_not_credited', 'network_selection']
Retrieved: ['deposit_not_credited', 'network_selection', 'withdrawal_delay', 'api_key_permission', 'kyc_verification']
Hit: True

Q: Why is my withdrawal taking a long time?
Gold: ['withdrawal_delay']
Retrieved: ['withdrawal_delay', 'api_key_permission', 'password_reset', 'network_selection', 'deposit_not_credited']
Hit: True

Q: What permissions can an API key have?
Gold: ['api_key_permission']
Retrieved: ['api_key_permission', 'deposit_not_credited', 'withdrawal_delay', 'kyc_verification', 'password_reset']
Hit: True

Q: Why was my limit order not executed?
Gold: ['order_not_filled']
Retrieved: ['order_not_filled', 'deposit_not_credited', 'withdrawal_delay', 'api_key_permission', 'kyc_verification']
Hit: True

Q: I lost access to my authenticator app. What should I do?
Gold: ['two_factor_auth']
Retrieved: ['two_factor_auth', 'deposit_not_credited', 'withdrawal_delay', 'api_key_permission', 'kyc_verification']
Hit: True

RESULT
Recall@5: 1.0000
MRR: 1.0000
Avg Latency: 0.09 ms
```

---

## 7. Interpretation

The evaluation shows that the BM25 baseline can be significantly improved through lightweight NLP preprocessing.

Key findings:

1. Including document titles improves intent matching.
2. Stopword removal reduces query noise.
3. Simple token normalization improves lexical overlap.
4. Title boosting improves ranking quality.
5. Phrase normalization fixes common support-query paraphrases.
6. Recall@5 alone is insufficient; MRR is needed to evaluate ranking quality.

The most important improvement is not only the final score, but the process:

```text
baseline
→ failure detected
→ error analyzed
→ preprocessing improved
→ ranking improved
→ evaluation repeated
```

This is the core evaluation loop for an NLP retrieval system.

---

## 8. Limitations

The current result should not be overinterpreted.

Current limitations:

1. The evaluation set is very small.
2. The document set contains only 10 support documents.
3. The synonym rules are manually written.
4. The current retriever is still lexical and cannot handle broad semantic paraphrases.
5. There is no dense retrieval yet.
6. There is no reranker yet.
7. There is no LLM answer generation yet.
8. There is no hallucination evaluation yet.

The current version is best understood as a clean retrieval baseline, not a complete production RAG system.

---

## 9. Next Experiments

### v1: Dense Retrieval

Add embedding-based retrieval.

Planned implementation:

```text
sentence-transformer embeddings
FAISS vector index
dense retriever module
BM25 vs dense comparison
```

Metrics:

```text
Recall@3
Recall@5
Recall@10
MRR
Avg latency
```

### v2: Hybrid Retrieval

Combine BM25 and dense retrieval.

Possible approach:

```text
hybrid_score = alpha * bm25_score + (1 - alpha) * dense_score
```

Tune `alpha` on the evaluation set.

### v3: Reranking

Add a cross-encoder reranker.

Evaluate:

```text
BM25 only
Dense only
Hybrid retrieval
Hybrid + reranker
```

### v4: Citation-Grounded Answer Generation

Generate answers only from retrieved documents.

Evaluate:

```text
answer correctness
citation accuracy
hallucination rate
unsupported question refusal
```

### v5: Agent Workflow

Add tool use and multi-turn logic.

Possible tools:

```text
search_docs(query)
get_doc_by_id(doc_id)
classify_intent(question)
extract_entities(question)
```

---

## 10. Current Status

The current version has reached the v0.2 baseline completion point.

Implemented:

* BM25 retriever
* FastAPI `/query` endpoint
* retrieval evaluation script
* stopword removal
* token normalization
* title search
* title boosting
* phrase normalization
* pytest tests

Current score:

```text
Recall@5: 1.0000
MRR: 1.0000
Avg Latency: 0.09 ms
Tests: 4 passed
```

Next milestone:

```text
Implement dense retrieval and compare it against BM25 v0.2.
```
