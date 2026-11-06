# Experiment Protocol

This protocol is fixed **before** results are seen. Any change after that point is logged in the *Deviations* section with a reason.

## Research question

Which retrieval configuration (R0–R3, optionally R4) gives the best retrieval quality for Vietnamese legal questions, at what latency cost, and why?

## Dataset and split

- **Source:** Zalo AI 2021 Legal Text Retrieval, using the Hugging Face mirror [`GreenNode/zalo-ai-legal-text-retrieval-vn`](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn). Corpus unit is the article (*Điều*).
- **Verified facts** (checked 2026-11-05):
  - 61,425 articles (unique ids, format `{law_id}+{article_id}`, e.g. `01/2009/tt-bnn+1`).
  - 3,196 unique labeled questions with 3,298 relevance pairs. Every labeled article exists in the corpus.
  - Relevant articles per question: 1 → 3,103 questions (97.1%), 2 → 84, 3 → 9. Multi-relevant questions are rare, so Recall@k ≈ Hit@k for most questions.
  - `queries.jsonl` has 3,298 rows for 3,196 unique ids, so duplicates must be removed.
  - The mirror's own train/test split shares 24 questions between the two sides. **It is not used.**
  - Article length (whitespace syllables): median 183, p90 592, p95 815, max 55,368.
- Mirror rejected: `another-symato/VMTEB-Zalo-legel-retrieval` keeps only one relevant article per question (3,196 pairs), so multi-relevant labels are lost.
- **Split:** the 3,196 unique question ids are shuffled with a fixed seed (`42`) and split into **dev (30%, ~959)** and **test (70%, ~2,237)**. The split is made over **question ids**, not relevance rows, so a question can never appear on both sides. Since no model is trained, no training split is needed, and a large test set gives the significance tests more power.
- The test set is used **once**, after every setting has been frozen on dev.

## Preprocessing

- Every retriever indexes the same `contents` field: article title + body. 357 articles have an empty body and would be unfindable without the title. None of them is a labeled answer.
- 1,272 articles share their body text with another article. They are kept, because labels point to ids. Ties are broken deterministically by `doc_id`.
- Every branch applies Unicode NFC. BM25 branches also lowercase. Dense input keeps its original case.
- R0 tokenizes by whitespace (syllables). R1 segments words with pyvi.
- Chunking is decided after EDA. If used, results are aggregated to article level (max score) before evaluation.

## Configurations

| ID | Settings | Tuned on dev |
|---|---|---|
| R0 | BM25 (bm25s), syllable tokens | – (library defaults `k1=1.5`, `b=0.75`) |
| R1 | BM25 (bm25s), pyvi word tokens | – (same defaults) |
| R2 | bge-m3 dense embeddings, cosine similarity via FAISS `IndexFlatIP` (exact) | `max_seq_length` is chosen from EDA, not tuned |
| R3 | RRF of R1 and R2, top-100 from each | `k ∈ {10, 30, 60, 100}` |
| R4 *(stretch)* | bge-reranker-v2-m3 over R3's top-N | `N ∈ {20, 50}` |

If R0 beats R1 on dev, R3 fuses R0 instead, and this is reported as a deviation.

## Metrics

For a question *q*: `R_q` is the set of relevant articles, `L_q` is the ranked, deduplicated result list, and `rel_i = 1` if the item at rank *i* is in `R_q`. All metrics are computed per question, then macro-averaged.

| Metric | Definition | Interpretation |
|---|---|---|
| Recall@k | `|R_q ∩ top-k| / |R_q|` | Share of relevant articles retrieved. The most important metric for RAG |
| Hit@k | `1` if `|R_q ∩ top-k| ≥ 1`, else `0` | Was at least one relevant article retrieved? |
| MRR@10 | `1 / rank of the first relevant article` if within top-10, else `0` | How high the first relevant hit appears |
| nDCG@10 | `DCG@10 / IDCG@10`, where `DCG@10 = Σ_{i=1..10} rel_i / log2(i+1)` and `IDCG@10 = Σ_{i=1..min(|R_q|,10)} 1 / log2(i+1)` | Rewards ranking *all* relevant articles high |

Reported cut-offs: Recall@{1,5,10,20}, Hit@{1,5,10}, MRR@10, nDCG@10.

**Worked example (also used as a unit test).** `R_q = {A, B}`, ranking `[X, A, Y, Z, B, …]`:
Recall@3 = 0.5, Recall@5 = 1.0, Hit@1 = 0, Hit@3 = 1, MRR@10 = 0.5,
DCG = 1/log2(3) + 1/log2(6) = 1.018, IDCG = 1 + 1/log2(3) = 1.631, so **nDCG@10 ≈ 0.624**.

## Statistical testing

- Two-sided paired randomization test (via `ranx`) on per-question nDCG@10 and Recall@10, with α = 0.05.
- Planned comparisons: R1 vs R0, R2 vs R1, R3 vs R1, R3 vs R2 (plus R4 vs R3). A Bonferroni correction is applied across them.

## Latency

- Measured from receiving the query string to returning the top-k list (normalization, tokenization or encoding, search, fusion).
- Hardware: the same laptop CPU for all configs, fixed `torch` thread count, CPU model recorded.
- Protocol: 10 warm-up queries, then every test query run sequentially, one at a time. Report p50 and p95.
- Index build time, RAM and index size are reported separately.

## Sanity checks (before trusting any number)

- A random ranking must score ≈ 0. An oracle ranking (relevant articles first) must score 1.0 on every metric except Recall@k when `|R_q| > k`.
- Every `doc_id` in `qrels` must exist in the corpus.

## Error analysis

Per-question comparison of R1, R2 and R3 on test. Sample about 30 questions where R3 wins and about 30 where it loses, then categorize them: synonym or paraphrase, exact legal term or number (e.g. article numbers), long query, label noise, and so on.

## Reporting

`results/metrics_test.csv` (with commit hash), a summary table and findings in the README, and the error-analysis notebook.

## Threats to validity

- Incomplete labels lower absolute scores, so focus on the relative comparison.
- Only 93 questions in the whole dataset (40 in dev, 53 in test) have more than one relevant article. Multi-relevant behaviour (Recall vs Hit) is reported descriptively, not as a significance claim.
- The dense model may have seen similar public legal text during pretraining. No evidence of Zalo data in its published fine-tuning sets.
- Latency depends on hardware, so it is compared only within the same machine.

## Deviations

*None yet.*
