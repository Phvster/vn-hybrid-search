# Vietnamese Hybrid Search for RAG

**An empirical comparison of lexical, dense, and hybrid retrieval for Vietnamese legal text, packaged as the retrieval layer of a RAG system.**

> Status: experiments complete (test split evaluated once). Demo: runs locally; [video walkthrough](https://www.youtube.com/watch?v=gwUhqeoF-d8).

**TL;DR (held-out test, 2,237 questions):** Vietnamese word segmentation improves BM25 (+0.026 nDCG@10). Dense bge-m3 beats it (+0.056). **Hybrid RRF is best within a CPU budget: nDCG@10 0.753, Hit@10 0.920 at ~0.2 s/query.** A cross-encoder reranker pushes nDCG@10 to 0.792 (Recall@1 +8.3 points) but costs ~24 s/query on CPU. Every planned comparison is significant (p < 0.0001), except that the reranker does not change Recall@10.

## Why this project

Retrieval quality caps RAG quality: if the right document is not retrieved, the LLM cannot answer correctly. Vietnamese adds a specific challenge. It is written with spaces between **syllables**, not words, so "quyền sử dụng đất" (land-use right) is 4 space-separated tokens but only 2 words. Naive keyword search therefore matches fragments of unrelated words.

This project answers one question with measurements:

> **Which retrieval configuration works best for Vietnamese legal search, and why?**

## Configurations compared

| ID | Configuration | What it tests |
|---|---|---|
| R0 | BM25 on syllables | Baseline: raw keyword matching |
| R1 | BM25 on segmented words (pyvi) | Does Vietnamese word segmentation pay off? |
| R2 | Dense retrieval (BAAI/bge-m3 + exact FAISS search) | Does semantic matching beat keywords on terminology-heavy legal text? |
| R3 | Hybrid: Reciprocal Rank Fusion of R1 + R2 | Are lexical and semantic signals complementary? |
| R4 *(stretch)* | R3 + cross-encoder reranker (bge-reranker-v2-m3) | Is the extra latency worth the accuracy gain? |

## Dataset

[Zalo AI 2021 Legal Text Retrieval](https://challenge.zalo.ai/) (via the [GreenNode mirror](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn)): 61,425 articles (*Điều*) from Vietnamese legal documents and 3,196 human-labeled questions. Most questions have one relevant article; about 3% have two or three. The official test set has no public labels, so the labeled questions are split into a **dev** set (for tuning) and a held-out **test** set (evaluated once). See [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Metrics

Recall@{1,5,10,20}, Hit@{1,5,10}, MRR@10, nDCG@10, and query latency (p50/p95, CPU). Differences between configurations are checked with paired significance tests.

## Results

Held-out **test split, 2,237 questions, evaluated once** after every setting was frozen on dev. Latency is per query on a laptop CPU (Intel, 12 threads, torch fixed at 6), measured sequentially after warm-up.

| Config | Recall@1 | Recall@5 | Recall@10 | Hit@10 | MRR@10 | nDCG@10 | p50 latency | p95 latency |
|---|---|---|---|---|---|---|---|---|
| R0 BM25, syllables | 0.462 | 0.748 | 0.819 | 0.824 | 0.592 | 0.645 | 0.9 ms | 1.6 ms |
| R1 BM25, words | 0.492 | 0.776 | 0.845 | 0.852 | 0.618 | 0.672 | 0.8 ms | 1.2 ms |
| R2 Dense (bge-m3) | 0.558 | 0.830 | 0.884 | 0.890 | 0.680 | 0.727 | 193 ms | 223 ms |
| **R3 Hybrid RRF (k = 10)** | 0.571 | 0.862 | **0.916** | **0.920** | 0.703 | 0.753 | 191 ms | 219 ms |
| R4 R3 + reranker (N = 20) | **0.654** | **0.868** | 0.911 | 0.916 | **0.756** | **0.792** | 24.2 s* | 25.3 s* |

\* R4 latency on a fixed 30-query CPU sample. On a Kaggle T4 GPU the reranker step takes 0.43 s per query. See *Deviations* in the [protocol](docs/EXPERIMENT_PROTOCOL.md).

**Paired randomization tests** (two-sided, 10,000 permutations, Bonferroni over 5 planned comparisons, α = 0.01; [results/significance_test.csv](results/significance_test.csv)):

| Comparison | Δ nDCG@10 | Δ Recall@10 |
|---|---|---|
| R1 vs R0 (word segmentation) | +0.026 ✔ | +0.026 ✔ |
| R2 vs R1 (dense vs best lexical) | +0.056 ✔ | +0.039 ✔ |
| R3 vs R1 | +0.082 ✔ | +0.070 ✔ |
| R3 vs R2 (fusion vs dense) | +0.026 ✔ | +0.032 ✔ |
| R4 vs R3 (reranking) | +0.039 ✔ | −0.005 ✘ (p = 0.34) |

✔ significant at p < 0.0001 · ✘ not significant

Dev and test agree closely (R3 nDCG@10: 0.751 dev vs 0.753 test), so the values tuned on dev did not overfit.

### What the errors show

From the [error analysis](results/error_analysis_findings.md) (segment breakdown + 71 hand-read questions):

- **The two signals fail differently, which is why fusion works.** BM25 wins on exact rare terms, numbers and telling apart sibling articles (e.g. the *motorbike* vs *bicycle* fines articles). Dense wins on paraphrased, colloquial and short questions (≤ 12 syllables: Hit@10 0.853 vs 0.759).
- **Long articles are the weak spot of every model.** 8% of test questions point to an article longer than bge-m3's 2,048-token window. On those, dense loses its advantage over BM25, and the 512-token reranker drops *below* the hybrid (nDCG@10 0.613 vs 0.681).
- **The cost of fusion:** when BM25's rank-1 result is a lexical distractor, it can push dense's correct top-1 down to rank 2–3. The reranker repairs this ordering, which is why its gain is concentrated at Recall@1.
- **Some remaining errors are label noise:** near-identical articles in different documents, and labels that only imply the answer. Absolute scores understate quality.

### Answer generation check

On 20 random test questions ([results/llm_sanity_check.md](results/llm_sanity_check.md)), Claude Haiku 5.5 over R3's top-5 cited the labelled article in 16/20 answers (it was retrieved in 17/20). When it was not retrieved, the model said so or limited itself to related provisions; it never invented a fine or a deadline. Cost: about $0.001 per answer.

## Demo

[![Hybrid Research demo video](https://img.youtube.com/vi/gwUhqeoF-d8/hqdefault.jpg)](https://www.youtube.com/watch?v=gwUhqeoF-d8)

*Video walkthrough (Vietnamese UI, on-screen captions): hybrid retrieval with citations, configuration comparison, an out-of-scope question, and the benchmark.*

**Hybrid Research** is a web UI (FastAPI + plain HTML/CSS/JS, no build step) over the same `vnsearch` code used in the experiments:

- **Ask:** each question shows the retrieval pipeline with measured per-stage timings (BM25 → bge-m3 → RRF → Claude Haiku), the R3 top-5 sources with their **BM25 rank and dense rank** (so you can see why fusion picked them), and an answer whose citation chips jump to the quoted passage in the source article.
- **Compare:** R0–R3 top-5 side by side; hovering an article highlights it in every configuration that found it.
- **Benchmark:** the test-split metrics read from `results/metrics_test.csv`.
- Session history stays in the browser; light/dark themes; responsive down to phone width; keyboard accessible.

The demo runs locally: `uv run python app/server.py`, then open http://127.0.0.1:7860. It is not hosted permanently: Hugging Face now requires a paid plan for non-static Spaces, and bge-m3 needs about 3 GB of RAM, more than other free tiers offer. The deployment path is ready if that changes: `scripts/deploy.py` uploads the artifacts to a private dataset repo and creates the Space, which downloads them at startup with a read-only token.

## Design decisions worth noting

- **No data leakage in the dense model.** `bkai-foundation-models/vietnamese-bi-encoder` was excluded because it was trained on 80% of the Zalo training set, which our test split is drawn from. `bge-m3` was chosen instead (long 8k-token context, no Zalo data in its published fine-tuning sets).
- **Exact vector search.** With about 60k vectors, brute-force search takes milliseconds. Approximate indexes (HNSW, etc.) would mix index error into the model comparison.
- **Tune on dev, report on test once.** This avoids optimistic, overfitted numbers. Selection rules were committed before tuning, and every deviation is logged in the protocol.
- **Measured before committing to a plan.** A 5-query probe showed the reranker needs ~25 s per query on CPU, about 10× the initial estimate, so its scores were computed on a GPU and the change was logged as a protocol deviation before any result was seen.
- **Cross-checks between machines.** Vectors encoded on a Kaggle GPU match local CPU encoding (cosine ≥ 0.9997), and GPU reranker scores reproduce the CPU ranking (Spearman 1.0). Offline fusion and the online retriever produce identical metrics.

More: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.11 automatically.

```bash
git clone <this-repo> && cd vn-hybrid-search
uv sync --all-groups
cp .env.example .env      # add ANTHROPIC_API_KEY (only needed for answer generation)
uv run pytest
uv run python scripts/prepare_data.py   # download (pinned revision), clean, validate, split
uv run python scripts/eda.py            # writes results/eda_report.md
uv run python scripts/sanity_check.py   # random ranking ≈ 0, oracle ranking = 1 on dev
uv run python scripts/build_bm25.py     # R0 + R1 indexes -> artifacts/
# R2: encode the corpus with notebooks/02_encode_kaggle.ipynb on a Kaggle GPU,
#     put its output in artifacts/dense/, then:
uv run python scripts/verify_dense.py   # alignment + local-vs-Kaggle cosine check
uv run python scripts/evaluate.py r1 --split dev   # run, metrics, latency -> results/
uv run python scripts/compare.py --split dev       # paired significance tests
uv run python scripts/tune_rrf.py                  # RRF k on dev (from saved runs)
# R4: export R3 candidates, score them with notebooks/03_rerank_kaggle.ipynb on a Kaggle GPU,
#     put its output in artifacts/rerank/, then:
uv run python scripts/export_rerank_candidates.py --split dev
uv run python scripts/evaluate_rerank.py --tune    # choose n on dev; drop --tune to record
uv run python app/server.py                        # demo UI at http://127.0.0.1:7860
uv run python scripts/deploy.py                    # private artifacts repo + HF Space (Space needs a paid HF plan)
```

The test split is guarded: `evaluate.py`, `export_rerank_candidates.py` and `evaluate_rerank.py` refuse `--split test` without `--allow-test`.

## Project structure

```
configs/          experiment configs (one YAML per run)
src/vnsearch/     library code: data, text processing, retrievers, evaluation, generation
scripts/          CLI entry points
notebooks/        Kaggle encoding, error analysis
results/          EDA report and metric tables
app/              demo: FastAPI backend (server.py) + static web UI (static/)
tests/            unit and integration tests
docs/             architecture and experiment protocol
```

## Limitations

- Labels may be incomplete (relevant articles left unlabeled), which lowers absolute scores. Relative comparisons remain valid.
- The RRF constant chosen on dev (`k = 10`) lies at the edge of the pre-registered grid; the grid was not extended after seeing results.
- Answer generation sends at most 20,000 characters per article. The sanity check first ran with a 6,000-character cap; in 1 of 20 answers the relevant clause sat beyond the cut, and the model said so instead of guessing. With the cap raised, that question is answered correctly with a citation. Longer articles are still truncated.
- The corpus is a 2021 snapshot and some laws have since changed. **The demo is not legal advice.**

## Data & license notes

Dataset files are not redistributed in this repository. They are downloaded by script and used for non-commercial research and education, following the original challenge terms.
