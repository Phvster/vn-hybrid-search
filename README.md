# Vietnamese Hybrid Search for RAG

**An empirical comparison of lexical, dense, and hybrid retrieval for Vietnamese legal text, packaged as the retrieval layer of a RAG system.**

> Status: 🚧 in progress. Results and demo links will be added once experiments are run.

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

*To be filled after the test run.*

| Config | Recall@5 | Recall@10 | Hit@10 | MRR@10 | nDCG@10 | p50 latency | p95 latency |
|---|---|---|---|---|---|---|---|
| R0 | – | – | – | – | – | – | – |
| R1 | – | – | – | – | – | – | – |
| R2 | – | – | – | – | – | – | – |
| R3 | – | – | – | – | – | – | – |

## Demo

*Coming soon:* Hugging Face Space and video walkthrough. The demo shows the top results of R0–R3 side by side, then generates an answer with Claude Haiku that cites specific articles.

## Design decisions worth noting

- **No data leakage in the dense model.** `bkai-foundation-models/vietnamese-bi-encoder` was excluded because it was trained on 80% of the Zalo training set, which our test split is drawn from. `bge-m3` was chosen instead (long 8k-token context, no Zalo data in its published fine-tuning sets).
- **Exact vector search.** With about 60k vectors, brute-force search takes milliseconds. Approximate indexes (HNSW, etc.) would mix index error into the model comparison.
- **Tune on dev, report on test once.** This avoids optimistic, overfitted numbers.

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
uv run python scripts/evaluate.py r1 --split dev   # run, metrics, latency -> results/
uv run python scripts/compare.py --split dev       # paired significance tests
```

*Indexing and evaluation commands will be added as they are implemented.*

## Project structure

```
configs/          experiment configs (one YAML per run)
src/vnsearch/     library code: data, text processing, retrievers, evaluation, generation
scripts/          CLI entry points
notebooks/        Kaggle encoding, error analysis
results/          EDA report and metric tables
app/              Gradio demo
tests/            unit and integration tests
docs/             architecture and experiment protocol
```

## Limitations

- Labels may be incomplete (relevant articles left unlabeled), which lowers absolute scores. Relative comparisons remain valid.
- The corpus is a 2021 snapshot and some laws have since changed. **The demo is not legal advice.**

## Data & license notes

Dataset files are not redistributed in this repository. They are downloaded by script and used for non-commercial research and education, following the original challenge terms.
