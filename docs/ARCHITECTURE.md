# Architecture

## Principles

1. **Offline vs. online separation.** Data preparation, index building and dense encoding run once and write files to `artifacts/`. Evaluation, latency benchmarking and the demo only read those files. Dense encoding runs on a Kaggle/Colab GPU while queries run on a laptop or a CPU Space, so files are the only shared interface.
2. **One `Retriever` interface for every configuration.** The evaluation harness, the latency benchmark and the demo never know which retriever they are calling. This guarantees all configurations are measured identically.
3. **Every run is reproducible.** Results are stored with their config, random seed, git commit hash and hardware description.

## Data flow

```
                         ┌──────────────── OFFLINE (run once) ─────────────────────┐
 HF Hub (Zalo mirror)    │                                                          │
        │ download        │   corpus.parquet ──┬─► syllable tokenize ─► bm25_syllable/ (R0)
        ▼                 │                    ├─► pyvi segment ──────► bm25_word/     (R1)
   data/raw/  ─► prepare ─┤                    └─► [Kaggle GPU] bge-m3 ─► embeddings.npy
                (NFC,     │   queries.parquet, qrels.tsv                + doc_ids.json (R2)
                 dedupe)  │   split.json (dev/test, fixed seed)         + meta.json
                          └──────────────────────────────────────────────────────────┘
                         ┌──────────────── ONLINE ─────────────────────────────────┐
  query ─► normalize ─►  │ R0/R1: BM25 │ R2: encode + FAISS │ R3: RRF(R1,R2) │ R4: rerank
                          └──────────────────────┬──────────────────────────────────┘
                                                 ▼  list[Hit(doc_id, score)]
                     ┌───────────────────────────┼──────────────────────────┐
                     ▼                           ▼                          ▼
             Evaluation harness           Latency benchmark           Gradio demo
           run files → ranx metrics      p50/p95, warm-up,        R0–R3 side by side
           + significance tests          fixed thread count       + Claude Haiku answer
```

## Components

| Module | Responsibility |
|---|---|
| `config.py` | Load experiment YAML, resolve artifact paths |
| `data.py` | Download dataset, normalize (Unicode NFC), dedupe, build `corpus` / `queries` / `qrels`, create the dev/test split |
| `text.py` | Text normalization, syllable tokenization, pyvi word segmentation |
| `retrievers/base.py` | `Retriever` interface, `Hit` type, chunk-to-article aggregation |
| `retrievers/bm25.py` | R0 and R1 (same code, different tokenizer) using `bm25s` |
| `retrievers/dense.py` | R2: bge-m3 query encoding (sentence-transformers) + FAISS `IndexFlatIP` |
| `retrievers/hybrid.py` | R3: Reciprocal Rank Fusion |
| `retrievers/rerank.py` | R4 (stretch): cross-encoder reranking of R3's top-N |
| `evaluation.py` | Produce run files, compute metrics with `ranx`, significance tests, latency benchmark |
| `generation.py` | Answer generation with Claude Haiku (`claude-haiku-5-5`) using the Citations feature |

### Retriever interface

- `search(query, k) -> list[Hit]`, where `Hit = (doc_id, score)`, sorted by score in descending order and deduplicated at article level.
- `search_batch(queries, k)` for evaluation, because batching dense query encoding is much faster.

### Key processing rules

- **Normalization.** Every branch applies Unicode NFC, because the same Vietnamese character can be encoded in several ways. Only the BM25 branches lowercase. The dense model gets cased text, since its tokenizer was trained on cased text.
- **Retrieval unit = article (Điều).** `doc_id` is the source id, kept unchanged to avoid mapping bugs: `"{law_id}+{article_id}"`, e.g. `01/2009/tt-bnn+1`. The EDA showed chunking is not needed: 98% of relevant articles fit in bge-m3's `max_seq_length = 2048`. If chunking is added later, every chunk keeps its parent `doc_id`, and retrievers aggregate chunks to articles (max score) **before** returning results.
- **Hybrid.** RRF over the top-100 of R1 and R2: `score(d) = Σ 1 / (k + rank_i(d))`. `k` is tuned on dev (see the experiment protocol).

## Artifacts

| File | Format | Content |
|---|---|---|
| `data/processed/corpus.parquet` | Parquet | `doc_id, law_id, article_id, title, text` |
| `data/processed/queries.parquet` | Parquet | `qid, text` |
| `data/processed/qrels.tsv` | TSV | `qid, doc_id, rel` |
| `data/processed/split.json` | JSON | dev/test `qid` lists + seed |
| `artifacts/bm25_{syllable,word}/` | bm25s index dir | plus tokenizer metadata |
| `artifacts/dense/embeddings.npy` | NumPy fp16, L2-normalized | row *i* ↔ `doc_ids.json[i]` (corpus order) |
| `artifacts/dense/doc_ids.json`, `meta.json` | JSON | model, revision, `max_seq_length`, library versions, GPU |
| `artifacts/dense/local_check.json` | JSON | cosine between local CPU and Kaggle vectors for a sample (must be ≥ 0.99) |

The FAISS `IndexFlatIP` is rebuilt from `embeddings.npy` at load time (under 1 s), so no index file can drift out of sync with the vectors.
| `results/runs/{config}_{split}.json` | ranx Run | `qid → {doc_id: score}` (gitignored) |
| `results/metrics_{split}.csv` | CSV | one row per config, plus commit hash (committed) |

## Demo (Hugging Face Spaces)

- Corpus, indexes and embeddings (~290 MB) live in a **private HF dataset repo** (`vnsearch.hub.ARTIFACT_PATTERNS`). The public Space downloads them at startup with a **read-only** token stored as a Space secret. The write token never leaves the developer machine. The bge-m3 model is pulled from the Hub.
- `scripts/deploy.py` uploads the artifacts, stages the Space (`deploy/space/` entry point, `src/vnsearch`, `configs/`, `app/app.py` as `demo.py`), sets secrets and variables (`TORCH_THREADS=2` for the 2-vCPU free tier), and uploads. `requirements.txt` pins the `uv.lock` versions and installs the CPU build of torch.
- UI: a query box; R0–R3 top-5 results with per-config latency; a "Generate answer" button that retrieves with R3 and calls Claude Haiku, returning an answer with `[Law, Article]` citations and a legal disclaimer.
- Cost protection: 500-character query limit, per-session and per-day generation caps, and a spend limit on the Anthropic Console.
- Zalo questions and labels are **not** deployed to the Space.

## Security

- API keys come from `.env` locally (gitignored; `.env.example` is committed) and from Space Secrets in deployment. Keys are never logged or printed.
- The generation prompt restricts answers to the provided articles. When the articles do not contain the answer, the model must say so.
- The corpus is trusted legal text. User queries cannot reach any sensitive data.

## Testing strategy

- **Unit:**
  - metrics (hand-computed example);
  - NFC normalization (e.g. "hoà" vs "hòa");
  - RRF (hand-computed example);
  - chunk-to-article aggregation;
  - tokenizers.
- **Integration:** a fixture corpus (~20 articles, ~5 queries) run end-to-end through R0, R1 and R3. R2 uses a tiny model or fake embeddings and is marked `slow`.
- **Generation:** the Anthropic client is mocked. Tests never call the real API.
- **Harness sanity checks:** a random ranking must score ≈ 0 and an oracle ranking must score 1.
