# CLAUDE.md

Vietnamese retrieval benchmark (BM25 syllable / BM25 word-segmented / dense bge-m3 / hybrid RRF) on Zalo AI 2021 Legal, with a Gradio + Claude Haiku RAG demo. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) before changing retrieval or evaluation code.

## Working agreement

- Phases: Discovery → Tech Stack → Architecture → Documentation → Environment Setup → Development → Testing & Evaluation → Deployment. Report at the end of each phase and wait for the user's confirmation before starting the next.
- Chat with the user in Vietnamese. Code, identifiers, comments, docs and commit messages in English.
- The user is learning: before any data or algorithm step, explain *why* first.
- Only build what is in the agreed scope. No over-engineering. Explain and compare options before introducing a new dependency.
- Ask instead of guessing when requirements are unclear.

## Fixed decisions (do not re-propose alternatives without a new reason)

- Stack: Python 3.11 + uv, pyvi, bm25s, bge-m3 via sentence-transformers, FAISS `IndexFlatIP`, ranx, pytest, ruff, Anthropic SDK with `claude-haiku-5-5`, Gradio on HF Spaces.
- Not used: `bkai-foundation-models/vietnamese-bi-encoder` (trained on Zalo train data, which leaks into our test set), LangChain/LlamaIndex, vector DB servers, Elasticsearch.
- Never tune on the test split. Tune on dev, run test once, and log any protocol change under *Deviations* in the experiment protocol.

## Conventions

- Package code lives in `src/vnsearch/`. Every retriever implements the `Retriever` interface in `retrievers/base.py`.
- Results are evaluated at article level. `doc_id` is the source id `"{law_id}+{article_id}"` (e.g. `01/2009/tt-bnn+1`); never reformat it.
- Data source: `GreenNode/zalo-ai-legal-text-retrieval-vn`. Ignore its built-in train/test split (it leaks 24 questions); use `data/processed/split.json`.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Critical logic (metrics, normalization, RRF, aggregation) must have unit tests. Tests never call the real Anthropic API.

## Security

- Never commit `data/`, `artifacts/`, `.env`, or any API key or HF token. Never print or log secrets.

## Commands

```bash
uv sync --all-groups      # create .venv (Python 3.11) and install everything
uv run pytest             # tests (add -m "not slow" to skip model-loading tests)
uv run ruff check .       # lint
uv run ruff format .      # format
```

On this machine `uv` is installed via pip, so call it as `py -3.13 -m uv ...` if `uv` is not on PATH (plain `py` resolves to the project venv when it is activated, and the venv has no `uv`).

Workflow for results: commit code first, then run evaluation (so metrics carry a clean commit hash), then commit `results/`.
