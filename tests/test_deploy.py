import fnmatch
import importlib.util

from vnsearch import hub
from vnsearch.config import PROJECT_ROOT

spec = importlib.util.spec_from_file_location("deploy", PROJECT_ROOT / "scripts" / "deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)

# Files the demo loads at start-up (R0–R2 plus the corpus).
REQUIRED = [
    "data/processed/corpus.parquet",
    "artifacts/bm25_syllable/meta.json",
    "artifacts/bm25_syllable/doc_ids.json",
    "artifacts/bm25_word/model/params.index.json",
    "artifacts/dense/embeddings.npy",
    "artifacts/dense/doc_ids.json",
    "artifacts/dense/meta.json",
]
# Labelled questions and reranker files must never be shipped.
FORBIDDEN = [
    "data/processed/queries.parquet",
    "data/processed/qrels.tsv",
    "data/processed/split.json",
    "artifacts/rerank/candidates_test.parquet",
    ".env",
]


def matches(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in hub.ARTIFACT_PATTERNS)


def test_artifact_patterns_cover_demo_and_exclude_labels():
    assert all(matches(p) for p in REQUIRED)
    assert not any(matches(p) for p in FORBIDDEN)


def test_artifact_files_lists_only_whitelisted_files(tmp_path):
    for path in REQUIRED + FORBIDDEN + ["artifacts/bm25_word/model/data.csc.index.npy"]:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("x")
    files = hub.artifact_files(tmp_path)
    assert set(REQUIRED) <= set(files)
    assert "artifacts/bm25_word/model/data.csc.index.npy" in files
    assert not set(FORBIDDEN) & set(files)


def test_staged_space_contains_entry_point_and_package(tmp_path):
    deploy.stage_space(tmp_path)
    for name in (
        "app.py",
        "demo.py",
        "requirements.txt",
        "README.md",
        "configs/r3.yaml",
        "src/vnsearch/retrievers/hybrid.py",
        "static/index.html",
        "static/app.js",
        "results/metrics_test.csv",
    ):
        assert (tmp_path / name).exists(), name
    assert not list(tmp_path.rglob("__pycache__"))
    assert not (tmp_path / ".env").exists()
