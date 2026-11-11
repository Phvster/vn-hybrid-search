"""Move the demo's artifacts to and from a private Hugging Face dataset repo.

Only what the demo needs is shipped: the corpus and the R0–R2 indexes. The Zalo questions and
labels stay local. Paths in the repo mirror the project layout, so a download into the project
root puts every file where `vnsearch.config` expects it.
"""

from huggingface_hub import HfApi, snapshot_download

from vnsearch.config import PROJECT_ROOT

ARTIFACT_PATTERNS = [
    "data/processed/corpus.parquet",
    "artifacts/bm25_syllable/**",
    "artifacts/bm25_word/**",
    "artifacts/dense/embeddings.npy",
    "artifacts/dense/doc_ids.json",
    "artifacts/dense/meta.json",
]


def upload_artifacts(repo_id: str, token: str) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=PROJECT_ROOT,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=ARTIFACT_PATTERNS,
        commit_message="Upload demo artifacts",
    )


def download_artifacts(repo_id: str, token: str | None) -> None:
    snapshot_download(
        repo_id,
        repo_type="dataset",
        token=token,
        local_dir=PROJECT_ROOT,
        allow_patterns=ARTIFACT_PATTERNS,
    )
