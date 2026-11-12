"""Move the demo's artifacts to and from a private Hugging Face dataset repo.

Only what the demo needs is shipped: the corpus and the R0–R2 indexes. The Zalo questions and
labels stay local. Paths in the repo mirror the project layout, so a download into the project
root puts every file where `vnsearch.config` expects it.
"""

import fnmatch
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from vnsearch.config import PROJECT_ROOT

UPLOAD_ATTEMPTS = 5

ARTIFACT_PATTERNS = [
    "data/processed/corpus.parquet",
    "artifacts/bm25_syllable/**",
    "artifacts/bm25_word/**",
    "artifacts/dense/embeddings.npy",
    "artifacts/dense/doc_ids.json",
    "artifacts/dense/meta.json",
]


def artifact_files(root: Path = PROJECT_ROOT) -> list[str]:
    """Repo-relative POSIX paths of every local file matching ARTIFACT_PATTERNS."""
    files = []
    for top in ("data/processed", "artifacts"):
        for p in (root / top).rglob("*"):
            rel = p.relative_to(root).as_posix()
            if p.is_file() and any(fnmatch.fnmatch(rel, pat) for pat in ARTIFACT_PATTERNS):
                files.append(rel)
    return sorted(files)


def upload_artifacts(repo_id: str, token: str) -> None:
    """Upload one file per commit, skipping files already on the Hub with the same size.

    One commit per file makes the upload resumable: on a slow or flaky connection a failure
    loses only the file in flight, and a rerun continues where the last one stopped.
    """
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    remote_sizes = {s.rfilename: s.size for s in info.siblings}
    for path in artifact_files():
        local = PROJECT_ROOT / path
        size = local.stat().st_size
        if remote_sizes.get(path) == size:
            print(f"skip {path} (already uploaded)", flush=True)
            continue
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            try:
                print(f"upload {path} ({size / 1e6:.1f} MB), attempt {attempt}", flush=True)
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload {path}",
                )
                break
            except Exception as e:  # network errors surface as several exception types
                if attempt == UPLOAD_ATTEMPTS:
                    raise
                print(f"  failed ({type(e).__name__}), retrying in 30 s", flush=True)
                time.sleep(30)


def download_artifacts(repo_id: str, token: str | None) -> None:
    snapshot_download(
        repo_id,
        repo_type="dataset",
        token=token,
        local_dir=PROJECT_ROOT,
        allow_patterns=ARTIFACT_PATTERNS,
    )
