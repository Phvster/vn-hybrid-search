"""Deploy the demo: upload artifacts to a private dataset repo, then publish the Gradio Space.

Reads from .env: HF_TOKEN (write; used here only), HF_READ_TOKEN (read-only; stored as the
Space's HF_TOKEN secret) and ANTHROPIC_API_KEY (stored as a Space secret).
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from vnsearch.config import PROJECT_ROOT
from vnsearch.hub import upload_artifacts

SPACE_THREADS = "2"  # free CPU Space: 2 vCPU


def stage_space(target: Path) -> None:
    """Copy exactly the files the Space needs into `target`."""
    shutil.copytree(
        PROJECT_ROOT / "src" / "vnsearch",
        target / "src" / "vnsearch",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(PROJECT_ROOT / "configs", target / "configs")
    shutil.copy(PROJECT_ROOT / "app" / "app.py", target / "demo.py")
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(PROJECT_ROOT / "deploy" / "space" / name, target / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="vn-hybrid-search")
    parser.add_argument("--artifacts", default="vn-hybrid-search-artifacts")
    parser.add_argument("--skip-artifacts", action="store_true", help="artifacts already uploaded")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    write_token, read_token = os.environ["HF_TOKEN"], os.environ["HF_READ_TOKEN"]
    api = HfApi(token=write_token)
    user = api.whoami()["name"]
    artifacts_repo, space_repo = f"{user}/{args.artifacts}", f"{user}/{args.space}"

    if not args.skip_artifacts:
        print(f"uploading artifacts to {artifacts_repo} (private dataset)...")
        upload_artifacts(artifacts_repo, write_token)

    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_secret(space_repo, "ANTHROPIC_API_KEY", os.environ["ANTHROPIC_API_KEY"])
    api.add_space_secret(space_repo, "HF_TOKEN", read_token)
    api.add_space_variable(space_repo, "ARTIFACTS_REPO", artifacts_repo)
    api.add_space_variable(space_repo, "TORCH_THREADS", SPACE_THREADS)

    with tempfile.TemporaryDirectory() as staging:
        stage_space(Path(staging))
        api.upload_folder(
            folder_path=staging, repo_id=space_repo, repo_type="space", commit_message="Deploy demo"
        )
    print(f"Space: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
