"""Hugging Face Space entry point: fetch the private artifacts, then start the demo.

Space settings: secrets ANTHROPIC_API_KEY and HF_TOKEN (read-only), variables ARTIFACTS_REPO and
TORCH_THREADS. scripts/deploy.py sets all of them.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from vnsearch.hub import download_artifacts  # noqa: E402

download_artifacts(os.environ["ARTIFACTS_REPO"], token=os.environ.get("HF_TOKEN"))

import demo  # noqa: E402  (app/app.py, copied by scripts/deploy.py)

demo.main()
