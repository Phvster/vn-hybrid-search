"""Project paths and experiment configs."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_config(name: str) -> dict:
    cfg = yaml.safe_load((CONFIGS_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    if cfg.get("name") != name:
        raise ValueError(f"configs/{name}.yaml declares name={cfg.get('name')!r}")
    return cfg
