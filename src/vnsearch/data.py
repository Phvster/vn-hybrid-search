"""Download, clean, validate and split the Zalo AI 2021 Legal Text Retrieval data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from vnsearch.config import PROCESSED_DIR, RAW_DIR
from vnsearch.text import normalize

DATASET_REPO = "GreenNode/zalo-ai-legal-text-retrieval-vn"
# Pinned so that re-downloading always yields exactly the data the results were computed on.
DATASET_REVISION = "12d76d4d04b94ceada970fcfbe7fec20bfa97389"
RAW_FILES = {
    "corpus": "corpus/test-00000-of-00001.parquet",
    "queries": "queries.jsonl",
    "qrels_train": "qrels/train.jsonl",
    "qrels_test": "qrels/test.jsonl",
}

DEV_FRACTION = 0.3
SPLIT_SEED = 42


def download_raw(raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Download the raw files at the pinned revision. Returns local paths keyed by name."""
    return {
        name: Path(
            hf_hub_download(
                DATASET_REPO,
                filename,
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=raw_dir,
            )
        )
        for name, filename in RAW_FILES.items()
    }


def read_jsonl(path: Path, id_columns: list[str]) -> pd.DataFrame:
    # Ids must stay strings: hex ids such as "123e4567" would otherwise be parsed as floats.
    return pd.read_json(path, lines=True, dtype={col: str for col in id_columns})


def article_contents(title: str, text: str) -> str:
    """Text that gets indexed: title plus body (some articles have an empty body)."""
    return f"{title}\n{text}" if text else title


def prepare_corpus(raw: pd.DataFrame) -> pd.DataFrame:
    corpus = pd.DataFrame({"doc_id": raw["id"].astype(str)})
    corpus[["law_id", "article_id"]] = corpus["doc_id"].str.rsplit("+", n=1, expand=True)
    corpus["title"] = raw["title"].fillna("").map(normalize)
    corpus["text"] = raw["text"].fillna("").map(normalize)
    corpus["contents"] = [
        article_contents(t, x) for t, x in zip(corpus["title"], corpus["text"], strict=True)
    ]
    return corpus


def prepare_queries(raw: pd.DataFrame) -> pd.DataFrame:
    queries = raw.rename(columns={"_id": "qid"})[["qid", "text"]]
    conflicting = queries.groupby("qid")["text"].nunique()
    if (conflicting > 1).any():
        bad = conflicting[conflicting > 1].index[:5].tolist()
        raise ValueError(f"Query ids with different texts: {bad}")
    queries = queries.drop_duplicates("qid").reset_index(drop=True)
    queries["text"] = queries["text"].map(normalize)
    return queries


def prepare_qrels(*raw_frames: pd.DataFrame) -> pd.DataFrame:
    qrels = pd.concat(raw_frames).rename(columns={"query-id": "qid", "corpus-id": "doc_id"})
    qrels = qrels[["qid", "doc_id"]].drop_duplicates().reset_index(drop=True)
    qrels["rel"] = 1
    return qrels


def validate(corpus: pd.DataFrame, queries: pd.DataFrame, qrels: pd.DataFrame) -> None:
    """Raise if the prepared tables are inconsistent."""
    problems = []
    if corpus["doc_id"].duplicated().any():
        problems.append("duplicate doc_id in corpus")
    if (corpus["contents"] == "").any():
        problems.append("articles with empty contents")
    if missing := set(qrels["doc_id"]) - set(corpus["doc_id"]):
        problems.append(f"{len(missing)} qrels doc_ids not in corpus")
    if missing := set(qrels["qid"]) - set(queries["qid"]):
        problems.append(f"{len(missing)} qrels qids not in queries")
    if unlabeled := set(queries["qid"]) - set(qrels["qid"]):
        problems.append(f"{len(unlabeled)} queries without relevance labels")
    if problems:
        raise ValueError("; ".join(problems))


def make_split(qids, dev_fraction: float = DEV_FRACTION, seed: int = SPLIT_SEED) -> dict:
    """Split question ids into disjoint dev/test sets, deterministically.

    Ids are sorted before shuffling so the result does not depend on input row order.
    """
    ids = np.array(sorted(set(qids)))
    order = np.random.default_rng(seed).permutation(len(ids))
    n_dev = round(len(ids) * dev_fraction)
    return {
        "seed": seed,
        "dev_fraction": dev_fraction,
        "dataset_revision": DATASET_REVISION,
        "dev": sorted(ids[order[:n_dev]].tolist()),
        "test": sorted(ids[order[n_dev:]].tolist()),
    }


def save_processed(
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    split: dict,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out_dir / "corpus.parquet", index=False)
    queries.to_parquet(out_dir / "queries.parquet", index=False)
    qrels.to_csv(out_dir / "qrels.tsv", sep="\t", index=False)
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")


def load_corpus(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return pd.read_parquet(processed_dir / "corpus.parquet")


def load_queries(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return pd.read_parquet(processed_dir / "queries.parquet")


def load_qrels(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return pd.read_csv(processed_dir / "qrels.tsv", sep="\t", dtype={"qid": str, "doc_id": str})


def load_split(processed_dir: Path = PROCESSED_DIR) -> dict:
    return json.loads((processed_dir / "split.json").read_text(encoding="utf-8"))
