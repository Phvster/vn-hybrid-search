"""Verify embeddings downloaded from Kaggle before using them (run after placing them in
artifacts/dense/). Checks alignment with the corpus, then re-encodes a sample of articles
locally on CPU: the cosine with the Kaggle vectors must be ~1, proving both sides share
the same model and preprocessing.
"""

import json

import numpy as np

from vnsearch import data
from vnsearch.config import ARTIFACTS_DIR, load_config
from vnsearch.retrievers.dense import DenseRetriever, check_embeddings, load_encoder

SAMPLE_SIZE = 16
MIN_COSINE = 0.99


def main() -> None:
    cfg = load_config("r2")
    directory = ARTIFACTS_DIR / cfg["index"]
    embeddings = np.load(directory / "embeddings.npy")
    doc_ids = json.loads((directory / "doc_ids.json").read_text(encoding="utf-8"))
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    check_embeddings(embeddings, doc_ids, meta, cfg)

    corpus = data.load_corpus()
    if doc_ids != corpus["doc_id"].tolist():
        raise ValueError("Embedding rows are not in corpus order")
    print(f"{len(doc_ids):,} vectors, dim {embeddings.shape[1]}, dtype {embeddings.dtype}: aligned")

    sample = np.random.default_rng(0).choice(len(corpus), size=SAMPLE_SIZE, replace=False)
    retriever = DenseRetriever(cfg["name"], load_encoder(cfg), embeddings, doc_ids)
    local = retriever.encode(corpus["contents"].iloc[sample].tolist())
    cosines = (local * embeddings[sample].astype(np.float32)).sum(axis=1)
    print(f"local vs Kaggle cosine: min {cosines.min():.5f}, mean {cosines.mean():.5f}")
    if cosines.min() < MIN_COSINE:
        raise ValueError("Local and Kaggle encoders disagree: check model revision and versions")

    check = {
        "sample_size": SAMPLE_SIZE,
        "min_cosine": float(cosines.min()),
        "mean_cosine": float(cosines.mean()),
    }
    (directory / "local_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print("dense artifacts verified")


if __name__ == "__main__":
    main()
