"""R2: bge-m3 dense retrieval with exact inner-product search (FAISS IndexFlatIP).

The corpus is encoded once on a Kaggle GPU (notebooks/02_encode_kaggle.ipynb); queries are
encoded here on CPU. Both sides must use the same model revision, otherwise query and
document vectors live in different spaces and the scores are silently meaningless.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np

from vnsearch.retrievers.base import Hit, Retriever, top_k
from vnsearch.text import normalize

# Extra candidates fetched from FAISS so ties at the cutoff (identical texts give identical
# vectors) can be re-sorted by doc_id like every other retriever.
TIE_MARGIN = 32
NORM_TOLERANCE = 1e-2


def check_embeddings(embeddings: np.ndarray, doc_ids: list[str], meta: dict, cfg: dict) -> None:
    """Raise if the stored vectors do not match the config or are malformed."""
    for key in ("model", "revision", "max_seq_length"):
        if meta.get(key) != cfg[key]:
            raise ValueError(f"Embeddings {key}={meta.get(key)!r}, config expects {cfg[key]!r}")
    if len(doc_ids) != len(embeddings) or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("doc_ids must be unique and aligned 1:1 with embedding rows")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or inf")
    norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
    if np.abs(norms - 1).max() > NORM_TOLERANCE:
        raise ValueError(
            f"Embeddings are not L2-normalized (norms {norms.min():.3f}-{norms.max():.3f})"
        )


def load_encoder(cfg: dict):
    import torch
    from sentence_transformers import SentenceTransformer

    # Fixed for comparable latency; TORCH_THREADS overrides it on smaller machines (the demo Space).
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", cfg["torch_threads"])))
    encoder = SentenceTransformer(cfg["model"], revision=cfg["revision"], device="cpu")
    encoder.max_seq_length = cfg["max_seq_length"]
    return encoder


class DenseRetriever(Retriever):
    def __init__(self, name: str, encoder, embeddings: np.ndarray, doc_ids, batch_size: int = 32):
        self.name = name
        self.encoder = encoder
        self.doc_ids = np.asarray(doc_ids)
        self.batch_size = batch_size
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.encoder.encode(
            [normalize(t) for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int) -> list[list[Hit]]:
        depth = min(k + TIE_MARGIN, self.index.ntotal)
        scores, rows = self.index.search(self.encode(queries), depth)
        results = []
        for row_scores, row_ids in zip(scores, rows, strict=True):
            valid = row_ids >= 0
            results.append(top_k(row_scores[valid], self.doc_ids[row_ids[valid]], k))
        return results

    @classmethod
    def load(cls, directory: Path, cfg: dict, encoder=None) -> "DenseRetriever":
        embeddings = np.load(directory / "embeddings.npy")
        doc_ids = json.loads((directory / "doc_ids.json").read_text(encoding="utf-8"))
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        check_embeddings(embeddings, doc_ids, meta, cfg)
        encoder = encoder if encoder is not None else load_encoder(cfg)
        return cls(cfg["name"], encoder, embeddings, doc_ids, cfg["query_batch_size"])
