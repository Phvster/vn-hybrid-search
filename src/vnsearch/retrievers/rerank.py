"""R4: cross-encoder re-ranking of R3's top-N.

A cross-encoder reads query and article together, so it is more precise than comparing two
separately computed vectors, but it must run once per (query, article) pair. It can only
reorder R3's candidates, never find new ones: ranks N+1..depth keep R3's order.

`apply_rerank` is shared by the online retriever (scores computed here on CPU) and by the
offline evaluation (scores computed on a Kaggle GPU), so both paths rank identically.
"""

import numpy as np
import pandas as pd

from vnsearch.retrievers.base import Hit, Retriever, top_k

RUN_DEPTH = 100


def apply_rerank(base: list[Hit], scores: dict[str, float], n: int) -> list[Hit]:
    """Reorder the first n hits by reranker score (ties by doc_id); keep the rest as is."""
    head, tail = base[:n], base[n:]
    if not head:
        return []
    head_scores = np.array([scores[h.doc_id] for h in head], dtype=float)
    return top_k(head_scores, np.array([h.doc_id for h in head]), len(head)) + tail


def load_cross_encoder(cfg: dict):
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(cfg["torch_threads"])
    return CrossEncoder(
        cfg["model"], revision=cfg["revision"], max_length=cfg["max_length"], device="cpu"
    )


class RerankRetriever(Retriever):
    def __init__(
        self,
        name: str,
        base: Retriever,
        cross_encoder,
        contents: pd.Series,
        n: int,
        batch_size: int = 20,
        depth: int = RUN_DEPTH,
    ):
        self.name = name
        self.base = base
        self.cross_encoder = cross_encoder
        self.contents = contents  # doc_id -> indexed text
        self.n = n
        self.batch_size = batch_size
        self.depth = depth

    def score(self, query: str, hits: list[Hit]) -> dict[str, float]:
        pairs = [(query, self.contents[h.doc_id]) for h in hits]
        if not pairs:
            return {}
        scores = self.cross_encoder.predict(pairs, batch_size=self.batch_size)
        return {h.doc_id: float(s) for h, s in zip(hits, scores, strict=True)}

    def search(self, query: str, k: int) -> list[Hit]:
        base = self.base.search(query, max(k, self.depth))
        return apply_rerank(base, self.score(query, base[: self.n]), self.n)[:k]
