"""R3: Reciprocal Rank Fusion of several retrievers.

RRF(d) = sum_i 1 / (k + rank_i(d)), rank starting at 1; a document missing from list i gets
nothing from it. Only ranks are used, so BM25 scores (unbounded) and cosine scores (<= 1)
need no calibration against each other.
"""

import numpy as np

from vnsearch.retrievers.base import Hit, Retriever, top_k

FUSION_DEPTH = 100


def rrf(rankings: list[list[Hit]], k: float, top: int) -> list[Hit]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (k + rank)
    if not scores:
        return []
    return top_k(np.fromiter(scores.values(), dtype=float), np.array(list(scores)), top)


class HybridRetriever(Retriever):
    def __init__(self, name: str, retrievers: list[Retriever], k: float, depth: int = FUSION_DEPTH):
        self.name = name
        self.retrievers = retrievers
        self.k = k
        self.depth = depth

    def search(self, query: str, k: int) -> list[Hit]:
        return rrf([r.search(query, self.depth) for r in self.retrievers], self.k, k)

    def search_batch(self, queries: list[str], k: int) -> list[list[Hit]]:
        per_retriever = [r.search_batch(queries, self.depth) for r in self.retrievers]
        return [rrf(list(rankings), self.k, k) for rankings in zip(*per_retriever, strict=True)]
