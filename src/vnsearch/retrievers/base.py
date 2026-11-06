"""Interface shared by every retrieval configuration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float


class Retriever(ABC):
    """Returns article-level results, best first, without duplicate doc_ids."""

    name: str

    @abstractmethod
    def search(self, query: str, k: int) -> list[Hit]: ...

    def search_batch(self, queries: list[str], k: int) -> list[list[Hit]]:
        return [self.search(q, k) for q in queries]


def top_k(scores: np.ndarray, doc_ids: np.ndarray, k: int) -> list[Hit]:
    """Pick the k best scores. Ties are broken by doc_id so rankings are deterministic.

    Every candidate tied with the k-th score is kept before the exact sort, otherwise
    argpartition would cut through a tie arbitrarily.
    """
    k = min(k, len(scores))
    if k == 0:
        return []
    threshold = np.partition(scores, -k)[-k]
    candidates = np.flatnonzero(scores >= threshold)
    # lexsort sorts by the last key first: score descending, then doc_id ascending.
    order = candidates[np.lexsort((doc_ids[candidates], -scores[candidates]))][:k]
    return [Hit(str(doc_ids[i]), float(scores[i])) for i in order]
