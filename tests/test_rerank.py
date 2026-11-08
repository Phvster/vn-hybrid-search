import pandas as pd
import pytest

from vnsearch.retrievers.base import Hit, Retriever
from vnsearch.retrievers.rerank import RerankRetriever, apply_rerank


def ranked(*doc_ids):
    return [Hit(d, 1.0 / (i + 1)) for i, d in enumerate(doc_ids)]


def test_head_is_reordered_and_tail_keeps_base_order():
    base = ranked("a", "b", "c", "d", "e")
    out = apply_rerank(base, {"a": 0.1, "b": 0.9, "c": 0.5}, n=3)
    assert [h.doc_id for h in out] == ["b", "c", "a", "d", "e"]


def test_reranker_ties_broken_by_doc_id():
    out = apply_rerank(ranked("z", "y", "x"), {"z": 1.0, "y": 1.0, "x": 0.0}, n=3)
    assert [h.doc_id for h in out] == ["y", "z", "x"]


def test_missing_score_is_an_error_not_a_silent_skip():
    with pytest.raises(KeyError):
        apply_rerank(ranked("a", "b"), {"a": 1.0}, n=2)


def test_recall_at_n_cannot_change():
    base = ranked(*[f"d{i}" for i in range(30)])
    out = apply_rerank(base, {f"d{i}": float(i) for i in range(20)}, n=20)
    assert {h.doc_id for h in out[:20]} == {h.doc_id for h in base[:20]}
    assert out[20:] == base[20:]


class FixedBase(Retriever):
    name = "r3"

    def search(self, query, k):
        return ranked("a", "b", "c", "d")[:k]


class FakeCrossEncoder:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs, batch_size=32):
        self.pairs.extend(pairs)
        return [len(text) for _, text in pairs]  # longer text = higher score


def test_online_retriever_scores_only_top_n_with_article_text():
    contents = pd.Series({"a": "x", "b": "xxx", "c": "xx", "d": "xxxx"})
    encoder = FakeCrossEncoder()
    retriever = RerankRetriever("r4", FixedBase(), encoder, contents, n=3)
    hits = retriever.search("câu hỏi", k=4)
    assert [h.doc_id for h in hits] == ["b", "c", "a", "d"]
    assert [text for _, text in encoder.pairs] == ["x", "xxx", "xx"]
    assert all(query == "câu hỏi" for query, _ in encoder.pairs)
