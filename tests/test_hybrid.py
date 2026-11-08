import pytest

from vnsearch.retrievers.base import Hit, Retriever
from vnsearch.retrievers.hybrid import HybridRetriever, rrf


def ranked(*doc_ids):
    return [Hit(d, 100.0 - i) for i, d in enumerate(doc_ids)]


def ten_with(doc_at: dict[int, str]):
    """A 10-item ranking with given docs at given 1-based ranks, fillers elsewhere."""
    return ranked(*[doc_at.get(r, f"fill{r}_{id(doc_at)}") for r in range(1, 11)])


def test_hand_computed_example_k60_favors_consistent_doc():
    # A: rank 1 in list 1, rank 10 in list 2. B: rank 3 in both.
    lists = [ten_with({1: "A", 3: "B"}), ten_with({3: "B", 10: "A"})]
    fused = {h.doc_id: h.score for h in rrf(lists, k=60, top=30)}
    assert fused["A"] == pytest.approx(1 / 61 + 1 / 70)  # 0.03068
    assert fused["B"] == pytest.approx(2 / 63)  # 0.03175
    assert fused["B"] > fused["A"]


def test_small_k_favors_a_single_top_rank():
    lists = [ten_with({1: "A", 3: "B"}), ten_with({3: "B", 10: "A"})]
    fused = {h.doc_id: h.score for h in rrf(lists, k=1, top=30)}
    assert fused["A"] == pytest.approx(1 / 2 + 1 / 11)
    assert fused["A"] > fused["B"]


def test_doc_in_one_list_only_gets_one_contribution():
    fused = {h.doc_id: h.score for h in rrf([ranked("X"), ranked("Y")], k=60, top=5)}
    assert fused == {"X": pytest.approx(1 / 61), "Y": pytest.approx(1 / 61)}


def test_ties_broken_by_doc_id_and_scores_ignored():
    # Raw scores are wildly different scales; only ranks matter, and X/Y tie.
    bm25 = [Hit("Y", 35.0)]
    dense = [Hit("X", 0.7)]
    assert [h.doc_id for h in rrf([bm25, dense], k=60, top=2)] == ["X", "Y"]


def test_empty_inputs():
    assert rrf([[], []], k=60, top=10) == []


class Fixed(Retriever):
    def __init__(self, name, docs):
        self.name, self.docs, self.requested = name, docs, []

    def search(self, query, k):
        self.requested.append(k)
        return ranked(*self.docs)[:k]


def test_hybrid_fetches_fusion_depth_and_batch_matches_single():
    a, b = Fixed("a", ["d1", "d2", "d3"]), Fixed("b", ["d3", "d4"])
    hybrid = HybridRetriever("r3", [a, b], k=60, depth=50)
    single = hybrid.search("q", k=2)
    assert a.requested == [50] and b.requested == [50]
    assert [h.doc_id for h in single] == ["d3", "d1"]  # d3 is in both lists
    assert hybrid.search_batch(["q", "q"], k=2) == [single, single]
