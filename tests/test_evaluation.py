import numpy as np
import pandas as pd
import pytest

from vnsearch import evaluation
from vnsearch.retrievers.base import Hit, Retriever, top_k


def qrels_for(relevant: dict[str, list[str]]) -> pd.DataFrame:
    rows = [(q, d, 1) for q, docs in relevant.items() for d in docs]
    return pd.DataFrame(rows, columns=["qid", "doc_id", "rel"])


def ranking(*doc_ids: str) -> list[Hit]:
    return [Hit(d, 1.0) for d in doc_ids]  # identical scores: order must still be respected


def test_protocol_worked_example():
    # docs/EXPERIMENT_PROTOCOL.md: R_q = {A, B}, ranking [X, A, Y, Z, B].
    run = {"q": ranking("X", "A", "Y", "Z", "B")}
    metrics = ["recall@3", "recall@5", "hit_rate@1", "hit_rate@3", "mrr@10", "ndcg@10"]
    scores = evaluation.evaluate(run, qrels_for({"q": ["A", "B"]}), metrics)
    assert scores["recall@3"] == pytest.approx(0.5)
    assert scores["recall@5"] == pytest.approx(1.0)
    assert scores["hit_rate@1"] == 0.0
    assert scores["hit_rate@3"] == 1.0
    assert scores["mrr@10"] == pytest.approx(0.5)
    assert scores["ndcg@10"] == pytest.approx(1.018 / 1.631, abs=1e-3)


def test_three_relevant_one_found_at_rank_one():
    run = {"q": ranking("A", "X", "Y")}
    scores = evaluation.evaluate(run, qrels_for({"q": ["A", "B", "C"]}))
    assert scores["hit_rate@10"] == 1.0
    assert scores["recall@10"] == pytest.approx(1 / 3)
    assert scores["mrr@10"] == 1.0
    assert scores["ndcg@10"] == pytest.approx(1 / (1 + 1 / np.log2(3) + 0.5), abs=1e-3)


def test_metrics_are_macro_averaged():
    run = {"q1": ranking("A"), "q2": ranking("X")}
    scores = evaluation.evaluate(run, qrels_for({"q1": ["A"], "q2": ["B"]}), ["hit_rate@1"])
    assert scores["hit_rate@1"] == pytest.approx(0.5)


def test_query_with_no_results_scores_zero():
    run = {"q1": ranking("A"), "q2": []}
    scores = evaluation.evaluate(run, qrels_for({"q1": ["A"], "q2": ["B"]}), ["recall@10"])
    assert scores["recall@10"] == pytest.approx(0.5)


def test_tied_hits_keep_retriever_order_in_ranx():
    # Same score for both; the retriever put B first, so MRR must be 1/2.
    run = {"q": ranking("B", "A")}
    assert evaluation.evaluate(run, qrels_for({"q": ["A"]}), ["mrr@10"])["mrr@10"] == 0.5


def test_top_k_breaks_ties_by_doc_id():
    scores = np.array([0.5, 0.9, 0.5, 0.1, 0.5])
    doc_ids = np.array(["d", "a", "c", "e", "b"])
    assert [h.doc_id for h in top_k(scores, doc_ids, 3)] == ["a", "b", "c"]


def test_top_k_keeps_ties_at_the_cutoff():
    scores = np.zeros(1000)
    doc_ids = np.array([f"d{i:04d}" for i in range(1000)][::-1])
    # All tied: the result must be the k smallest ids, not an arbitrary subset.
    assert [h.doc_id for h in top_k(scores, doc_ids, 3)] == ["d0000", "d0001", "d0002"]


def test_top_k_handles_k_larger_than_corpus():
    assert len(top_k(np.array([1.0, 2.0]), np.array(["a", "b"]), 10)) == 2


def test_oracle_and_random_runs():
    qrels = qrels_for({"q1": ["A"], "q2": ["B", "C"]})
    oracle = evaluation.evaluate(evaluation.oracle_run(qrels, ["q1", "q2"]), qrels)
    assert oracle["ndcg@10"] == 1.0
    assert oracle["recall@1"] == pytest.approx((1 + 0.5) / 2)
    doc_ids = np.array([f"d{i}" for i in range(10_000)] + ["A", "B", "C"])
    random = evaluation.evaluate(evaluation.random_run(["q1", "q2"], doc_ids, k=10), qrels)
    assert random["recall@10"] < 0.05


def test_run_round_trip(tmp_path):
    run = {"q": [Hit("01/2009/tt-bnn+1", 2.5), Hit("x+2", 1.0)]}
    path = tmp_path / "runs" / "r.json"
    evaluation.save_run(run, path)
    assert evaluation.load_run(path) == run


def test_test_split_requires_explicit_permission():
    with pytest.raises(PermissionError):
        evaluation.split_queries("test")


class CountingRetriever(Retriever):
    name = "counting"

    def __init__(self):
        self.calls = 0

    def search(self, query, k):
        self.calls += 1
        return [Hit("a", 1.0)]


def test_latency_excludes_warmup_from_timings():
    retriever = CountingRetriever()
    result = evaluation.benchmark_latency(retriever, ["q"] * 20, warmup=5)
    assert result["n_queries"] == 20
    assert retriever.calls == 25
    assert result["p95_ms"] >= result["p50_ms"] >= 0


def test_run_retriever_covers_every_query():
    queries = pd.DataFrame({"qid": [f"q{i}" for i in range(130)], "text": ["x"] * 130})
    run = evaluation.run_retriever(CountingRetriever(), queries, k=5, batch_size=64)
    assert list(run) == queries["qid"].tolist()
