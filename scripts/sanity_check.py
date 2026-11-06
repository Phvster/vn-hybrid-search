"""Harness sanity checks on the dev split: a random ranking must score ~0, an oracle ~1."""

from vnsearch import data
from vnsearch.evaluation import METRICS, evaluate, oracle_run, random_run, split_queries


def main() -> None:
    corpus, qrels = data.load_corpus(), data.load_qrels()
    qids = split_queries("dev")["qid"].tolist()

    random_scores = evaluate(random_run(qids, corpus["doc_id"].to_numpy()), qrels)
    oracle_scores = evaluate(oracle_run(qrels, qids), qrels)

    # Oracle recall@1 is below 1 only because of multi-relevant questions.
    n_rel = qrels[qrels["qid"].isin(qids)].groupby("qid").size()
    expected_oracle_recall1 = float((1 / n_rel).mean())

    print(f"{'metric':<12} {'random':>8} {'oracle':>8}")
    for m in METRICS:
        print(f"{m:<12} {random_scores[m]:>8.4f} {oracle_scores[m]:>8.4f}")
    print(f"expected oracle recall@1 = {expected_oracle_recall1:.4f}")

    assert all(v < 0.01 for v in random_scores.values()), "random ranking scores too high"
    assert abs(oracle_scores["recall@1"] - expected_oracle_recall1) < 1e-9
    assert all(oracle_scores[m] == 1.0 for m in METRICS if m != "recall@1")
    print("sanity checks passed")


if __name__ == "__main__":
    main()
