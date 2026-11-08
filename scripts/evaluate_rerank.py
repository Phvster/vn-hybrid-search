"""Evaluate R4 from reranker scores computed on Kaggle (protocol deviation 1).

Steps: check the scores match configs/r4.yaml and cover R3's candidates; re-score 2 queries on
CPU and compare (cross-check); then either tune n on dev (--tune) or build the R4 run with
the configured n, record its metrics and a CPU latency sample.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vnsearch import data, evaluation
from vnsearch.config import ARTIFACTS_DIR, RESULTS_DIR, load_config
from vnsearch.retrievers import load_retriever
from vnsearch.retrievers.rerank import apply_rerank

N_GRID = [20, 50]
DEFAULT_N = 20
TOLERANCE = 0.001
CROSS_CHECK_QUERIES = 2
MIN_SPEARMAN = 0.98


def load_scores(cfg: dict, split: str, base_run: evaluation.Run) -> dict[str, dict[str, float]]:
    directory = ARTIFACTS_DIR / "rerank"
    meta = json.loads((directory / f"meta_{split}.json").read_text(encoding="utf-8"))
    for key in ("model", "revision", "max_length"):
        if meta.get(key) != cfg[key]:
            raise ValueError(f"Scores {key}={meta.get(key)!r}, config expects {cfg[key]!r}")
    table = pd.read_parquet(directory / f"scores_{split}.parquet")
    scores: dict[str, dict[str, float]] = {}
    for qid, doc_id, score in table[["qid", "doc_id", "score"]].itertuples(index=False):
        scores.setdefault(qid, {})[doc_id] = float(score)
    for qid, hits in base_run.items():
        missing = {h.doc_id for h in hits[: cfg["scored_depth"]]} - set(scores.get(qid, {}))
        if missing:
            raise ValueError(f"Query {qid}: {len(missing)} candidates have no reranker score")
    print(f"scores: {len(table):,} pairs, GPU {meta.get('gpu')}, {meta.get('seconds', 0):.0f} s")
    return scores


def cross_check(cfg, base_run, scores, queries: pd.DataFrame) -> None:
    from vnsearch.retrievers.rerank import load_cross_encoder

    encoder = load_cross_encoder(cfg)
    contents = data.load_corpus().set_index("doc_id")["contents"]
    text = queries.set_index("qid")["text"]
    for qid in sorted(base_run)[:CROSS_CHECK_QUERIES]:
        hits = base_run[qid][: cfg["n"]]
        local = encoder.predict([(text[qid], contents[h.doc_id]) for h in hits], batch_size=20)
        remote = np.array([scores[qid][h.doc_id] for h in hits])
        rho = spearmanr(local, remote).statistic
        diff = np.abs(local - remote).max()
        print(f"cross-check {qid[:8]}: spearman {rho:.4f}, max |diff| {diff:.4f}")
        if rho < MIN_SPEARMAN:
            raise ValueError("CPU and GPU reranker scores disagree: check revision/versions")


def rerank_run(base_run, scores, n) -> evaluation.Run:
    return {qid: apply_rerank(hits, scores[qid], n) for qid, hits in base_run.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--tune", action="store_true", help="choose n on dev")
    parser.add_argument("--latency-sample", type=int, default=30)
    args = parser.parse_args()
    if args.tune and args.split != "dev":
        raise ValueError("n is tuned on dev only")

    cfg = load_config("r4")
    queries = evaluation.split_queries(args.split, allow_test=args.allow_test)
    base_run = evaluation.load_run(RESULTS_DIR / "runs" / f"{cfg['base']}_{args.split}.json")
    qrels = data.load_qrels()
    scores = load_scores(cfg, args.split, base_run)
    cross_check(cfg, base_run, scores, queries)

    if args.tune:
        table = pd.DataFrame(
            [
                {"n": n, **evaluation.evaluate(rerank_run(base_run, scores, n), qrels)}
                for n in N_GRID
            ]
        )
        table.to_csv(RESULTS_DIR / "rerank_tuning_dev.csv", index=False)
        near_best = table[table["ndcg@10"] >= table["ndcg@10"].max() - TOLERANCE]["n"].tolist()
        chosen = (
            DEFAULT_N if DEFAULT_N in near_best else int(table.loc[table["ndcg@10"].idxmax(), "n"])
        )
        print(table.round(4).to_string(index=False))
        print(f"chosen n = {chosen} (near-best: {near_best})")
        return

    run = rerank_run(base_run, scores, cfg["n"])
    evaluation.save_run(run, RESULTS_DIR / "runs" / f"r4_{args.split}.json")
    metrics = evaluation.evaluate(run, qrels)
    sample = queries.sample(args.latency_sample, random_state=0)["text"].tolist()
    latency = evaluation.benchmark_latency(load_retriever(cfg), sample, warmup=2)
    evaluation.record_metrics("r4", args.split, metrics, latency)
    print(f"r4 (n={cfg['n']}) on {args.split} ({len(queries)} queries)")
    for metric, value in metrics.items():
        print(f"  {metric:<12} {value:.4f}")
    print(f"  CPU latency ({latency['n_queries']} queries) p50 {latency['p50_ms'] / 1000:.1f} s")


if __name__ == "__main__":
    main()
