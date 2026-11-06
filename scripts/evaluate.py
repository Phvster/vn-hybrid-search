"""Evaluate one config on a split: save the run, compute metrics and latency, record them."""

import argparse

from vnsearch import data, evaluation
from vnsearch.config import RESULTS_DIR, load_config
from vnsearch.retrievers import load_retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--allow-test", action="store_true", help="the test split is run once")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    queries = evaluation.split_queries(args.split, allow_test=args.allow_test)
    retriever = load_retriever(cfg)

    run = evaluation.run_retriever(retriever, queries)
    evaluation.save_run(run, RESULTS_DIR / "runs" / f"{args.config}_{args.split}.json")
    scores = evaluation.evaluate(run, data.load_qrels())
    latency = (
        None
        if args.skip_latency
        else evaluation.benchmark_latency(retriever, queries["text"].tolist())
    )
    evaluation.record_metrics(args.config, args.split, scores, latency)

    print(f"{args.config} on {args.split} ({len(queries)} queries)")
    for metric, value in scores.items():
        print(f"  {metric:<12} {value:.4f}")
    if latency:
        print(f"  latency p50 {latency['p50_ms']:.1f} ms | p95 {latency['p95_ms']:.1f} ms")


if __name__ == "__main__":
    main()
