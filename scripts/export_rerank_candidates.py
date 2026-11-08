"""Export R3's top candidates for one split so the reranker can score them on Kaggle.

Writes artifacts/rerank/candidates_{split}.parquet with (qid, query, doc_id, rank). The corpus
text is not included: the Kaggle notebook joins it from corpus.parquet.
"""

import argparse

import pandas as pd

from vnsearch import evaluation
from vnsearch.config import ARTIFACTS_DIR, RESULTS_DIR, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config("r4")
    queries = evaluation.split_queries(args.split, allow_test=args.allow_test)
    run = evaluation.load_run(RESULTS_DIR / "runs" / f"{cfg['base']}_{args.split}.json")
    text = queries.set_index("qid")["text"]
    rows = [
        (qid, text[qid], hit.doc_id, rank)
        for qid in queries["qid"]
        for rank, hit in enumerate(run[qid][: cfg["scored_depth"]], start=1)
    ]
    table = pd.DataFrame(rows, columns=["qid", "query", "doc_id", "rank"])
    out = ARTIFACTS_DIR / "rerank" / f"candidates_{args.split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"{len(table):,} pairs for {table['qid'].nunique()} queries -> {out}")


if __name__ == "__main__":
    main()
