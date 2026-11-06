"""Paired significance tests between saved runs. Writes results/significance_{split}.csv.

Pairs default to the planned comparisons whose runs exist; e.g. `compare.py --pairs r1:r0`.
"""

import argparse

import pandas as pd

from vnsearch import data, evaluation
from vnsearch.config import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--pairs", nargs="*", help="system:baseline, e.g. r1:r0")
    args = parser.parse_args()

    runs_dir = RESULTS_DIR / "runs"
    pairs = (
        [tuple(p.split(":")) for p in args.pairs]
        if args.pairs
        else [
            pair
            for pair in evaluation.PLANNED_COMPARISONS
            if all((runs_dir / f"{name}_{args.split}.json").exists() for name in pair)
        ]
    )
    qrels = data.load_qrels()
    rows = []
    for system, baseline in pairs:
        runs = {
            n: evaluation.load_run(runs_dir / f"{n}_{args.split}.json") for n in (system, baseline)
        }
        rows += evaluation.paired_test((system, runs[system]), (baseline, runs[baseline]), qrels)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / f"significance_{args.split}.csv", index=False)
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
