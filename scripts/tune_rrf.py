"""Tune the RRF constant k on dev by fusing the saved R1 and R2 dev runs (no re-retrieval).

Selection rule (docs/EXPERIMENT_PROTOCOL.md): highest dev nDCG@10; within 0.001 of the
best, prefer the default k = 60. Writes results/rrf_tuning_dev.csv.
"""

import pandas as pd

from vnsearch import data, evaluation
from vnsearch.config import RESULTS_DIR, load_config
from vnsearch.retrievers.hybrid import rrf

K_GRID = [10, 30, 60, 100]
DEFAULT_K = 60
TOLERANCE = 0.001


def main() -> None:
    cfg = load_config("r3")
    runs = [evaluation.load_run(RESULTS_DIR / "runs" / f"{c}_dev.json") for c in cfg["components"]]
    qrels = data.load_qrels()

    rows = []
    for k in K_GRID:
        fused = {
            qid: rrf([run[qid][: cfg["depth"]] for run in runs], k, evaluation.RUN_DEPTH)
            for qid in runs[0]
        }
        rows.append({"rrf_k": k, **evaluation.evaluate(fused, qrels)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "rrf_tuning_dev.csv", index=False)

    best = table["ndcg@10"].max()
    near_best = table[table["ndcg@10"] >= best - TOLERANCE]["rrf_k"].tolist()
    chosen = (
        DEFAULT_K if DEFAULT_K in near_best else int(table.loc[table["ndcg@10"].idxmax(), "rrf_k"])
    )
    print(table.round(4).to_string(index=False))
    print(f"chosen rrf_k = {chosen} (near-best: {near_best})")


if __name__ == "__main__":
    main()
