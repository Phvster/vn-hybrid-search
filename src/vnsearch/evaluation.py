"""Run retrievers, score them with ranx, benchmark latency and record results."""

import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import ranx

from vnsearch import data
from vnsearch.config import PROJECT_ROOT, RESULTS_DIR
from vnsearch.retrievers.base import Hit, Retriever

METRICS = [
    "recall@1",
    "recall@5",
    "recall@10",
    "recall@20",
    "hit_rate@1",
    "hit_rate@5",
    "hit_rate@10",
    "mrr@10",
    "ndcg@10",
]
RUN_DEPTH = 100  # results kept per query; also the fusion depth for R3

# docs/EXPERIMENT_PROTOCOL.md: planned comparisons (system, baseline), Bonferroni-corrected.
PLANNED_COMPARISONS = [("r1", "r0"), ("r2", "r1"), ("r3", "r1"), ("r3", "r2")]
ALPHA = 0.05
SIGNIFICANCE_METRICS = ["ndcg@10", "recall@10"]
N_PERMUTATIONS = 10_000

Run = dict[str, list[Hit]]


def split_queries(split: str, allow_test: bool = False) -> pd.DataFrame:
    """Queries of one split. The test split must be requested explicitly (protocol: run once)."""
    if split == "test" and not allow_test:
        raise PermissionError("The test split is evaluated once; pass allow_test=True on purpose.")
    qids = set(data.load_split()[split])
    queries = data.load_queries()
    return queries[queries["qid"].isin(qids)].reset_index(drop=True)


def run_retriever(
    retriever: Retriever, queries: pd.DataFrame, k: int = RUN_DEPTH, batch_size: int = 64
) -> Run:
    run: Run = {}
    for start in range(0, len(queries), batch_size):
        batch = queries.iloc[start : start + batch_size]
        results = retriever.search_batch(batch["text"].tolist(), k)
        run.update(zip(batch["qid"], results, strict=True))
    return run


def to_ranx_run(run: Run, name: str = "run") -> ranx.Run:
    # ranx re-sorts by score, which could reorder tied hits. Strictly decreasing
    # rank-based scores keep exactly the order the retriever produced.
    return ranx.Run(
        {
            qid: {h.doc_id: float(len(hits) - i) for i, h in enumerate(hits)}
            for qid, hits in run.items()
        },
        name=name,
    )


def to_ranx_qrels(qrels: pd.DataFrame, qids=None) -> ranx.Qrels:
    if qids is not None:
        qrels = qrels[qrels["qid"].isin(set(qids))]
    # Built from a dict: Qrels.from_df rejects pandas' string dtype.
    judgments: dict[str, dict[str, int]] = {}
    for qid, doc_id, rel in qrels[["qid", "doc_id", "rel"]].itertuples(index=False):
        judgments.setdefault(str(qid), {})[str(doc_id)] = int(rel)
    return ranx.Qrels(judgments)


def evaluate(run: Run, qrels: pd.DataFrame, metrics: list[str] = METRICS) -> dict[str, float]:
    """Macro-averaged metrics over the queries in `run`. Missing results count as zero."""
    scores = ranx.evaluate(
        to_ranx_qrels(qrels, run.keys()), to_ranx_run(run), metrics, make_comparable=True
    )
    if not isinstance(scores, dict):  # ranx returns a bare number for a single metric
        scores = {metrics[0]: scores}
    return {m: float(scores[m]) for m in metrics}


def paired_test(
    system: tuple[str, Run],
    baseline: tuple[str, Run],
    qrels: pd.DataFrame,
    metrics: list[str] = SIGNIFICANCE_METRICS,
    n_comparisons: int = len(PLANNED_COMPARISONS),
) -> list[dict]:
    """Two-sided paired randomization test (Fisher) per metric, Bonferroni-corrected."""
    (sys_name, sys_run), (base_name, base_run) = system, baseline
    if set(sys_run) != set(base_run):
        raise ValueError("Runs must cover the same queries for a paired test")
    alpha = ALPHA / n_comparisons
    report = ranx.compare(
        to_ranx_qrels(qrels, sys_run.keys()),
        [to_ranx_run(sys_run, sys_name), to_ranx_run(base_run, base_name)],
        metrics,
        stat_test="fisher",
        n_permutations=N_PERMUTATIONS,
        rounding_digits=6,
        make_comparable=True,
    ).to_dict()
    rows = []
    for m in metrics:
        wtl = report[sys_name]["win_tie_loss"][base_name][m]
        p_value = report[sys_name]["comparisons"][base_name][m]
        sys_mean, base_mean = report[sys_name]["scores"][m], report[base_name]["scores"][m]
        rows.append(
            {
                "system": sys_name,
                "baseline": base_name,
                "metric": m,
                "system_mean": sys_mean,
                "baseline_mean": base_mean,
                "delta": sys_mean - base_mean,
                "p_value": p_value,
                "alpha_corrected": alpha,
                "significant": p_value < alpha,
                "wins": wtl["W"],
                "ties": wtl["T"],
                "losses": wtl["L"],
            }
        )
    return rows


def random_run(qids, doc_ids, k: int = RUN_DEPTH, seed: int = 0) -> Run:
    rng = np.random.default_rng(seed)
    return {
        qid: [Hit(str(d), 0.0) for d in rng.choice(doc_ids, size=k, replace=False)] for qid in qids
    }


def oracle_run(qrels: pd.DataFrame, qids) -> Run:
    relevant = qrels.groupby("qid")["doc_id"].apply(sorted)
    return {qid: [Hit(d, 1.0) for d in relevant[qid]] for qid in qids}


def save_run(run: Run, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {qid: [[h.doc_id, h.score] for h in hits] for qid, hits in run.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_run(path: Path) -> Run:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {qid: [Hit(d, s) for d, s in hits] for qid, hits in payload.items()}


def benchmark_latency(
    retriever: Retriever, queries: list[str], k: int = 10, warmup: int = 10
) -> dict[str, float]:
    """Sequential single-query latency in milliseconds, after a warm-up."""
    for q in queries[:warmup]:
        retriever.search(q, k)
    timings = []
    for q in queries:
        start = time.perf_counter()
        retriever.search(q, k)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(np.mean(timings)),
        "n_queries": len(timings),
    }


def git_commit() -> str:
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    commit = git("rev-parse", "--short", "HEAD")
    return f"{commit}-dirty" if git("status", "--porcelain", "--untracked-files=no") else commit


def record_metrics(
    config: str, split: str, scores: dict[str, float], latency: dict | None = None
) -> pd.DataFrame:
    """Insert or replace the row for (config, split) in results/metrics_{split}.csv."""
    path = RESULTS_DIR / f"metrics_{split}.csv"
    row = {"config": config, "split": split, **scores, **(latency or {})}
    row |= {
        "hardware": f"{platform.processor()} ({os.cpu_count()} threads)",
        "commit": git_commit(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not table.empty:
        table = table[table["config"] != config]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(path, index=False)
    return table
