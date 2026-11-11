"""Error analysis on a split (protocol: Error analysis). Writes results/error_analysis_{split}.md.

1. Segment breakdown: Hit@10 / nDCG@10 per config for query and article properties that test
   hypotheses stated earlier (numbers in the query, articles truncated by R2, multi-relevant,
   query length).
2. Seeded samples of questions where R3 beats / loses to R2 and where every config fails,
   for manual categorization.
"""

import argparse
import re

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from vnsearch import data, evaluation
from vnsearch.config import RESULTS_DIR, load_config

CONFIGS = ["r0", "r1", "r2", "r3", "r4"]
SAMPLE_WINS = SAMPLE_LOSSES = 30
SAMPLE_ALL_FAIL = 15
SEED = 0
_DIGIT = re.compile(r"\d")


def per_query(run: evaluation.Run, relevant: dict[str, set[str]]) -> pd.DataFrame:
    """Rank of the first relevant hit, Hit@10 and binary nDCG@10 per query."""
    rows = []
    for qid, rel in relevant.items():
        ids = [h.doc_id for h in run.get(qid, [])]
        gains = [1.0 if d in rel else 0.0 for d in ids[:10]]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), 10)))
        first = next((i + 1 for i, d in enumerate(ids) if d in rel), None)
        rows.append(
            {"qid": qid, "first_rank": first, "hit10": float(any(gains)), "ndcg10": dcg / idcg}
        )
    return pd.DataFrame(rows).set_index("qid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    queries = evaluation.split_queries(args.split, allow_test=args.allow_test).set_index("qid")
    qrels = data.load_qrels()
    qrels = qrels[qrels["qid"].isin(queries.index)]
    relevant = qrels.groupby("qid")["doc_id"].apply(set).to_dict()
    corpus = data.load_corpus().set_index("doc_id")

    configs = [c for c in CONFIGS if (RESULTS_DIR / "runs" / f"{c}_{args.split}.json").exists()]
    runs = {
        c: evaluation.load_run(RESULTS_DIR / "runs" / f"{c}_{args.split}.json") for c in configs
    }
    stats = {c: per_query(runs[c], relevant) for c in configs}
    for c in configs:  # cross-check against the official numbers
        official = evaluation.evaluate(runs[c], qrels, ["ndcg@10"])["ndcg@10"]
        assert abs(stats[c]["ndcg10"].mean() - official) < 1e-9, f"{c}: per-query nDCG mismatch"

    # Query / article properties.
    tokenizer = AutoTokenizer.from_pretrained(load_config("r2")["model"])
    gold_docs = sorted(set(qrels["doc_id"]))
    gold_tokens = dict(
        zip(
            gold_docs,
            (
                len(ids)
                for ids in tokenizer(corpus.loc[gold_docs, "contents"].tolist())["input_ids"]
            ),
            strict=True,
        )
    )
    max_len = load_config("r2")["max_seq_length"]
    props = pd.DataFrame(index=queries.index)
    props["has_number"] = queries["text"].str.contains(_DIGIT)
    props["gold_truncated_by_r2"] = [
        any(gold_tokens[d] > max_len for d in relevant[q]) for q in props.index
    ]
    props["multi_relevant"] = [len(relevant[q]) > 1 for q in props.index]
    n_syll = queries["text"].str.split().str.len()
    props["query_length"] = pd.cut(n_syll, [0, 12, 20, 1000], labels=["≤12", "13–20", ">20"])

    segments = [("All", pd.Series(True, index=props.index))]
    for col in ("has_number", "gold_truncated_by_r2", "multi_relevant"):
        segments.append((f"{col} = yes", props[col]))
    for label in props["query_length"].cat.categories:
        segments.append((f"query length {label} syllables", props["query_length"] == label))

    header = (
        "| Segment | n | " + " | ".join(f"{c.upper()} Hit@10 / nDCG@10" for c in configs) + " |"
    )
    lines = [header, "|" + "---|" * (2 + len(configs))]
    for name, mask in segments:
        cells = [
            f"{stats[c].loc[mask[mask].index, 'hit10'].mean():.3f} / "
            f"{stats[c].loc[mask[mask].index, 'ndcg10'].mean():.3f}"
            for c in configs
        ]
        lines.append(f"| {name} | {int(mask.sum())} | " + " | ".join(cells) + " |")

    # Samples for manual reading.
    delta = stats["r3"]["ndcg10"] - stats["r2"]["ndcg10"]
    rng = np.random.default_rng(SEED)

    def sample(ids, n):
        ids = sorted(ids)
        return sorted(rng.choice(ids, size=min(n, len(ids)), replace=False)) if ids else []

    wins = sample(delta[delta > 0].index, SAMPLE_WINS)
    losses = sample(delta[delta < 0].index, SAMPLE_LOSSES)
    all_fail = sample(
        [
            q
            for q in props.index
            if all(
                stats[c].loc[q, "first_rank"] is None or stats[c].loc[q, "first_rank"] > 20
                for c in configs
            )
        ],
        SAMPLE_ALL_FAIL,
    )

    def describe(qid: str) -> str:
        ranks = ", ".join(f"{c.upper()}={_rank(stats[c].loc[qid, 'first_rank'])}" for c in configs)
        gold = "; ".join(f"`{d}` {corpus.at[d, 'title'][:90]}" for d in sorted(relevant[qid]))
        tops = []
        for c in ("r1", "r2"):
            titles = [corpus.at[h.doc_id, "title"][:60] for h in runs[c][qid][:2]]
            tops.append(f"{c.upper()} top-2: " + " · ".join(titles))
        head = [f"- **{queries.at[qid, 'text']}**", f"gold: {gold}"]
        return "\n  - ".join([*head, f"first relevant rank: {ranks}", *tops])

    counts = {
        k: int(v)
        for k, v in {
            "wins": (delta > 0).sum(),
            "ties": (delta == 0).sum(),
            "losses": (delta < 0).sum(),
        }.items()
    }
    report = [
        f"# Error analysis ({args.split})",
        "",
        "Generated by `scripts/error_analysis.py`. Per-query nDCG@10 matches the official "
        "metrics (asserted).",
        "",
        "## Segment breakdown",
        "",
        *lines,
        "",
        f"R3 vs R2 per query (nDCG@10): {counts['wins']} wins, {counts['ties']} ties, "
        f"{counts['losses']} losses.",
        "",
        f"## R3 beats R2 (sample of {len(wins)})",
        "",
        *map(describe, wins),
        "",
        f"## R3 loses to R2 (sample of {len(losses)})",
        "",
        *map(describe, losses),
        "",
        f"## Every config misses the top-20 (sample of {len(all_fail)})",
        "",
        *map(describe, all_fail),
        "",
    ]
    out = RESULTS_DIR / f"error_analysis_{args.split}.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


def _rank(value) -> str:
    return "–" if value is None or pd.isna(value) else str(int(value))


if __name__ == "__main__":
    main()
