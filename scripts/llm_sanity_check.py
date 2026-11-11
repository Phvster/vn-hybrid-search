"""Manual sanity check of answer generation on a seeded sample of test questions.

Retrieves R3's top-5, calls Claude Haiku once per question and writes
results/llm_sanity_check.md for reading: is the answer grounded, does it cite the labelled
article, and does it say "not found" when the labelled article was not retrieved?
Costs about $0.001 per question.
"""

import argparse

import numpy as np
from dotenv import load_dotenv

from vnsearch import data, evaluation, generation
from vnsearch.config import RESULTS_DIR
from vnsearch.generation import Article

SEED = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    load_dotenv()

    queries = evaluation.split_queries("test", allow_test=True)
    relevant = data.load_qrels().groupby("qid")["doc_id"].apply(set)
    corpus = data.load_corpus().set_index("doc_id")
    run = evaluation.load_run(RESULTS_DIR / "runs" / "r3_test.json")  # saved R3 top-k
    sample = queries.iloc[np.random.default_rng(SEED).choice(len(queries), args.n, replace=False)]

    lines = [
        "# LLM sanity check (test sample)",
        "",
        f"Model `{generation.MODEL}`, R3 top-{generation.TOP_K}, seed {SEED}.",
        "",
    ]
    total_cost, gold_retrieved, gold_cited = 0.0, 0, 0
    for qid, query in zip(sample["qid"], sample["text"], strict=True):
        hits = run[qid][: generation.TOP_K]
        articles = [
            Article(h.doc_id, corpus.at[h.doc_id, "title"], corpus.at[h.doc_id, "text"])
            for h in hits
        ]
        result = generation.answer(query, articles)
        total_cost += result.cost_usd
        retrieved = bool(relevant[qid] & {a.doc_id for a in articles})
        cited = {c.doc_id for s in result.segments for c in s.citations}
        gold_retrieved += retrieved
        gold_cited += bool(relevant[qid] & cited)
        text = "".join(s.text for s in result.segments).strip()
        lines += [
            f"## {query}",
            f"- labelled: {', '.join(sorted(relevant[qid]))} · in top-5: **{retrieved}** · "
            f"cited: **{bool(relevant[qid] & cited)}** · cited docs: {sorted(cited) or '–'}",
            f"- stop: {result.stop_reason} · {result.input_tokens} in / {result.output_tokens} out",
            "",
            text or "*(empty)*",
            "",
        ]
    summary = (
        f"**Summary:** labelled article in top-5 for {gold_retrieved}/{args.n}; cited by the "
        f"answer for {gold_cited}/{args.n}. Total cost ≈ ${total_cost:.4f}."
    )
    lines[4:4] = [summary, ""]
    out = RESULTS_DIR / "llm_sanity_check.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(summary, "->", out)


if __name__ == "__main__":
    main()
