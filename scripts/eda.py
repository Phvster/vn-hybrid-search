"""Exploratory analysis that drives preprocessing decisions. Writes results/eda_report.md."""

import json
import re
import time

import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from transformers import AutoTokenizer

from vnsearch import data
from vnsearch.config import RESULTS_DIR

DENSE_MODEL = "BAAI/bge-m3"
SEQ_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
SAMPLE_SIZE = 2000
SEED = 0
_SYLLABLE = re.compile(r"\w+")
# "Điều" capitalized after a lowercase word and not followed by a number, e.g. "bảo vệ đê Điều".
_MIDSENTENCE_DIEU = re.compile(r"(?<=[a-zà-ỹđ] )Điều(?! \d)")


def describe(values) -> dict:
    v = np.asarray(values)
    return {
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": int(v.max()),
    }


def coverage(lengths, limits=SEQ_LENGTHS) -> dict:
    v = np.asarray(lengths)
    return {str(n): float((v <= n).mean()) for n in limits}


def syllables(text: str) -> set[str]:
    return set(_SYLLABLE.findall(text.lower()))


def token_lengths(corpus: pd.DataFrame, queries: pd.DataFrame, qrels: pd.DataFrame) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL)
    doc_tokens = np.array([len(ids) for ids in tokenizer(corpus["contents"].tolist())["input_ids"]])
    query_tokens = [len(ids) for ids in tokenizer(queries["text"].tolist())["input_ids"]]
    doc_syllables = corpus["contents"].str.split().str.len().to_numpy()
    relevant = corpus["doc_id"].isin(set(qrels["doc_id"])).to_numpy()
    return {
        "tokens_per_syllable": float(doc_tokens.sum() / doc_syllables.sum()),
        "total_doc_tokens": int(doc_tokens.sum()),
        "all_articles": describe(doc_tokens),
        "relevant_articles": describe(doc_tokens[relevant]),
        "coverage_all": coverage(doc_tokens),
        "coverage_relevant": coverage(doc_tokens[relevant]),
        "queries": describe(query_tokens),
    }


def lexical_overlap(corpus: pd.DataFrame, queries: pd.DataFrame, qrels: pd.DataFrame) -> dict:
    """Share of a query's syllables that appear in its relevant article vs. a random article."""
    contents = corpus.set_index("doc_id")["contents"]
    query_text = queries.set_index("qid")["text"]
    rng = np.random.default_rng(SEED)
    random_docs = rng.choice(contents.index.to_numpy(), size=len(qrels))
    relevant, baseline = [], []
    for (qid, doc_id), rand_id in zip(
        qrels[["qid", "doc_id"]].itertuples(index=False), random_docs, strict=True
    ):
        q = syllables(query_text[qid])
        relevant.append(len(q & syllables(contents[doc_id])) / len(q))
        baseline.append(len(q & syllables(contents[rand_id])) / len(q))
    return {"relevant_mean": float(np.mean(relevant)), "random_mean": float(np.mean(baseline))}


def lowercase_order(corpus: pd.DataFrame, queries: pd.DataFrame) -> dict:
    """Compare segment-then-lowercase with lowercase-then-segment on a sample."""
    articles = corpus["contents"].sample(SAMPLE_SIZE, random_state=SEED).tolist()
    sample = articles + queries["text"].sample(SAMPLE_SIZE, random_state=SEED).tolist()
    start = time.perf_counter()
    cased = [ViTokenizer.tokenize(t).lower().split() for t in articles]
    article_seconds = time.perf_counter() - start
    cased += [ViTokenizer.tokenize(t).lower().split() for t in sample[len(articles) :]]
    lowered = [ViTokenizer.tokenize(t.lower()).split() for t in sample]
    differ = sum(a != b for a, b in zip(cased, lowered, strict=True))
    examples = []
    for text, a, b in zip(sample, cased, lowered, strict=True):
        if a != b and len(examples) < 5:
            only_cased = sorted(set(a) - set(b))[:4]
            only_lowered = sorted(set(b) - set(a))[:4]
            examples.append(
                {"text": text[:120], "cased_first": only_cased, "lower_first": only_lowered}
            )
    return {
        "texts_compared": len(sample),
        "texts_with_different_tokens": differ,
        "share_different": differ / len(sample),
        "pyvi_seconds_per_1000_articles": article_seconds / len(articles) * 1000,
        "estimated_corpus_minutes": article_seconds / len(articles) * len(corpus) / 60,
        "examples": examples,
    }


def midsentence_dieu(corpus: pd.DataFrame, queries: pd.DataFrame) -> dict:
    hits = corpus["contents"].str.count(_MIDSENTENCE_DIEU.pattern)
    examples = []
    for text in corpus.loc[hits > 0, "contents"].head(200):
        for m in _MIDSENTENCE_DIEU.finditer(text):
            examples.append(text[max(0, m.start() - 40) : m.end() + 20].replace("\n", " "))
            break
        if len(examples) >= 6:
            break
    return {
        "articles_affected": int((hits > 0).sum()),
        "occurrences": int(hits.sum()),
        "queries_affected": int(queries["text"].str.contains(_MIDSENTENCE_DIEU.pattern).sum()),
        "examples": examples,
    }


def write_report(stats: dict) -> None:
    t, o, lc, d = stats["tokens"], stats["overlap"], stats["lowercase"], stats["dieu"]
    rows = "\n".join(
        f"| {n} | {t['coverage_all'][str(n)]:.2%} | {t['coverage_relevant'][str(n)]:.2%} |"
        for n in SEQ_LENGTHS
    )

    def stat_row(name, s):
        return f"| {name} | {s['median']:.0f} | {s['p90']:.0f} | {s['p95']:.0f} | {s['p99']:.0f} | {s['max']} |"

    examples = "\n".join(
        f"- `{e['text']}` → cased-first only: {e['cased_first']}, lower-first only: {e['lower_first']}"
        for e in lc["examples"]
    )
    dieu_examples = "\n".join(f"- `…{e}…`" for e in d["examples"])
    report = f"""# EDA report

Generated by `scripts/eda.py`. Numbers feed the preprocessing decisions in
[docs/EXPERIMENT_PROTOCOL.md](../docs/EXPERIMENT_PROTOCOL.md).

## 1. Length in bge-m3 tokens (`contents` = title + body, special tokens included)

Tokens per whitespace syllable: **{t["tokens_per_syllable"]:.2f}**. Corpus total: {t["total_doc_tokens"]:,} tokens.

| Set | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
{stat_row("All articles", t["all_articles"])}
{stat_row("Relevant articles", t["relevant_articles"])}
{stat_row("Queries", t["queries"])}

Share of articles that fit without truncation:

| max_seq_length | all articles | relevant articles |
|---|---|---|
{rows}

## 2. Lexical overlap (share of query syllables found in the article)

| Article | mean overlap |
|---|---|
| Relevant article | {o["relevant_mean"]:.2%} |
| Random article (baseline) | {o["random_mean"]:.2%} |

## 3. pyvi: segment-then-lowercase vs lowercase-then-segment

Sample: {lc["texts_compared"]:,} texts ({SAMPLE_SIZE} articles + {SAMPLE_SIZE} queries).
Texts whose token lists differ: **{lc["texts_with_different_tokens"]:,} ({lc["share_different"]:.1%})**.
Speed: {lc["pyvi_seconds_per_1000_articles"]:.1f} s per 1,000 articles
(≈ {lc["estimated_corpus_minutes"]:.1f} min for the full corpus).

Examples:
{examples}

## 4. Capitalized "Điều" in the middle of a sentence

{d["articles_affected"]:,} articles ({d["occurrences"]:,} occurrences), {d["queries_affected"]} queries.

{dieu_examples}
"""
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "eda_report.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "eda_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    corpus, queries, qrels = data.load_corpus(), data.load_queries(), data.load_qrels()
    stats = {
        "tokens": token_lengths(corpus, queries, qrels),
        "overlap": lexical_overlap(corpus, queries, qrels),
        "lowercase": lowercase_order(corpus, queries),
        "dieu": midsentence_dieu(corpus, queries),
    }
    write_report(stats)
    print((RESULTS_DIR / "eda_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
