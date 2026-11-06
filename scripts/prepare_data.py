"""Download the Zalo Legal mirror, clean it, validate it and write data/processed/."""

import pandas as pd

from vnsearch import data


def main() -> None:
    paths = data.download_raw()
    corpus = data.prepare_corpus(pd.read_parquet(paths["corpus"]))
    queries = data.prepare_queries(data.read_jsonl(paths["queries"], ["_id"]))
    qrels = data.prepare_qrels(
        data.read_jsonl(paths["qrels_train"], ["query-id", "corpus-id"]),
        data.read_jsonl(paths["qrels_test"], ["query-id", "corpus-id"]),
    )
    data.validate(corpus, queries, qrels)
    split = data.make_split(queries["qid"])
    data.save_processed(corpus, queries, qrels, split)

    rel_per_query = qrels.groupby("qid").size()
    print(f"corpus:  {len(corpus):,} articles ({(corpus['text'] == '').sum()} with empty body)")
    print(f"queries: {len(queries):,} | qrels: {len(qrels):,} pairs")
    print(f"relevant/query: {rel_per_query.value_counts().sort_index().to_dict()}")
    print(f"split:   dev={len(split['dev'])} test={len(split['test'])} (seed={split['seed']})")


if __name__ == "__main__":
    main()
