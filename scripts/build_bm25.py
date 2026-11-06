"""Build and save the BM25 indexes for the given configs (default: r0 r1)."""

import argparse

from vnsearch import data
from vnsearch.config import ARTIFACTS_DIR, load_config
from vnsearch.retrievers.bm25 import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", default=["r0", "r1"])
    args = parser.parse_args()

    corpus = data.load_corpus()
    for name in args.configs:
        cfg = load_config(name)
        retriever = BM25Retriever.build(
            name,
            cfg["tokenizer"],
            corpus["contents"].tolist(),
            corpus["doc_id"].tolist(),
            k1=cfg["k1"],
            b=cfg["b"],
        )
        out_dir = ARTIFACTS_DIR / cfg["index"]
        retriever.save(out_dir)
        size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
        info = retriever.build_info
        print(
            f"{name}: tokenize {info['tokenize_seconds']:.1f}s, "
            f"index {info['index_seconds']:.1f}s, vocab {info['vocab_size']:,}, "
            f"{size_mb:.1f} MB -> {out_dir}"
        )


if __name__ == "__main__":
    main()
