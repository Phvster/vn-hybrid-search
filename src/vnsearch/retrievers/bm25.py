"""R0 / R1: BM25 with bm25s. The two configurations differ only in the tokenizer."""

import json
import time
from pathlib import Path

import bm25s
import numpy as np

from vnsearch.retrievers.base import Hit, Retriever, top_k
from vnsearch.text import TOKENIZERS

METHOD = "lucene"


class BM25Retriever(Retriever):
    def __init__(self, name: str, tokenizer: str, model: bm25s.BM25, doc_ids: np.ndarray):
        self.name = name
        self.tokenizer = tokenizer
        self.tokenize = TOKENIZERS[tokenizer]
        self.model = model
        self.doc_ids = doc_ids
        self.build_info: dict = {}

    @classmethod
    def build(
        cls,
        name: str,
        tokenizer: str,
        contents: list[str],
        doc_ids: list[str],
        k1: float,
        b: float,
    ) -> "BM25Retriever":
        start = time.perf_counter()
        tokens = [TOKENIZERS[tokenizer](text) for text in contents]
        tokenize_seconds = time.perf_counter() - start
        model = bm25s.BM25(method=METHOD, k1=k1, b=b)
        start = time.perf_counter()
        model.index(tokens, show_progress=False)
        retriever = cls(name, tokenizer, model, np.array(doc_ids))
        retriever.build_info = {
            "tokenize_seconds": tokenize_seconds,
            "index_seconds": time.perf_counter() - start,
            "vocab_size": len(model.vocab_dict),
        }
        return retriever

    def search(self, query: str, k: int) -> list[Hit]:
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self.model.get_scores(tokens)
        # Zero means no shared token: not a real match, and it would earn RRF credit in R3.
        return [h for h in top_k(scores, self.doc_ids, k) if h.score > 0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(str(directory / "model"))
        (directory / "doc_ids.json").write_text(json.dumps(self.doc_ids.tolist()), encoding="utf-8")
        meta = {
            "name": self.name,
            "tokenizer": self.tokenizer,
            "method": METHOD,
            "k1": self.model.k1,
            "b": self.model.b,
            "n_docs": len(self.doc_ids),
            **self.build_info,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "BM25Retriever":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        doc_ids = np.array(json.loads((directory / "doc_ids.json").read_text(encoding="utf-8")))
        model = bm25s.BM25.load(str(directory / "model"))
        return cls(meta["name"], meta["tokenizer"], model, doc_ids)
