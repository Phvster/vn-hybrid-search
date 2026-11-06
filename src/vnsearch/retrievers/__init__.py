from vnsearch.config import ARTIFACTS_DIR
from vnsearch.retrievers.base import Hit, Retriever, top_k


def load_retriever(cfg: dict) -> Retriever:
    """Load a built retriever from its experiment config (configs/*.yaml)."""
    if cfg["type"] == "bm25":
        from vnsearch.retrievers.bm25 import BM25Retriever

        return BM25Retriever.load(ARTIFACTS_DIR / cfg["index"])
    raise ValueError(f"Unknown retriever type: {cfg['type']}")


__all__ = ["Hit", "Retriever", "load_retriever", "top_k"]
