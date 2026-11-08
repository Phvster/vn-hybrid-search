from vnsearch.config import ARTIFACTS_DIR
from vnsearch.retrievers.base import Hit, Retriever, top_k


def load_retriever(cfg: dict) -> Retriever:
    """Load a built retriever from its experiment config (configs/*.yaml)."""
    if cfg["type"] == "bm25":
        from vnsearch.retrievers.bm25 import BM25Retriever

        return BM25Retriever.load(ARTIFACTS_DIR / cfg["index"])
    if cfg["type"] == "dense":
        from vnsearch.retrievers.dense import DenseRetriever

        return DenseRetriever.load(ARTIFACTS_DIR / cfg["index"], cfg)
    if cfg["type"] == "hybrid":
        from vnsearch.config import load_config
        from vnsearch.retrievers.hybrid import HybridRetriever

        components = [load_retriever(load_config(name)) for name in cfg["components"]]
        return HybridRetriever(cfg["name"], components, k=cfg["rrf_k"], depth=cfg["depth"])
    if cfg["type"] == "rerank":
        from vnsearch import data
        from vnsearch.config import load_config
        from vnsearch.retrievers.rerank import RerankRetriever, load_cross_encoder

        contents = data.load_corpus().set_index("doc_id")["contents"]
        base = load_retriever(load_config(cfg["base"]))
        return RerankRetriever(
            cfg["name"], base, load_cross_encoder(cfg), contents, cfg["n"], cfg["batch_size"]
        )
    raise ValueError(f"Unknown retriever type: {cfg['type']}")


__all__ = ["Hit", "Retriever", "load_retriever", "top_k"]
