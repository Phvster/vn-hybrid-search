import json

import numpy as np
import pytest

from vnsearch.retrievers.dense import DenseRetriever, check_embeddings

CFG = {
    "name": "r2",
    "model": "BAAI/bge-m3",
    "revision": "abc",
    "max_seq_length": 2048,
    "query_batch_size": 32,
}
META = {"model": "BAAI/bge-m3", "revision": "abc", "max_seq_length": 2048}


class FakeEncoder:
    """Maps a text to a fixed unit vector so tests need no model download."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        out = np.array([self.vectors[t] for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def unit(*rows):
    m = np.array(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def make_retriever():
    # Docs "b" and "a" are identical vectors (duplicate texts in the corpus).
    embeddings = unit([1, 0, 0], [0.9, 0.1, 0], [0.9, 0.1, 0], [0, 0, 1])
    encoder = FakeEncoder({"query": [1, 0.1, 0], "other": [0, 0, 1]})
    return DenseRetriever("r2", encoder, embeddings, ["top", "b", "a", "far"])


def test_ranks_by_cosine_and_breaks_ties_by_doc_id():
    # cos(query, a) = cos(query, b) ≈ 0.99993 > cos(query, top) ≈ 0.99504
    hits = make_retriever().search("query", k=3)
    assert [h.doc_id for h in hits] == ["a", "b", "top"]
    assert hits[0].score == hits[1].score


def test_batch_and_single_search_agree():
    retriever = make_retriever()
    batch = retriever.search_batch(["query", "other"], k=2)
    assert batch == [retriever.search("query", k=2), retriever.search("other", k=2)]


def test_k_larger_than_corpus():
    assert len(make_retriever().search("query", k=50)) == 4


def test_check_rejects_revision_mismatch():
    with pytest.raises(ValueError, match="revision"):
        check_embeddings(unit([1, 0]), ["d"], {**META, "revision": "old"}, CFG)


def test_check_rejects_unnormalized_or_misaligned_vectors():
    with pytest.raises(ValueError, match="normalized"):
        check_embeddings(np.array([[3.0, 4.0]]), ["d"], META, CFG)
    with pytest.raises(ValueError, match="aligned"):
        check_embeddings(unit([1, 0], [0, 1]), ["d"], META, CFG)


def test_load_round_trip(tmp_path):
    embeddings = unit([1, 0, 0], [0, 1, 0]).astype(np.float16)
    np.save(tmp_path / "embeddings.npy", embeddings)
    (tmp_path / "doc_ids.json").write_text(json.dumps(["x", "y"]))
    (tmp_path / "meta.json").write_text(json.dumps(META))
    encoder = FakeEncoder({"q": [0, 1, 0]})
    retriever = DenseRetriever.load(tmp_path, CFG, encoder=encoder)
    assert retriever.search("q", k=1)[0].doc_id == "y"
