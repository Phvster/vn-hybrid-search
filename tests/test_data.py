import pandas as pd
import pytest

from vnsearch import data


def raw_corpus():
    return pd.DataFrame(
        {
            "id": ["01/2009/tt-bnn+1", "01/2009/tt-bnn+2"],
            "title": ["Điều 1. Phạm vi", "Điều 2. Bãi bỏ"],
            "text": ["Thông tư này  hướng dẫn.", ""],
        }
    )


def test_prepare_corpus_splits_id_and_builds_contents():
    corpus = data.prepare_corpus(raw_corpus())
    assert corpus.loc[0, "law_id"] == "01/2009/tt-bnn"
    assert corpus.loc[0, "article_id"] == "1"
    assert corpus.loc[0, "contents"] == "Điều 1. Phạm vi\nThông tư này hướng dẫn."
    # Empty body: the article stays searchable through its title.
    assert corpus.loc[1, "contents"] == "Điều 2. Bãi bỏ"


def test_prepare_queries_drops_identical_duplicates():
    raw = pd.DataFrame({"_id": ["a", "a", "b"], "text": ["câu 1", "câu 1", "câu 2"]})
    assert data.prepare_queries(raw)["qid"].tolist() == ["a", "b"]


def test_prepare_queries_rejects_conflicting_duplicates():
    raw = pd.DataFrame({"_id": ["a", "a"], "text": ["câu 1", "khác"]})
    with pytest.raises(ValueError):
        data.prepare_queries(raw)


def test_hex_like_ids_stay_strings(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"_id": "123e4567", "text": "x"}\n', encoding="utf-8")
    assert data.read_jsonl(path, ["_id"]).loc[0, "_id"] == "123e4567"


def test_prepare_qrels_merges_and_dedupes():
    a = pd.DataFrame({"query-id": ["q1", "q2"], "corpus-id": ["d1", "d2"], "score": [1, 1]})
    b = pd.DataFrame({"query-id": ["q1"], "corpus-id": ["d1"], "score": [1]})
    qrels = data.prepare_qrels(a, b)
    assert len(qrels) == 2
    assert list(qrels.columns) == ["qid", "doc_id", "rel"]


def test_validate_detects_unknown_doc_id():
    corpus = data.prepare_corpus(raw_corpus())
    queries = pd.DataFrame({"qid": ["q1"], "text": ["x"]})
    qrels = pd.DataFrame({"qid": ["q1"], "doc_id": ["missing+9"], "rel": [1]})
    with pytest.raises(ValueError, match="not in corpus"):
        data.validate(corpus, queries, qrels)


def test_split_is_disjoint_complete_and_sized():
    qids = [f"q{i}" for i in range(100)]
    split = data.make_split(qids)
    dev, test = set(split["dev"]), set(split["test"])
    assert not dev & test
    assert dev | test == set(qids)
    assert len(dev) == 30


def test_split_is_deterministic_and_order_independent():
    qids = [f"q{i}" for i in range(100)]
    assert data.make_split(qids) == data.make_split(list(reversed(qids)))
    assert data.make_split(qids, seed=1)["dev"] != data.make_split(qids)["dev"]
