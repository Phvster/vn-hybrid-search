from vnsearch.retrievers.bm25 import BM25Retriever
from vnsearch.text import syllable_tokens, word_tokens

DOCS = {
    "d_injury": "Người nào cố ý gây thương tích cho người khác thì bị truy cứu trách nhiệm hình sự",
    "d_finance": "Doanh nghiệp phải phân tích báo cáo tài chính và tích lũy quỹ dự phòng.",
    "d_land": "Hộ gia đình được cấp giấy chứng nhận quyền sử dụng đất.",
}


def build(tokenizer: str) -> BM25Retriever:
    return BM25Retriever.build(
        tokenizer, tokenizer, list(DOCS.values()), list(DOCS.keys()), k1=1.5, b=0.75
    )


def test_syllable_tokens_lowercase_and_drop_punctuation():
    assert syllable_tokens("Điều 5. Phạt tiền?") == ["điều", "5", "phạt", "tiền"]


def test_word_tokens_keep_compounds_and_drop_punctuation():
    tokens = word_tokens("Cố ý gây thương tích bị truy cứu trách nhiệm hình sự như thế nào?")
    assert "cố_ý" in tokens and "thương_tích" in tokens
    assert "?" not in tokens and all(t == t.lower() for t in tokens)


def test_syllable_bm25_matches_unrelated_syllable_but_word_bm25_does_not():
    # "thương tích" shares the syllable "tích" with "phân tích" / "tích lũy" (finance doc).
    query = "thương tích"
    r0_docs = {h.doc_id for h in build("syllable").search(query, k=3)}
    r1_docs = {h.doc_id for h in build("word").search(query, k=3)}
    assert "d_finance" in r0_docs
    assert r1_docs == {"d_injury"}


def test_zero_score_documents_are_not_returned():
    assert [h.doc_id for h in build("word").search("quyền sử dụng đất", k=3)] == ["d_land"]


def test_query_without_known_tokens_returns_nothing():
    assert build("syllable").search("xyz", k=3) == []
    assert build("syllable").search("?!", k=3) == []


def test_save_and_load_give_identical_results(tmp_path):
    original = build("word")
    original.save(tmp_path / "idx")
    loaded = BM25Retriever.load(tmp_path / "idx")
    query = "cố ý gây thương tích"
    assert loaded.tokenizer == "word"
    assert loaded.search(query, k=3) == original.search(query, k=3)
