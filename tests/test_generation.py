from types import SimpleNamespace as NS

import pytest

from vnsearch import generation
from vnsearch.generation import Article

ARTICLES = [
    Article(
        "100/2019/nđ-cp+6",
        "Điều 6. Xử phạt người điều khiển xe mô tô",
        "Phạt tiền từ 400.000 đồng...",
    ),
    Article("01/2009/tt-bnn+2", "Điều 2. Bãi bỏ", ""),
]


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def response(blocks, stop_reason="end_turn"):
    return NS(
        content=blocks, stop_reason=stop_reason, usage=NS(input_tokens=3000, output_tokens=400)
    )


def cited(text, index, quote):
    citation = NS(type="char_location", document_index=index, document_title="t", cited_text=quote)
    return NS(type="text", text=text, citations=[citation])


def test_request_uses_one_cited_document_per_article():
    request = generation.build_request("Đi xe máy không đội mũ bị phạt bao nhiêu?", ARTICLES)
    content = request["messages"][0]["content"]
    docs = [c for c in content if c["type"] == "document"]
    assert len(docs) == 2 and all(d["citations"] == {"enabled": True} for d in docs)
    assert docs[0]["title"] == "Điều 6. Xử phạt người điều khiển xe mô tô (100/2019/NĐ-CP)"
    assert docs[1]["source"]["data"] == "Điều 2. Bãi bỏ"  # empty body falls back to title
    assert content[-1]["type"] == "text"
    assert request["model"] == "claude-haiku-5-5"
    assert "temperature" not in request


def test_long_article_is_truncated_with_visible_note():
    long_article = Article("x+1", "Điều 1.", "a" * (generation.MAX_ARTICLE_CHARS + 1000))
    data = generation.document_block(long_article)["source"]["data"]
    assert len(data) == generation.MAX_ARTICLE_CHARS + len(generation.TRUNCATION_NOTE)
    assert data.endswith(generation.TRUNCATION_NOTE)


@pytest.mark.parametrize("query", ["", "   ", "x" * 501])
def test_invalid_questions_are_rejected_before_any_api_call(query):
    client = FakeClient(response([]))
    with pytest.raises(ValueError):
        generation.answer(query, ARTICLES, client=client)
    assert client.requests == []


def test_citations_map_back_to_doc_ids_and_thinking_is_skipped():
    blocks = [
        NS(type="thinking", thinking=""),
        cited("Mức phạt là 400.000 đồng.", 0, "Phạt tiền từ 400.000 đồng"),
        NS(type="text", text=" Không có quy định khác.", citations=None),
    ]
    result = generation.answer("câu hỏi", ARTICLES, client=FakeClient(response(blocks)))
    assert [s.text for s in result.segments] == [
        "Mức phạt là 400.000 đồng.",
        " Không có quy định khác.",
    ]
    assert result.segments[0].citations[0].doc_id == "100/2019/nđ-cp+6"
    assert result.segments[1].citations == []
    assert not result.refused
    assert result.cost_usd == pytest.approx((3000 * 0.10 + 400 * 0.50) / 1e6)


def test_refusal_is_reported():
    result = generation.answer("câu hỏi", ARTICLES, client=FakeClient(response([], "refusal")))
    assert result.refused and result.segments == []
