import importlib.util

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vnsearch.config import PROJECT_ROOT
from vnsearch.generation import Answer, Article, Citation, Segment
from vnsearch.limits import UsageLimiter
from vnsearch.retrievers import Hit, Retriever
from vnsearch.retrievers.hybrid import rrf

spec = importlib.util.spec_from_file_location("demo_server", PROJECT_ROOT / "app" / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

CORPUS = pd.DataFrame(
    {
        "doc_id": ["100/2019/nđ-cp+6", "45/2019/qh14+113", "x+2", "y+1"],
        "title": [
            "Điều 6. Xử phạt xe mô tô",
            "Điều 113. Nghỉ hằng năm",
            "Điều 2. Bãi bỏ",
            "Điều 1.",
        ],
        "text": ["Phạt tiền từ 400.000 đồng.", "Người lao động được nghỉ 12 ngày.", "", "Nội dung"],
    }
).set_index("doc_id")
ARTICLES = [Article(d, CORPUS.at[d, "title"], CORPUS.at[d, "text"]) for d in CORPUS.index[:2]]


class FakeRetriever(Retriever):
    def __init__(self, name: str, doc_ids: list[str]):
        self.name = name
        self.doc_ids = doc_ids

    def search(self, query: str, k: int) -> list[Hit]:
        return [Hit(d, 1.0 / (i + 1)) for i, d in enumerate(self.doc_ids[:k])]


RETRIEVERS = {
    "r0": FakeRetriever("r0", ["x+2", "y+1"]),
    "r1": FakeRetriever("r1", ["100/2019/nđ-cp+6", "x+2"]),
    "r2": FakeRetriever("r2", ["45/2019/qh14+113", "100/2019/nđ-cp+6", "y+1"]),
}


def make_client(client=None, limiter=None, metrics_path=None) -> TestClient:
    app = server.create_app(
        RETRIEVERS,
        CORPUS,
        client,
        limiter=limiter,
        metrics_path=metrics_path or PROJECT_ROOT / "results" / "missing.csv",
    )
    return TestClient(app)


def test_validate_query():
    assert server.validate("  ") is not None
    assert server.validate("x" * 501) is not None
    assert server.validate("Mức phạt?") is None


def test_answer_payload_numbers_and_dedupes_citations():
    cite = Citation("100/2019/nđ-cp+6", "t", "Phạt tiền từ 400.000 đồng")
    result = Answer([Segment("Bị phạt 400.000 đồng.", [cite, cite])], "end_turn", 3000, 400)
    payload = server.answer_payload(result, ARTICLES)
    assert payload["segments"] == [{"text": "Bị phạt 400.000 đồng.", "refs": [1]}]
    assert payload["quotes"] == {"1": ["Phạt tiền từ 400.000 đồng"]}
    assert not payload["refused"] and not payload["not_found"]
    assert "không phải tư vấn pháp lý" in payload["disclaimer"]


def test_answer_payload_refused_and_not_found():
    assert server.answer_payload(Answer([], "refusal", 10, 0), ARTICLES)["refused"]
    empty = server.answer_payload(Answer([], "end_turn", 10, 0), ARTICLES)
    assert empty["not_found"] and not empty["refused"]


def test_search_returns_pipeline_sources_and_all_configs():
    data = make_client().post("/api/search", json={"query": "nghỉ phép năm"}).json()
    assert [s["id"] for s in data["stages"]] == ["r1", "r2", "fusion"]
    assert set(data["compare"]) == {"r0", "r1", "r2", "r3"}
    expected = rrf(
        [RETRIEVERS["r1"].search("", 100), RETRIEVERS["r2"].search("", 100)],
        k=10,
        top=server.SHOWN,
    )
    assert [s["doc_id"] for s in data["sources"]] == [h.doc_id for h in expected]
    first = data["sources"][0]
    assert first["doc_id"] == "100/2019/nđ-cp+6"  # rank 1 in R1, rank 2 in R2
    assert (first["r1_rank"], first["r2_rank"], first["n"]) == (1, 2, 1)
    assert first["law_id"] == "100/2019/NĐ-CP"
    missing = next(s for s in data["sources"] if s["doc_id"] == "y+1")
    assert missing["r1_rank"] is None and missing["r2_rank"] == 3


def test_search_rejects_empty_query_with_message():
    res = make_client().post("/api/search", json={"query": "   "})
    assert res.status_code == 400 and "nhập câu hỏi" in res.json()["detail"]


def test_answer_without_key_is_disabled():
    res = make_client().post("/api/answer", json={"query": "q", "doc_ids": ["x+2"]})
    assert res.status_code == 503


@pytest.mark.parametrize("doc_ids", [[], ["not-in-corpus"], ["x+2"] * 6])
def test_answer_rejects_invalid_sources(doc_ids):
    res = make_client(client=object()).post("/api/answer", json={"query": "q", "doc_ids": doc_ids})
    assert res.status_code == 400


def test_answer_uses_given_sources_and_enforces_session_limit(monkeypatch):
    seen = []

    def fake_answer(query, articles, client=None):
        seen.append([a.doc_id for a in articles])
        cite = Citation(articles[1].doc_id, "t", "nghỉ 12 ngày")
        return Answer([Segment("Được nghỉ 12 ngày.", [cite])], "end_turn", 100, 20)

    monkeypatch.setattr(server.generation, "answer", fake_answer)
    client = make_client(client=object(), limiter=UsageLimiter(daily_limit=10, session_limit=1))
    body = {"query": "nghỉ phép", "doc_ids": ["100/2019/nđ-cp+6", "45/2019/qh14+113"]}
    first = client.post("/api/answer", json=body)
    assert first.status_code == 200
    assert first.json()["segments"][0]["refs"] == [2]
    assert first.json()["session_remaining"] == 0
    assert seen == [body["doc_ids"]]
    second = client.post("/api/answer", json=body)  # same cookie, same session
    assert second.status_code == 429


def test_metrics_reads_csv(tmp_path):
    path = tmp_path / "metrics.csv"
    header = "config,split," + ",".join(server.METRIC_COLUMNS) + ",hardware\n"
    path.write_text(header + "r3,test,0.57,0.91,0.92,0.70,0.75,190.7,2237,cpu\n", encoding="utf-8")
    rows = make_client(metrics_path=path).get("/api/metrics").json()
    assert rows == [
        {
            "config": "r3",
            "label": server.LABELS["r3"],
            "recall@1": 0.57,
            "recall@10": 0.91,
            "hit_rate@10": 0.92,
            "mrr@10": 0.70,
            "ndcg@10": 0.75,
            "p50_ms": 190.7,
            "n_queries": 2237,
        }
    ]
    assert make_client().get("/api/metrics").json() == []


def test_index_and_static_files_are_served_with_security_headers():
    client = make_client()
    res = client.get("/")
    assert res.status_code == 200 and "Hybrid Research" in res.text
    assert "default-src 'self'" in res.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200
    info = client.get("/api/info").json()
    assert info["corpus_size"] == 4 and info["has_key"] is False
