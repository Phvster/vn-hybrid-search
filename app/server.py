"""Hybrid Research demo: a FastAPI backend for the static web UI in app/static/.

Endpoints
- GET  /api/info     corpus size, examples, limits, whether answer generation is enabled
- POST /api/search   R0–R3 top-5 plus the R3 pipeline (per-stage timings, R1/R2 ranks per source)
- POST /api/answer   cited answer from Claude Haiku over the given R3 sources
- GET  /api/metrics  test-split benchmark from results/metrics_test.csv

Run locally with `uv run python app/server.py` after building the artifacts.
"""

import argparse
import csv
import os
import threading
import time
import uuid
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vnsearch import data, generation
from vnsearch.config import PROJECT_ROOT, load_config
from vnsearch.generation import Answer, Article
from vnsearch.limits import UsageLimiter
from vnsearch.retrievers import Hit, Retriever, load_retriever
from vnsearch.retrievers.hybrid import rrf

STATIC_DIR = Path(__file__).resolve().parent / "static"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics_test.csv"
SHOWN = 5
SNIPPET_CHARS = 280
FULL_TEXT_CHARS = 6000  # what the UI shows when a source card is expanded
SESSION_COOKIE = "hr_sid"
LABELS = {
    "r0": "BM25 · âm tiết",
    "r1": "BM25 · tách từ",
    "r2": "Dense · bge-m3",
    "r3": "Hybrid · RRF",
    "r4": "Hybrid + Reranker",
}
# Hand-written examples (the Zalo questions are not shipped with the demo).
EXAMPLES = [
    {"topic": "Giao thông", "query": "Đi xe máy không đội mũ bảo hiểm bị phạt bao nhiêu tiền?"},
    {"topic": "Lao động", "query": "Người lao động được nghỉ phép năm bao nhiêu ngày?"},
    {"topic": "Hộ tịch", "query": "Hồ sơ đăng ký kết hôn gồm những giấy tờ gì?"},
    {"topic": "Đất đai", "query": "Thời hạn sử dụng đất nông nghiệp là bao lâu?"},
]
METRIC_COLUMNS = [
    "recall@1",
    "recall@10",
    "hit_rate@10",
    "mrr@10",
    "ndcg@10",
    "p50_ms",
    "n_queries",
]


class SearchRequest(BaseModel):
    query: str


class AnswerRequest(BaseModel):
    query: str
    doc_ids: list[str]


def validate(query: str) -> str | None:
    if not query.strip():
        return "Vui lòng nhập câu hỏi."
    if len(query) > generation.MAX_QUERY_CHARS:
        return f"Câu hỏi tối đa {generation.MAX_QUERY_CHARS} ký tự."
    return None


def shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def answer_payload(result: Answer, articles: list[Article]) -> dict:
    """Turn an Answer into numbered citation markers that point at the source cards."""
    number = {a.doc_id: i for i, a in enumerate(articles, start=1)}
    paragraphs, quotes = [], {}
    for segment in result.segments:
        refs = []
        for c in segment.citations:
            n = number[c.doc_id]
            refs.append(n)
            quotes.setdefault(n, [])
            if c.cited_text not in quotes[n]:
                quotes[n].append(c.cited_text)
        paragraphs.append({"text": segment.text, "refs": list(dict.fromkeys(refs))})
    text = "".join(p["text"] for p in paragraphs).strip()
    return {
        "refused": result.refused,
        "not_found": not result.refused and (not text or text == generation.NOT_FOUND),
        "segments": paragraphs,
        "quotes": {str(n): q for n, q in quotes.items()},
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "disclaimer": generation.DISCLAIMER,
    }


def read_metrics(path: Path = METRICS_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        values = {c: float(row[c]) for c in METRIC_COLUMNS}
        values["n_queries"] = int(values["n_queries"])
        out.append({"config": row["config"], "label": LABELS.get(row["config"], ""), **values})
    return out


def load_retrievers() -> dict[str, Retriever]:
    return {name: load_retriever(load_config(name)) for name in ("r0", "r1", "r2")}


def create_app(
    retrievers: dict[str, Retriever],
    corpus: pd.DataFrame,
    client: anthropic.Anthropic | None,
    limiter: UsageLimiter | None = None,
    metrics_path: Path = METRICS_PATH,
) -> FastAPI:
    """`retrievers` holds r0, r1 and r2; R3 is fused here so each stage can be timed."""
    cfg = load_config("r3")
    rrf_k, depth = cfg["rrf_k"], cfg["depth"]
    limiter = limiter or UsageLimiter()
    sessions: dict[str, int] = {}
    retrieval_lock = threading.Lock()  # one query at a time: the encoder is not shared safely
    app = FastAPI(title="Hybrid Research", docs_url=None, redoc_url=None)

    def article(doc_id: str) -> Article:
        return Article(doc_id, corpus.at[doc_id, "title"], corpus.at[doc_id, "text"])

    def card(doc_id: str) -> dict:
        a = article(doc_id)
        return {
            "doc_id": doc_id,
            "law_id": generation.law_id(doc_id).upper(),
            "title": a.title,
            "snippet": shorten(a.text or a.title, SNIPPET_CHARS),
        }

    def timed(fn, *args) -> tuple[list[Hit], float]:
        start = time.perf_counter()
        hits = fn(*args)
        return hits, (time.perf_counter() - start) * 1000

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/info")
    def info() -> dict:
        return {
            "corpus_size": len(corpus),
            "has_key": client is not None,
            "examples": EXAMPLES,
            "max_query_chars": generation.MAX_QUERY_CHARS,
            "session_limit": limiter.session_limit,
            "model": generation.MODEL,
            "labels": LABELS,
        }

    @app.post("/api/search")
    def search(req: SearchRequest) -> dict:
        if error := validate(req.query):
            raise HTTPException(400, error)
        with retrieval_lock:
            r0, r0_ms = timed(retrievers["r0"].search, req.query, SHOWN)
            r1, r1_ms = timed(retrievers["r1"].search, req.query, depth)
            r2, r2_ms = timed(retrievers["r2"].search, req.query, depth)
            r3, fusion_ms = timed(rrf, [r1, r2], rrf_k, SHOWN)
        r1_rank = {h.doc_id: i for i, h in enumerate(r1, start=1)}
        r2_rank = {h.doc_id: i for i, h in enumerate(r2, start=1)}
        sources = []
        for n, hit in enumerate(r3, start=1):
            a = article(hit.doc_id)
            body = a.text or a.title
            sources.append(
                card(hit.doc_id)
                | {
                    "n": n,
                    "text": body[:FULL_TEXT_CHARS],
                    "truncated": len(body) > FULL_TEXT_CHARS,
                    "rrf": hit.score,
                    "r1_rank": r1_rank.get(hit.doc_id),
                    "r2_rank": r2_rank.get(hit.doc_id),
                }
            )
        compare = {
            "r0": (r0, r0_ms),
            "r1": (r1[:SHOWN], r1_ms),
            "r2": (r2[:SHOWN], r2_ms),
            "r3": (r3, r1_ms + r2_ms + fusion_ms),
        }
        return {
            "query": req.query,
            "stages": [
                {"id": "r1", "label": "BM25 · tách từ (R1)", "ms": r1_ms},
                {"id": "r2", "label": "Dense · bge-m3 (R2)", "ms": r2_ms},
                {"id": "fusion", "label": f"Hợp nhất RRF, k={rrf_k} (R3)", "ms": fusion_ms},
            ],
            "depth": depth,
            "sources": sources,
            "compare": {
                name: {"label": LABELS[name], "ms": ms, "hits": [card(h.doc_id) for h in hits]}
                for name, (hits, ms) in compare.items()
            },
        }

    @app.post("/api/answer")
    def answer(req: AnswerRequest, request: Request, response: Response) -> dict:
        if error := validate(req.query):
            raise HTTPException(400, error)
        if not 1 <= len(req.doc_ids) <= generation.TOP_K or any(
            d not in corpus.index for d in req.doc_ids
        ):
            raise HTTPException(400, "Danh sách nguồn không hợp lệ.")
        if client is None:
            raise HTTPException(
                503, "Chưa cấu hình ANTHROPIC_API_KEY nên chức năng trả lời đang tắt."
            )
        sid = request.cookies.get(SESSION_COOKIE) or uuid.uuid4().hex
        response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
        if refusal := limiter.try_acquire(sessions.get(sid, 0)):
            raise HTTPException(429, refusal)
        sessions[sid] = sessions.get(sid, 0) + 1
        articles = [article(d) for d in req.doc_ids]
        start = time.perf_counter()
        try:
            result = generation.answer(req.query, articles, client=client)
        except anthropic.RateLimitError as e:
            raise HTTPException(503, "Dịch vụ đang quá tải, vui lòng thử lại sau ít phút.") from e
        except anthropic.APIError as e:
            raise HTTPException(502, f"Lỗi khi gọi mô hình ({type(e).__name__}).") from e
        payload = answer_payload(result, articles)
        payload["ms"] = (time.perf_counter() - start) * 1000
        payload["session_remaining"] = limiter.session_limit - sessions[sid]
        return payload

    @app.get("/api/metrics")
    def metrics() -> list[dict]:
        return read_metrics(metrics_path)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Hybrid Research demo server")
    parser.add_argument("--host", default=os.environ.get("DEMO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DEMO_PORT", "7860")))
    args, _ = parser.parse_known_args()
    load_dotenv(PROJECT_ROOT / ".env")
    print("loading corpus and retrievers (about a minute)...", flush=True)
    corpus = data.load_corpus().set_index("doc_id")
    client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
    app = create_app(load_retrievers(), corpus, client)
    print(f"Hybrid Research: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
