"""Gradio demo: R0–R3 side by side, then a cited answer from Claude Haiku over R3's top-5.

Run locally with `uv run python app/app.py` after building the artifacts
(add `--share` for a temporary public link).
"""

import argparse
import os
import time

import anthropic
import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from vnsearch import data, generation
from vnsearch.config import load_config
from vnsearch.generation import Answer, Article
from vnsearch.limits import UsageLimiter
from vnsearch.retrievers import Hit, Retriever, load_retriever
from vnsearch.retrievers.hybrid import HybridRetriever

SHOWN = 5
SNIPPET_CHARS = 280
QUOTE_CHARS = 220
LABELS = {
    "r0": "R0 · BM25 (âm tiết)",
    "r1": "R1 · BM25 (tách từ)",
    "r2": "R2 · Dense (bge-m3)",
    "r3": "R3 · Hybrid (RRF)",
}
# Hand-written examples (the Zalo questions are not shipped with the demo).
EXAMPLES = [
    "Đi xe máy không đội mũ bảo hiểm bị phạt bao nhiêu tiền?",
    "Người lao động được nghỉ phép năm bao nhiêu ngày?",
    "Hồ sơ đăng ký kết hôn gồm những giấy tờ gì?",
]


def load_retrievers() -> dict[str, Retriever]:
    retrievers = {name: load_retriever(load_config(name)) for name in ("r0", "r1", "r2")}
    cfg = load_config("r3")
    # Reuse the loaded components instead of loading bge-m3 a second time.
    components = [retrievers[name] for name in cfg["components"]]
    retrievers["r3"] = HybridRetriever("r3", components, k=cfg["rrf_k"], depth=cfg["depth"])
    return retrievers


def to_articles(hits: list[Hit], corpus: pd.DataFrame) -> list[Article]:
    return [
        Article(h.doc_id, corpus.at[h.doc_id, "title"], corpus.at[h.doc_id, "text"]) for h in hits
    ]


def shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def format_hits(label: str, articles: list[Article], ms: float) -> str:
    lines = [f"### {label}", f"*{ms:.0f} ms*", ""]
    for i, a in enumerate(articles, start=1):
        lines.append(f"**{i}. {a.title}** · `{generation.law_id(a.doc_id).upper()}`")
        lines.append(f"> {shorten(a.text or a.title, SNIPPET_CHARS)}")
        lines.append("")
    if not articles:
        lines.append("*Không có kết quả.*")
    return "\n".join(lines)


def format_answer(result: Answer, articles: list[Article]) -> str:
    if result.refused:
        return "Mô hình từ chối trả lời câu hỏi này.\n\n*" + generation.DISCLAIMER + "*"
    number = {a.doc_id: i for i, a in enumerate(articles, start=1)}
    body, quotes = [], {}
    for segment in result.segments:
        refs = []
        for c in segment.citations:
            refs.append(number[c.doc_id])
            quotes.setdefault(c.doc_id, []).append(c.cited_text)
        markers = "".join(f" [{n}]" for n in dict.fromkeys(refs))
        body.append(segment.text + markers)
    lines = ["".join(body).strip() or generation.NOT_FOUND, "", "**Nguồn**"]
    for a in articles:
        lines.append(f"{number[a.doc_id]}. {a.title} · `{generation.law_id(a.doc_id).upper()}`")
        for quote in dict.fromkeys(quotes.get(a.doc_id, [])):
            lines.append(f"   > {shorten(quote, QUOTE_CHARS)}")
    lines += [
        "",
        f"*{result.input_tokens:,} input + {result.output_tokens:,} output tokens "
        f"≈ ${result.cost_usd:.4f}*",
        "",
        f"*{generation.DISCLAIMER}*",
    ]
    return "\n".join(lines)


def validate(query: str) -> str | None:
    if not query.strip():
        return "Vui lòng nhập câu hỏi."
    if len(query) > generation.MAX_QUERY_CHARS:
        return f"Câu hỏi tối đa {generation.MAX_QUERY_CHARS} ký tự."
    return None


def build_app(retrievers: dict[str, Retriever], corpus: pd.DataFrame) -> gr.Blocks:
    limiter = UsageLimiter()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    client = anthropic.Anthropic() if has_key else None

    def search(query: str):
        if error := validate(query):
            return [error] + [""] * (len(LABELS) - 1)
        panels = []
        for name, label in LABELS.items():
            start = time.perf_counter()
            hits = retrievers[name].search(query, SHOWN)
            ms = (time.perf_counter() - start) * 1000
            panels.append(format_hits(label, to_articles(hits, corpus), ms))
        return panels

    def generate(query: str, session_count: int):
        if error := validate(query):
            return error, session_count
        if client is None:
            return (
                "Chưa cấu hình `ANTHROPIC_API_KEY`, nên chức năng trả lời đang tắt.",
                session_count,
            )
        if refusal := limiter.try_acquire(session_count):
            return refusal, session_count
        articles = to_articles(retrievers["r3"].search(query, generation.TOP_K), corpus)
        try:
            result = generation.answer(query, articles, client=client)
        except anthropic.RateLimitError:
            return "Dịch vụ đang quá tải, vui lòng thử lại sau ít phút.", session_count + 1
        except anthropic.APIError as e:
            return f"Lỗi khi gọi mô hình ({type(e).__name__}). Vui lòng thử lại.", session_count + 1
        return format_answer(result, articles), session_count + 1

    with gr.Blocks(title="Vietnamese Hybrid Search for RAG") as demo:
        gr.Markdown(
            "# Vietnamese Hybrid Search for RAG\n"
            "Tìm kiếm trên 61.425 điều luật (Zalo AI 2021 Legal). So sánh 4 cấu hình retrieval, "
            "rồi sinh câu trả lời có trích dẫn từ top-5 của R3 bằng Claude Haiku."
        )
        session_count = gr.State(0)
        query = gr.Textbox(label="Câu hỏi", max_length=generation.MAX_QUERY_CHARS, lines=2)
        gr.Examples(EXAMPLES, inputs=query)
        with gr.Row():
            search_btn = gr.Button("Tìm kiếm", variant="primary")
            answer_btn = gr.Button("Trả lời (R3 + Claude Haiku)")
        answer_box = gr.Markdown()
        with gr.Row():
            panels = [gr.Markdown() for _ in LABELS]
        search_btn.click(search, inputs=query, outputs=panels)
        query.submit(search, inputs=query, outputs=panels)
        answer_btn.click(
            generate, inputs=[query, session_count], outputs=[answer_box, session_count]
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio demo for R0–R3 + Claude Haiku answers")
    parser.add_argument(
        "--share",
        action="store_true",
        help="also open a temporary public gradio.live link (72 h); generation caps still apply",
    )
    args, _ = parser.parse_known_args()
    load_dotenv()
    corpus = data.load_corpus().set_index("doc_id")
    demo = build_app(load_retrievers(), corpus)
    demo.queue(default_concurrency_limit=2).launch(share=args.share)


if __name__ == "__main__":
    main()
