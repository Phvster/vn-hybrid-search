"""Grounded answer generation with Claude Haiku and the Citations feature.

Each retrieved article is sent as its own document block with citations enabled, so the
answer comes back with machine-checkable references (document index + quoted text)
instead of citations the model writes free-hand.
"""

from dataclasses import dataclass, field

import anthropic

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 2048
EFFORT = "low"  # grounded Q&A over given text needs little reasoning
TOP_K = 5  # articles passed to the model (R3 dev Recall@5 = 0.862)
MAX_QUERY_CHARS = 500
MAX_ARTICLE_CHARS = 6000  # some articles exceed 60k tokens; cap cost per request
TRUNCATION_NOTE = "\n[... nội dung đã được rút gọn ...]"
# USD per million tokens (prompts up to 100K tokens).
INPUT_PRICE_PER_MTOK = 0.10
OUTPUT_PRICE_PER_MTOK = 0.50

DISCLAIMER = (
    "Thông tin chỉ mang tính tham khảo, dựa trên văn bản pháp luật năm 2021 và có thể đã "
    "được sửa đổi. Đây không phải tư vấn pháp lý."
)
NOT_FOUND = "Không tìm thấy quy định phù hợp trong các điều luật được cung cấp."
SYSTEM_PROMPT = f"""Bạn là trợ lý tra cứu văn bản pháp luật Việt Nam.

Quy tắc:
- Chỉ trả lời dựa trên các điều luật được cung cấp trong tài liệu.
  Không dùng kiến thức bên ngoài.
- Nếu các điều luật không chứa câu trả lời, hãy nói rõ: "{NOT_FOUND}" và không suy đoán.
- Trả lời ngắn gọn, chính xác, bằng tiếng Việt.
  Nêu rõ mức phạt, điều kiện, thời hạn đúng như văn bản.
- Không đưa ra tư vấn pháp lý cho trường hợp cá nhân."""


@dataclass(frozen=True)
class Article:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Citation:
    doc_id: str
    document_title: str
    cited_text: str


@dataclass
class Segment:
    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Answer:
    segments: list[Segment]
    stop_reason: str
    input_tokens: int
    output_tokens: int

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * INPUT_PRICE_PER_MTOK + self.output_tokens * OUTPUT_PRICE_PER_MTOK
        ) / 1e6


def law_id(doc_id: str) -> str:
    return doc_id.rsplit("+", 1)[0]


def document_block(article: Article) -> dict:
    body = article.text or article.title
    if len(body) > MAX_ARTICLE_CHARS:
        body = body[:MAX_ARTICLE_CHARS] + TRUNCATION_NOTE
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": body},
        "title": f"{article.title} ({law_id(article.doc_id).upper()})",
        "citations": {"enabled": True},
    }


def build_request(query: str, articles: list[Article]) -> dict:
    if not query.strip():
        raise ValueError("Empty question")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"Question longer than {MAX_QUERY_CHARS} characters")
    content = [document_block(a) for a in articles]
    content.append({"type": "text", "text": f"Câu hỏi: {query}"})
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "output_config": {"effort": EFFORT},
        "messages": [{"role": "user", "content": content}],
    }


def parse_response(response, articles: list[Article]) -> Answer:
    segments = []
    for block in response.content:
        if block.type != "text":
            continue  # thinking blocks carry no answer text
        citations = [
            Citation(
                doc_id=articles[c.document_index].doc_id,
                document_title=c.document_title or "",
                cited_text=c.cited_text,
            )
            for c in (getattr(block, "citations", None) or [])
            if getattr(c, "document_index", None) is not None
        ]
        segments.append(Segment(block.text, citations))
    return Answer(
        segments=segments,
        stop_reason=response.stop_reason,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def answer(
    query: str, articles: list[Article], client: anthropic.Anthropic | None = None
) -> Answer:
    """Answer `query` from `articles` only. Raises anthropic.APIError on API failures."""
    client = client or anthropic.Anthropic()
    response = client.messages.create(**build_request(query, articles))
    return parse_response(response, articles)
