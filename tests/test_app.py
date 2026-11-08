import importlib.util

from vnsearch.config import PROJECT_ROOT
from vnsearch.generation import Answer, Article, Citation, Segment

spec = importlib.util.spec_from_file_location("demo_app", PROJECT_ROOT / "app" / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

ARTICLES = [
    Article("100/2019/nđ-cp+6", "Điều 6. Xử phạt xe mô tô", "Phạt tiền từ 400.000 đồng."),
    Article("x+2", "Điều 2. Bãi bỏ", ""),
]


def test_answer_has_numbered_markers_and_sources():
    cite = Citation("100/2019/nđ-cp+6", "t", "Phạt tiền từ 400.000 đồng")
    result = Answer([Segment("Bị phạt 400.000 đồng.", [cite, cite])], "end_turn", 3000, 400)
    text = app.format_answer(result, ARTICLES)
    assert "Bị phạt 400.000 đồng. [1]" in text and "[1] [1]" not in text
    assert "1. Điều 6. Xử phạt xe mô tô · `100/2019/NĐ-CP`" in text
    assert "> Phạt tiền từ 400.000 đồng" in text
    assert "không phải tư vấn pháp lý" in text


def test_refused_and_empty_answers():
    assert "từ chối" in app.format_answer(Answer([], "refusal", 10, 0), ARTICLES)
    assert app.generation.NOT_FOUND in app.format_answer(Answer([], "end_turn", 10, 0), ARTICLES)


def test_validate_query():
    assert app.validate("  ") is not None
    assert app.validate("x" * 501) is not None
    assert app.validate("Mức phạt?") is None


def test_hits_panel_lists_titles_and_handles_empty():
    assert "**1. Điều 6. Xử phạt xe mô tô**" in app.format_hits("R1", ARTICLES, 1.2)
    assert "Không có kết quả" in app.format_hits("R1", [], 1.0)
