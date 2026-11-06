import unicodedata

from vnsearch.text import normalize


def test_nfd_and_nfc_forms_become_equal():
    composed = "Luật hòa giải"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    assert normalize(composed) == normalize(decomposed) == composed


def test_case_is_preserved():
    assert normalize("Điều 5. Đối tượng") == "Điều 5. Đối tượng"


def test_whitespace_is_tidied_but_line_breaks_kept():
    raw = "  1. Khoản   một \r\n\r\n   2. Khoản\thai  "
    assert normalize(raw) == "1. Khoản một\n2. Khoản hai"
