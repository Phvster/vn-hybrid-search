"""Text normalization shared by the corpus, the queries and every retriever."""

import re
import unicodedata

_SPACES = re.compile(r"[^\S\n]+")  # whitespace except newline
_NEWLINES = re.compile(r"\s*\n\s*")


def normalize(text: str) -> str:
    """Apply Unicode NFC and tidy whitespace. Case and line breaks are preserved.

    NFC matters for Vietnamese: the same letter (e.g. "ò") can be one code point or a base
    letter plus a combining mark, and those compare unequal unless normalized.
    """
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = _NEWLINES.sub("\n", text)
    return text.strip()
