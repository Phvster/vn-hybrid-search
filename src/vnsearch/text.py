"""Text normalization shared by the corpus, the queries and every retriever."""

import re
import unicodedata

from pyvi import ViTokenizer

_SPACES = re.compile(r"[^\S\n]+")  # whitespace except newline
_NEWLINES = re.compile(r"\s*\n\s*")
_TOKEN = re.compile(r"\w+")  # letters, digits and "_" (pyvi's compound joiner); drops punctuation


def normalize(text: str) -> str:
    """Apply Unicode NFC and tidy whitespace. Case and line breaks are preserved.

    NFC matters for Vietnamese: the same letter (e.g. "ò") can be one code point or a base
    letter plus a combining mark, and those compare unequal unless normalized.
    """
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = _NEWLINES.sub("\n", text)
    return text.strip()


def syllable_tokens(text: str) -> list[str]:
    """R0: lowercase syllables, punctuation removed."""
    return _TOKEN.findall(normalize(text).lower())


def word_tokens(text: str) -> list[str]:
    """R1: pyvi word segmentation on cased text, then lowercase (see EDA report)."""
    return _TOKEN.findall(ViTokenizer.tokenize(normalize(text)).lower())


TOKENIZERS = {"syllable": syllable_tokens, "word": word_tokens}
