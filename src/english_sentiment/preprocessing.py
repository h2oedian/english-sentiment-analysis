"""Conservative normalization for English social-media text."""

from __future__ import annotations

import html
import re
import unicodedata

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
_WHITESPACE = re.compile(r"\s+")
_REPEATED = re.compile(r"([A-Za-z])\1{2,}", re.IGNORECASE)

_CONTRACTIONS = (
    (re.compile(r"\b(can)['’]t\b", re.IGNORECASE), r"\1 not"),
    (re.compile(r"\b(won)['’]t\b", re.IGNORECASE), "will not"),
    (re.compile(r"\b(ain)['’]t\b", re.IGNORECASE), "is not"),
    (re.compile(r"\b(\w+)n['’]t\b", re.IGNORECASE), r"\1 not"),
    (re.compile(r"\b(\w+)['’]re\b", re.IGNORECASE), r"\1 are"),
    (re.compile(r"\b(\w+)['’]ve\b", re.IGNORECASE), r"\1 have"),
    (re.compile(r"\b(\w+)['’]ll\b", re.IGNORECASE), r"\1 will"),
    (re.compile(r"\b(\w+)['’]m\b", re.IGNORECASE), r"\1 am"),
)


def normalize_english_text(text: str) -> str:
    """Normalize noise while preserving sentiment-bearing punctuation and negation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKC", html.unescape(text))
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    text = _URL.sub(" URL ", text)
    text = _MENTION.sub(" USER ", text)
    text = _REPEATED.sub(r"\1\1", text)
    return _WHITESPACE.sub(" ", text).strip()
