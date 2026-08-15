"""Persian text preprocessing utilities."""

from __future__ import annotations

import re

_CHAR_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
    }
)
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MENTION = re.compile(r"@[\w_]+")
_WHITESPACE = re.compile(r"\s+")
_REPEATED_CHARS = re.compile(r"(.)\1{2,}")


def normalize_persian_text(text: str) -> str:
    """Return a conservative normalization suitable for Persian user reviews."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.translate(_CHAR_TRANSLATION)
    text = _DIACRITICS.sub("", text)
    text = _URL.sub(" URL ", text)
    text = _MENTION.sub(" USER ", text)
    text = text.replace("\u200c", " ").replace("\u200f", " ").replace("\u200e", " ")
    text = _REPEATED_CHARS.sub(r"\1\1", text)
    return _WHITESPACE.sub(" ", text).strip()

