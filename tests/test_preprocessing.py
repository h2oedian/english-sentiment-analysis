import pytest

from english_sentiment.preprocessing import normalize_english_text


@pytest.mark.parametrize(("raw", "expected"), [
    ("I don't like it", "I do not like it"),
    ("She isn't happy", "She is not happy"),
    ("We won't buy it", "We will not buy it"),
    ("I can’t recommend it", "I can not recommend it"),
])
def test_expands_sentiment_critical_negation(raw: str, expected: str) -> None:
    assert normalize_english_text(raw) == expected


def test_normalizes_social_text() -> None:
    assert normalize_english_text("Sooo good! https://example.com @friend") == "Soo good! URL USER"


def test_decodes_html_entities() -> None:
    assert normalize_english_text("good &amp; useful") == "good & useful"


def test_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        normalize_english_text(None)  # type: ignore[arg-type]
