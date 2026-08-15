import pytest

from persian_sentiment.preprocessing import normalize_persian_text


def test_normalizes_arabic_character_variants() -> None:
    assert normalize_persian_text("كتاب زيبا") == "کتاب زیبا"


def test_normalizes_social_text() -> None:
    assert normalize_persian_text("عاااالی  https://example.com  @someone") == "عاالی URL USER"


def test_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        normalize_persian_text(None)  # type: ignore[arg-type]

