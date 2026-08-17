import pytest

from english_sentiment.dataset import LABEL_MAP, build_tweeteval_split


def test_builds_official_split() -> None:
    frame = build_tweeteval_split(["bad", "fine", "great"], ["0", "1", "2"], "train")
    assert frame["label"].tolist() == ["negative", "neutral", "positive"]
    assert set(frame["split"]) == {"train"}
    assert all(frame["source_id"].str.len() == 16)


def test_rejects_mismatched_parallel_files() -> None:
    with pytest.raises(ValueError, match="Mismatched"):
        build_tweeteval_split(["one"], ["0", "1"], "test")


def test_tweeteval_label_contract() -> None:
    assert LABEL_MAP == {0: "negative", 1: "neutral", 2: "positive"}
