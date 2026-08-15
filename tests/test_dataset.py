from persian_sentiment.dataset import TWITTER_LABEL_MAP, collapse_aspect_records


def test_collapses_consistent_aspects_to_review_level() -> None:
    records = [
        {"review_id": "1", "review": "عالی بود", "label": "1"},
        {"review_id": "1", "review": "عالی بود", "label": "2"},
    ]
    frame = collapse_aspect_records(records, "train")
    record = frame.to_dict("records")[0]
    assert record["text"] == "عالی بود"
    assert record["label"] == "positive"
    assert record["split"] == "train"
    assert len(record["source_id"]) == 16


def test_drops_conflicting_and_non_sentiment_reviews() -> None:
    records = [
        {"review_id": "1", "review": "ترکیبی", "label": "1"},
        {"review_id": "1", "review": "ترکیبی", "label": "-1"},
        {"review_id": "2", "review": "بدون احساس", "label": "-3"},
    ]
    assert collapse_aspect_records(records, "test").empty


def test_twitter_emotion_to_sentiment_mapping() -> None:
    assert TWITTER_LABEL_MAP == {0: "positive", 1: "negative", 2: "negative", 3: "neutral"}
