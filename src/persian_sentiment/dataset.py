"""Download and prepare the licensed ParsiNLU sentiment dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_URL = "https://raw.githubusercontent.com/persiannlp/parsinlu/master/data/sentiment-analysis"
SOURCES = {
    "train": "ABSA_Dataset_train.jsonl",
    "validation_food": "food_dev.jsonl",
    "validation_movies": "movie_dev.jsonl",
    "test_food": "food_test.jsonl",
    "test_movies": "movie_test.jsonl",
}
LABEL_MAP = {
    "-2": "negative",
    "-1": "negative",
    "0": "neutral",
    "1": "positive",
    "2": "positive",
}
TWITTER_URL = (
    "https://huggingface.co/datasets/moali-mkh-2000/"
    "PersianTwitterDataset-SentimentAnalysis/resolve/main/PersianTwitterDataset.csv"
)
TWITTER_LABEL_MAP = {
    0: "positive",  # Happy
    1: "negative",  # Sad
    2: "negative",  # Angry
    3: "neutral",
}


def _download_jsonl(url: str) -> list[dict[str, str]]:
    with urlopen(url, timeout=60) as response:
        return [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line]


def collapse_aspect_records(records: list[dict[str, str]], split: str) -> pd.DataFrame:
    """Collapse aspect labels into unambiguous review-level sentiment labels.

    Reviews whose aspects map to conflicting sentiments are excluded. Labels -3
    (no sentiment) and 3 (mixed) are also excluded from three-class classification.
    """
    grouped: dict[str, dict[str, object]] = defaultdict(lambda: {"labels": set()})
    for row in records:
        mapped = LABEL_MAP.get(str(row.get("label", "")))
        if mapped is None:
            continue
        review_id = str(row["review_id"])
        grouped[review_id]["text"] = row["review"]
        labels = grouped[review_id]["labels"]
        assert isinstance(labels, set)
        labels.add(mapped)

    output = []
    for review_id, item in grouped.items():
        labels = item["labels"]
        assert isinstance(labels, set)
        if len(labels) == 1:
            output.append(
                {
                    "text": item["text"],
                    "label": next(iter(labels)),
                    "source_id": sha256(str(item["text"]).encode("utf-8")).hexdigest()[:16],
                    "split": split,
                }
            )
    return pd.DataFrame(output, columns=["text", "label", "source_id", "split"])


def prepare_parsinlu(output_path: Path) -> pd.DataFrame:
    """Download, aggregate, validate, and save ParsiNLU as a three-class CSV."""
    frames = []
    for source_split, filename in SOURCES.items():
        target_split = source_split.split("_", maxsplit=1)[0]
        records = _download_jsonl(f"{BASE_URL}/{filename}")
        frames.append(collapse_aspect_records(records, target_split))

    frame = pd.concat(frames, ignore_index=True)
    if set(frame["label"]) != {"negative", "neutral", "positive"}:
        raise ValueError("Prepared dataset does not contain all three sentiment classes")
    if frame.groupby("source_id")["split"].nunique().max() != 1:
        raise ValueError("A review appears in more than one official split")
    frame = frame.drop_duplicates(subset=["text", "label"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    return frame


def prepare_persian_twitter(output_path: Path, random_state: int = 42) -> pd.DataFrame:
    """Download and map the ODbL Persian Twitter emotion data to three sentiments."""
    raw = pd.read_csv(TWITTER_URL)
    frame = raw[["Tweets", "Numeric Labels"]].rename(
        columns={"Tweets": "text", "Numeric Labels": "original_label"}
    )
    frame["label"] = frame["original_label"].map(TWITTER_LABEL_MAP)
    frame = frame.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"]).copy()
    frame["source_id"] = frame["text"].map(
        lambda text: sha256(str(text).encode("utf-8")).hexdigest()[:16]
    )

    train, remainder = train_test_split(
        frame, test_size=0.30, random_state=random_state, stratify=frame["label"]
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_state,
        stratify=remainder["label"],
    )
    train = train.assign(split="train")
    validation = validation.assign(split="validation")
    test = test.assign(split="test")
    prepared = pd.concat([train, validation, test], ignore_index=True)
    prepared = prepared[["text", "label", "source_id", "split"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False, encoding="utf-8")
    return prepared
