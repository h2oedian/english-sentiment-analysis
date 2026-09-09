"""Download and validate the official TweetEval sentiment benchmark."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DATASET_REVISION = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
BASE_URL = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/{DATASET_REVISION}/datasets/sentiment"
SPLITS = ("train", "validation", "test")
FILE_STEMS = {"train": "train", "validation": "val", "test": "test"}
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def _download_lines(url: str) -> list[str]:
    with urlopen(url, timeout=90) as response:
        return response.read().decode("utf-8").splitlines()


def build_tweeteval_split(texts: list[str], labels: list[str], split: str) -> pd.DataFrame:
    """Create one validated split from parallel TweetEval files."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if len(texts) != len(labels):
        raise ValueError(f"Mismatched text and label counts for {split}")
    mapped = []
    for value in labels:
        try:
            mapped.append(LABEL_MAP[int(value)])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Unknown TweetEval label: {value}") from error
    frame = pd.DataFrame({"text": texts, "label": mapped})
    frame["source_id"] = frame["text"].map(
        lambda value: sha256(value.encode("utf-8")).hexdigest()[:16]
    )
    frame["split"] = split
    return frame


def prepare_tweeteval(output_path: Path) -> pd.DataFrame:
    """Download official train/validation/test files and save one UTF-8 CSV."""
    frames = []
    for split in SPLITS:
        stem = FILE_STEMS[split]
        texts = _download_lines(f"{BASE_URL}/{stem}_text.txt")
        labels = _download_lines(f"{BASE_URL}/{stem}_labels.txt")
        frames.append(build_tweeteval_split(texts, labels, split))
    frame = pd.concat(frames, ignore_index=True)
    if set(frame["label"]) != set(LABEL_MAP.values()):
        raise ValueError("TweetEval data is missing one or more sentiment classes")
    if frame.groupby("source_id")["split"].nunique().max() != 1:
        raise ValueError("Data leakage detected: a text occurs in multiple official splits")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
    from .experiment import digest, save_json
    save_json(output_path.with_suffix(".manifest.json"), {"revision": DATASET_REVISION,
              "sha256": digest(output_path), "rows": len(frame)})
    return frame
