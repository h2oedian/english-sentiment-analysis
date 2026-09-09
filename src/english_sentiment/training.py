"""Train and evaluate reproducible classical sentiment baselines."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .constants import LABELS
from .preprocessing import normalize_english_text


def _features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80_000,
                                 sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=50_000, sublinear_tf=True)),
    ])


def _models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2_000, class_weight="balanced", C=3),
        "linear_svm": CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=1.5, random_state=42), cv=3),
        "multinomial_nb": MultinomialNB(alpha=0.5),
    }


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if not {"text", "label", "split"}.issubset(frame.columns):
        raise ValueError("Dataset requires text, label, and explicit split columns")
    if set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("Expected exactly train, validation, test splits")
    if not set(frame.label).issubset(LABELS):
        raise ValueError("Unknown labels")
    frame["text"] = frame.text.map(normalize_english_text)
    if (frame.text.str.strip().str.len() == 0).any():
        raise ValueError("Blank text")
    keys = frame.text.str.casefold()
    if frame.assign(key=keys).groupby("key").split.nunique().max() > 1:
        raise ValueError("Data leakage: normalized text occurs across splits")
    if "source_id" in frame and frame.groupby("source_id").split.nunique().max() > 1:
        raise ValueError("Data leakage: source_id occurs across splits")
    for split in ("train", "validation", "test"):
        if set(frame.loc[frame.split == split, "label"]) != set(LABELS):
            raise ValueError(f"Missing classes in {split}")
    return frame


def train_classical_models(data_path: Path, output_dir: Path, report_dir: Path) -> dict:
    from sklearn.dummy import DummyClassifier
    from .experiment import analyze, register
    if (report_dir / "selection.json").exists():
        raise ValueError("Selection is frozen")
    frame = load_dataset(data_path)
    train = frame[frame.split == "train"].drop_duplicates(subset=["text", "label"])
    validation = frame[frame.split == "validation"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    models = {**_models(), "majority": DummyClassifier(strategy="prior")}
    for name, estimator in models.items():
        pipeline = Pipeline([("features", _features()), ("classifier", estimator)])
        pipeline.fit(train.text, train.label)
        metrics = analyze(validation, pipeline.predict(validation.text), report_dir,
                          f"validation_{name}")
        artifact = output_dir / f"{name}.joblib"
        joblib.dump(pipeline, artifact)
        register(report_dir, name, artifact, "classical", metrics, data_path)
        results[name] = metrics
    best = max(sorted(results), key=lambda name: results[name]["f1_macro"])
    summary = {"evaluation_split": "validation", "train_rows": len(train),
               "validation_rows": len(validation), "best_model": best, "models": results}
    (report_dir / "classical_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
