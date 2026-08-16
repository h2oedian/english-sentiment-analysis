"""Training and evaluation for classical sentiment baselines."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .constants import LABELS
from .preprocessing import normalize_persian_text


def _features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60_000)),
            (
                "char",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40_000),
            ),
        ]
    )


def _models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2_000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced"),
        "multinomial_nb": MultinomialNB(alpha=0.5),
    }


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    optional = [column for column in ("split", "source_id") if column in frame.columns]
    frame = frame[["text", "label", *optional]].dropna(subset=["text", "label"]).copy()
    frame = frame.drop_duplicates(subset=["text", "label"])
    frame["label"] = frame["label"].astype(str).str.lower().str.strip()
    unknown = sorted(set(frame["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}. Expected: {LABELS}")
    if frame["label"].value_counts().min() < 2:
        raise ValueError("Each class needs at least two examples for a stratified split")
    frame["text"] = frame["text"].astype(str).map(normalize_persian_text)
    return frame[frame["text"].str.len() > 0]


def train_classical_models(
    data_path: Path,
    output_dir: Path,
    report_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    frame = load_dataset(data_path)
    if "split" in frame.columns and set(frame["split"]) >= {"train", "test"}:
        train_frame = frame[frame["split"].isin(["train", "validation"])]
        test_frame = frame[frame["split"] == "test"]
        x_train, y_train = train_frame["text"], train_frame["label"]
        x_test, y_test = test_frame["text"], test_frame["label"]
        split_strategy = "official_train_validation_vs_test"
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            frame["text"],
            frame["label"],
            test_size=test_size,
            random_state=random_state,
            stratify=frame["label"],
        )
        split_strategy = "stratified_random"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    best_name, best_score, best_pipeline = "", -1.0, None

    for name, estimator in _models().items():
        pipeline = Pipeline([("features", _features()), ("classifier", estimator)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        report = classification_report(
            y_test, predictions, labels=LABELS, output_dict=True, zero_division=0
        )
        results[name] = {
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        }
        with (report_dir / f"{name}_classification_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        matrix = confusion_matrix(y_test, predictions, labels=LABELS)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
        plt.title(name.replace("_", " ").title())
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(report_dir / f"{name}_confusion_matrix.png", dpi=160)
        plt.close()

        if results[name]["f1_macro"] > best_score:
            best_name, best_score, best_pipeline = name, results[name]["f1_macro"], pipeline

    summary = {
        "dataset_rows": len(frame),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "class_distribution": frame["label"].value_counts().to_dict(),
        "split_strategy": split_strategy,
        "test_fraction": len(x_test) / len(frame),
        "requested_test_size": test_size if split_strategy == "stratified_random" else None,
        "random_state": random_state,
        "best_model": best_name,
        "models": results,
    }
    with (report_dir / "classical_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    joblib.dump(best_pipeline, output_dir / "best_classical_model.joblib")
    return results
