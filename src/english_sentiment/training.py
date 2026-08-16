"""Train and evaluate reproducible classical sentiment baselines."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
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
        "linear_svm": CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=1.5), cv=3),
        "multinomial_nb": MultinomialNB(alpha=0.5),
    }


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError("Dataset must contain text and label columns")
    columns = ["text", "label", *[c for c in ("split", "source_id") if c in frame.columns]]
    frame = frame[columns].dropna(subset=["text", "label"]).copy()
    frame["label"] = frame["label"].astype(str).str.lower().str.strip()
    unknown = sorted(set(frame["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}. Expected: {LABELS}")
    frame["text"] = frame["text"].astype(str).map(normalize_english_text)
    return frame[frame["text"].str.len() > 0].drop_duplicates(subset=["text", "label"])


def train_classical_models(data_path: Path, output_dir: Path, report_dir: Path) -> dict:
    frame = load_dataset(data_path)
    if "split" not in frame or not {"train", "validation", "test"}.issubset(set(frame["split"])):
        raise ValueError("Training requires explicit train, validation, and test splits")
    train = frame[frame["split"].isin(["train", "validation"])]
    test = frame[frame["split"] == "test"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results, best = {}, ("", -1.0, None)
    for name, estimator in _models().items():
        pipeline = Pipeline([("features", _features()), ("classifier", estimator)])
        pipeline.fit(train["text"], train["label"])
        predicted = pipeline.predict(test["text"])
        report = classification_report(test["label"], predicted, labels=LABELS,
                                       output_dict=True, zero_division=0)
        metrics = {
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        }
        results[name] = metrics
        (report_dir / f"{name}_classification_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        matrix = confusion_matrix(test["label"], predicted, labels=LABELS)
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS,
                    yticklabels=LABELS)
        plt.title(name.replace("_", " ").title())
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(report_dir / f"{name}_confusion_matrix.png", dpi=180)
        plt.close()
        if metrics["f1_macro"] > best[1]:
            best = (name, metrics["f1_macro"], pipeline)
    summary = {
        "dataset": "TweetEval sentiment",
        "dataset_rows": len(frame), "train_rows": len(train), "test_rows": len(test),
        "class_distribution": frame["label"].value_counts().to_dict(),
        "split_strategy": "official_train_plus_validation_vs_test", "random_state": 42,
        "primary_metric": "f1_macro", "best_model": best[0], "models": results,
    }
    (report_dir / "classical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump(best[2], output_dir / "best_classical_model.joblib")
    return summary
