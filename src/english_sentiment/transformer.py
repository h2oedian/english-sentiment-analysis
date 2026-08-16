"""Evaluate a production-ready English RoBERTa sentiment model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

from .constants import ID_TO_LABEL, LABELS
from .training import load_dataset

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def compute_transformer_metrics(prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = prediction
    predicted = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        labels, predicted, average="weighted", zero_division=0)[2]
    return {"precision_macro": float(precision), "recall_macro": float(recall),
            "f1_macro": float(f1), "f1_weighted": float(weighted)}


def write_model_comparison(report_dir: Path) -> pd.DataFrame:
    classical_path, transformer_path = (
        report_dir / "classical_summary.json", report_dir / "transformer_summary.json")
    if not classical_path.exists() or not transformer_path.exists():
        return pd.DataFrame()
    classical = json.loads(classical_path.read_text(encoding="utf-8"))
    transformer = json.loads(transformer_path.read_text(encoding="utf-8"))
    rows = [{"model": name, **metrics} for name, metrics in classical["models"].items()]
    rows.append({"model": "twitter_roberta", **transformer["metrics"]})
    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    import matplotlib.pyplot as plt
    import seaborn as sns
    chart = comparison.melt(id_vars="model", value_vars=["precision_macro", "recall_macro",
        "f1_macro"], var_name="metric", value_name="score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=chart, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("English Sentiment Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(report_dir / "model_comparison.png", dpi=180)
    plt.close()
    return comparison


def evaluate_transformer(data_path: Path, output_dir: Path, report_dir: Path,
                         model_name: str = DEFAULT_MODEL, batch_size: int = 32,
                         max_length: int = 128) -> dict[str, float]:
    """Evaluate an English sentiment transformer on the official TweetEval test split."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError('Install dependencies with: pip install -e ".[transformer]"') from error
    frame = load_dataset(data_path)
    test = frame[frame["split"] == "test"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_logits = []
    for start in range(0, len(test), batch_size):
        encoded = tokenizer(test["text"].iloc[start:start + batch_size].tolist(), padding=True,
                            truncation=True, max_length=max_length, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
        all_logits.append(logits.cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = test["label"].map({label: idx for idx, label in ID_TO_LABEL.items()}).to_numpy()
    metrics = compute_transformer_metrics((logits, labels))
    predicted = np.argmax(logits, axis=-1)
    report = classification_report(labels, predicted, labels=list(ID_TO_LABEL),
        target_names=LABELS, output_dict=True, zero_division=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "transformer_model")
    tokenizer.save_pretrained(output_dir / "transformer_model")
    summary = {"model": model_name, "evaluation_dataset": "TweetEval sentiment test",
               "test_rows": len(test), "batch_size": batch_size, "max_length": max_length,
               "device": str(device), "metrics": metrics}
    (report_dir / "transformer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report_dir / "transformer_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    write_model_comparison(report_dir)
    return metrics
