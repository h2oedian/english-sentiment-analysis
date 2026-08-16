"""Fine-tune and evaluate a Persian transformer sentiment classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

from .constants import LABELS
from .training import load_dataset

DEFAULT_MODEL = "HooshvareLab/distilbert-fa-zwnj-base"
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def compute_transformer_metrics(eval_prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Calculate imbalance-aware metrics expected by Hugging Face Trainer."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    return {
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "f1_weighted": float(weighted_f1),
    }


def write_model_comparison(report_dir: Path) -> pd.DataFrame:
    """Combine classical and transformer summaries into a table and chart."""
    classical_path = report_dir / "classical_summary.json"
    transformer_path = report_dir / "transformer_summary.json"
    if not classical_path.exists() or not transformer_path.exists():
        return pd.DataFrame()

    with classical_path.open(encoding="utf-8") as file:
        classical = json.load(file)
    with transformer_path.open(encoding="utf-8") as file:
        transformer = json.load(file)
    rows = [{"model": name, **metrics} for name, metrics in classical["models"].items()]
    rows.append({"model": "persian_distilbert", **transformer["metrics"]})
    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)

    import matplotlib.pyplot as plt
    import seaborn as sns

    chart = comparison.melt(
        id_vars="model",
        value_vars=["precision_macro", "recall_macro", "f1_macro"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=chart, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Persian Sentiment Model Comparison")
    plt.xticks(rotation=12)
    plt.tight_layout()
    plt.savefig(report_dir / "model_comparison.png", dpi=180)
    plt.close()
    return comparison


def train_transformer(
    data_path: Path,
    output_dir: Path,
    report_dir: Path,
    model_name: str = DEFAULT_MODEL,
    epochs: float = 2.0,
    batch_size: int = 8,
    max_length: int = 128,
    random_state: int = 42,
) -> dict[str, float]:
    """Fine-tune a sequence classifier and evaluate it on the held-out test split."""
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as error:
        raise RuntimeError('Install transformer dependencies with: pip install -e ".[transformer]"') from error

    frame = load_dataset(data_path)
    if "split" not in frame.columns or not {"train", "validation", "test"}.issubset(
        set(frame["split"])
    ):
        raise ValueError("Transformer training requires explicit train, validation, and test splits")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class SentimentDataset(Dataset):
        def __init__(self, data: pd.DataFrame) -> None:
            self.texts = data["text"].tolist()
            self.labels = [LABEL_TO_ID[label] for label in data["label"]]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, object]:
            encoded = tokenizer(
                self.texts[index], truncation=True, max_length=max_length, padding=False
            )
            encoded["labels"] = self.labels[index]
            return encoded

    train_frame = frame[frame["split"] == "train"]
    validation_frame = frame[frame["split"] == "validation"]
    test_frame = frame[frame["split"] == "test"]
    train_dataset = SentimentDataset(train_frame)
    validation_dataset = SentimentDataset(validation_frame)
    test_dataset = SentimentDataset(test_frame)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    counts = train_frame["label"].value_counts()
    class_weights = torch.tensor(
        [len(train_frame) / (len(LABELS) * counts[label]) for label in LABELS],
        dtype=torch.float32,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=class_weights.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    set_seed(random_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=random_state,
    )
    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_transformer_metrics,
    )
    checkpoint_dir = str(output_dir / "checkpoints")
    last_checkpoint = get_last_checkpoint(checkpoint_dir) if Path(checkpoint_dir).exists() else None
    trainer.train(resume_from_checkpoint=last_checkpoint)
    prediction = trainer.predict(test_dataset)
    predicted_ids = np.argmax(prediction.predictions, axis=-1)
    metrics = compute_transformer_metrics((prediction.predictions, prediction.label_ids))
    detailed_report = classification_report(
        prediction.label_ids,
        predicted_ids,
        labels=list(ID_TO_LABEL),
        target_names=[ID_TO_LABEL[index] for index in ID_TO_LABEL],
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "test_rows": len(test_dataset),
        "metrics": metrics,
    }
    with (report_dir / "transformer_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with (report_dir / "transformer_classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(detailed_report, file, ensure_ascii=False, indent=2)
    final_model_dir = output_dir / "transformer_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    write_model_comparison(report_dir)
    return metrics
