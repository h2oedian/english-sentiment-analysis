"""Fine-tune RoBERTa and select checkpoints using validation only."""

from __future__ import annotations


import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .constants import ID_TO_LABEL
from .training import load_dataset

DEFAULT_MODEL = "FacebookAI/roberta-base"


def compute_transformer_metrics(prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = prediction
    predicted = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=list(ID_TO_LABEL), average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        labels, predicted, labels=list(ID_TO_LABEL), average="weighted", zero_division=0)[2]
    return {"precision_macro": float(precision), "recall_macro": float(recall),
            "f1_macro": float(f1), "f1_weighted": float(weighted)}


DEFAULT_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"


def predict_local(path, texts, batch_size=32):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    model.eval()
    result = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
        with torch.inference_mode():
            ids = model(**batch).logits.argmax(-1).tolist()
        result.extend(model.config.id2label[i] for i in ids)
    return result


def finetune_transformer(data_path, output_dir, report_dir, epochs=3, batch_size=16):
    """Train a base RoBERTa; select checkpoint only by validation macro F1."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    from .constants import LABEL_TO_ID
    from .experiment import analyze, register, save_json
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if (report_dir / "selection.json").exists():
        raise ValueError("Selection is frozen")
    set_seed(42)
    frame = load_dataset(data_path)
    train = frame[frame.split == "train"].drop_duplicates(subset=["text", "label"])
    validation = frame[frame.split == "validation"]
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL, revision=DEFAULT_REVISION, num_labels=3,
        id2label=ID_TO_LABEL, label2id=LABEL_TO_ID)

    class EncodedDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.tokens = tokenizer(rows.text.tolist(), truncation=True, max_length=128)
            self.labels = rows.label.map(LABEL_TO_ID).tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {**{key: value[index] for key, value in self.tokens.items()},
                    "labels": self.labels[index]}

    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        learning_rate=2e-5, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        greater_is_better=True, save_total_limit=2, seed=42, data_seed=42,
        report_to="none", dataloader_num_workers=0, use_cpu=not torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, train_dataset=EncodedDataset(train),
                      eval_dataset=EncodedDataset(validation),
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_transformer_metrics)
    trainer.train()
    prediction = trainer.predict(EncodedDataset(validation))
    predicted = [ID_TO_LABEL[int(i)] for i in prediction.predictions.argmax(-1)]
    metrics = analyze(validation, predicted, report_dir, "validation_roberta")
    artifact = output_dir / "transformer_model"
    trainer.save_model(str(artifact))
    tokenizer.save_pretrained(artifact)
    register(report_dir, "roberta", artifact, "transformer", metrics, data_path)
    save_json(report_dir / "transformer_summary.json", {
        "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION, "seed": 42,
        "evaluation_split": "validation", "metrics": metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint, "epochs": epochs,
        "batch_size": batch_size, "max_length": 128, "learning_rate": 2e-5})
    return metrics
