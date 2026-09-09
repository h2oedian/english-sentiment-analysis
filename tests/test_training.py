import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from english_sentiment.constants import LABELS
from english_sentiment.experiment import evaluate_test, select_model
from english_sentiment.training import load_dataset, train_classical_models


@pytest.fixture
def dataset(tmp_path):
    rows = [{"text": f"{split} {label} sample {i}", "label": label, "split": split}
            for split in ("train", "validation", "test") for label in LABELS for i in range(6)]
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_full_pipeline_and_one_shot_test(dataset, tmp_path):
    models, reports = tmp_path / "models", tmp_path / "reports"
    summary = train_classical_models(dataset, models, reports)
    assert "majority" in summary["models"]
    assert not list(reports.glob("test*"))
    for path in models.glob("*.joblib"):
        vocabulary = joblib.load(path).named_steps["features"].transformer_list[0][1].vocabulary_
        assert "validation" not in vocabulary and "test" not in vocabulary
    winner = select_model(reports)
    assert winner["name"] == summary["best_model"]
    from english_sentiment.api import ClassicalPredictor
    predictor = ClassicalPredictor(Path(winner["artifact"]))
    assert len(predictor.predict_many(["positive sample"])) == 1
    result = evaluate_test(dataset, reports)
    assert result["evaluation_split"] == "test"
    with pytest.raises(FileExistsError):
        evaluate_test(dataset, reports)
    with pytest.raises(ValueError, match="frozen"):
        train_classical_models(dataset, models, reports)


def test_rejects_normalized_cross_split_leakage(dataset):
    frame = pd.read_csv(dataset)
    frame.loc[18, "text"] = frame.loc[0, "text"].upper()
    frame.to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="leakage"):
        load_dataset(dataset)


def test_changed_dataset_is_rejected(dataset, tmp_path):
    reports = tmp_path / "reports"
    train_classical_models(dataset, tmp_path / "models", reports)
    select_model(reports)
    dataset.write_text(dataset.read_text() + "\n")
    with pytest.raises(ValueError, match="Dataset changed"):
        evaluate_test(dataset, reports)
    assert not (reports / "test_consumed.json").exists()


def test_validation_alone_controls_selection(tmp_path):
    for name, score in [("a", 0.2), ("b", 0.8)]:
        (tmp_path / f"candidate_{name}.json").write_text(json.dumps({
            "name": name, "dataset_sha256": "same", "evaluation_split": "validation",
            "metrics": {"f1_macro": score}}))
    assert select_model(tmp_path)["name"] == "b"


def test_changed_artifact_is_rejected(dataset, tmp_path):
    reports = tmp_path / "reports"
    train_classical_models(dataset, tmp_path / "models", reports)
    winner = select_model(reports)
    with Path(winner["artifact"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        evaluate_test(dataset, reports)


def test_test_labels_do_not_affect_validation_scores(dataset, tmp_path):
    first = train_classical_models(dataset, tmp_path / "m1", tmp_path / "r1")
    frame = pd.read_csv(dataset)
    mask = frame.split == "test"
    frame.loc[mask, "label"] = frame.loc[mask, "label"].map(
        {"negative": "positive", "positive": "neutral", "neutral": "negative"})
    frame.to_csv(dataset, index=False)
    second = train_classical_models(dataset, tmp_path / "m2", tmp_path / "r2")
    assert first["models"] == second["models"]
    assert first["best_model"] == second["best_model"]
