import numpy as np

from persian_sentiment.transformer_training import compute_transformer_metrics


def test_transformer_metrics_are_perfect_for_correct_predictions() -> None:
    logits = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 5.0]])
    labels = np.array([0, 1, 2])
    metrics = compute_transformer_metrics((logits, labels))
    assert metrics == {
        "precision_macro": 1.0,
        "recall_macro": 1.0,
        "f1_macro": 1.0,
        "f1_weighted": 1.0,
    }
