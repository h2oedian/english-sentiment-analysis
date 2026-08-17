from pathlib import Path

from fastapi.testclient import TestClient

from english_sentiment.api import ClassicalPredictor, Prediction, create_app


class FakePredictor:
    backend = "fake"
    def predict_many(self, texts: list[str]) -> list[Prediction]:
        return [Prediction(label="positive", confidence=0.8,
            probabilities={"negative": 0.1, "neutral": 0.1, "positive": 0.8},
            backend=self.backend) for _ in texts]


client = TestClient(create_app(FakePredictor()))


def test_health_endpoint() -> None:
    assert client.get("/health").json() == {"status": "healthy", "backend": "fake"}


def test_demo_is_fully_english() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "English Sentiment Studio" in response.text
    assert 'lang="en"' in response.text


def test_predict_endpoint() -> None:
    response = client.post("/predict", json={"text": "I absolutely love it."})
    assert response.status_code == 200
    assert response.json()["label"] == "positive"


def test_batch_rejects_blank_text() -> None:
    assert client.post("/predict/batch", json={"texts": ["good", "  "]}).status_code == 422


def test_request_length_is_validated() -> None:
    assert client.post("/predict", json={"text": "x" * 1001}).status_code == 422


def test_committed_model_handles_clear_sentiment_and_negation() -> None:
    predictor = ClassicalPredictor(Path("models/best_classical_model.joblib"))
    samples = ["I love this product!", "I don't like this product at all."]
    predictions = predictor.predict_many(samples)
    assert [item.label for item in predictions] == ["positive", "negative"]
    assert all(abs(sum(item.probabilities.values()) - 1) < 1e-8 for item in predictions)
