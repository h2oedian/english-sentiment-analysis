from pathlib import Path

from fastapi.testclient import TestClient

from persian_sentiment.api import ClassicalPredictor, Prediction, create_app


class FakePredictor:
    backend = "fake"

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        return [
            Prediction(
                label="positive",
                confidence=0.8,
                probabilities={"negative": 0.1, "neutral": 0.1, "positive": 0.8},
                backend=self.backend,
            )
            for _ in texts
        ]


client = TestClient(create_app(FakePredictor()))


def test_health_endpoint() -> None:
    assert client.get("/health").json() == {"status": "healthy", "backend": "fake"}


def test_demo_page_is_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "تحلیل احساسات فارسی" in response.text


def test_predict_endpoint() -> None:
    response = client.post("/predict", json={"text": "این محصول عالی است"})
    assert response.status_code == 200
    assert response.json()["label"] == "positive"


def test_batch_endpoint_rejects_blank_text() -> None:
    response = client.post("/predict/batch", json={"texts": ["خوب", "  "]})
    assert response.status_code == 422


def test_request_length_is_validated() -> None:
    response = client.post("/predict", json={"text": "x" * 1001})
    assert response.status_code == 422


def test_committed_classical_model_can_predict() -> None:
    predictor = ClassicalPredictor(Path("models/best_classical_model.joblib"))
    prediction = predictor.predict_many(["این محصول عالی است"])[0]
    assert prediction.label in {"negative", "neutral", "positive"}
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-9
