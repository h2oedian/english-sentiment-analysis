from fastapi.testclient import TestClient

from english_sentiment.api import Prediction, create_app


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


def test_missing_model_returns_503(monkeypatch):
    from english_sentiment.api import build_predictor
    monkeypatch.setenv("SENTIMENT_MODEL_PATH", "missing.joblib")
    build_predictor.cache_clear()
    missing = TestClient(create_app())
    assert missing.get("/live").status_code == 200
    assert missing.get("/health").status_code == 503
    assert missing.post("/predict", json={"text": "hello"}).status_code == 503
    build_predictor.cache_clear()
