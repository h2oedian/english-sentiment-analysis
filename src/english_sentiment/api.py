"""FastAPI inference service for English sentiment analysis."""

from __future__ import annotations

import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .constants import LABELS
from .preprocessing import normalize_english_text


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    backend: str


class SentimentPredictor(Protocol):
    backend: str
    def predict_many(self, texts: list[str]) -> list[Prediction]: ...


class ClassicalPredictor:
    backend = "classical"
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        normalized = [normalize_english_text(text) for text in texts]
        probabilities = self.model.predict_proba(normalized)
        classes = list(self.model.classes_)
        output = []
        for row in probabilities:
            scores = {label: float(row[classes.index(label)]) for label in LABELS}
            label = max(scores, key=scores.get)
            output.append(Prediction(label=label, confidence=scores[label],
                                     probabilities=scores, backend=self.backend))
        return output


class TransformerPredictor:
    backend = "transformer"
    def __init__(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError('Install dependencies with: pip install -e ".[transformer]"') from error
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        if set(self.model.config.id2label.values()) != set(LABELS):
            raise ValueError("Transformer artifact has invalid sentiment labels")
        self.model.eval()

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        encoded = self.tokenizer([normalize_english_text(t) for t in texts], padding=True,
            truncation=True, max_length=128, return_tensors="pt")
        with self.torch.inference_mode():
            probs = self.torch.softmax(self.model(**encoded).logits, dim=-1).numpy()
        output = []
        for row in probs:
            scores = {label: float(row[index]) for index, label in self.model.config.id2label.items()}
            label = max(scores, key=scores.get)
            output.append(Prediction(label=label, confidence=scores[label],
                                     probabilities=scores, backend=self.backend))
        return output


@lru_cache(maxsize=1)
def build_predictor() -> SentimentPredictor:
    if not os.getenv("SENTIMENT_MODEL_PATH"):
        selection = Path(os.getenv("SENTIMENT_SELECTION_PATH", "reports/run/selection.json"))
        metadata = json.loads(selection.read_text(encoding="utf-8"))
        backend = metadata["backend"]
        path = Path(metadata["artifact"])
        if backend == "transformer":
            return TransformerPredictor(path)
        if backend == "classical":
            return ClassicalPredictor(path)
        raise ValueError("Unknown selected backend")
    backend = os.getenv("SENTIMENT_BACKEND", "classical").lower()
    default = "artifacts/transformer_model" if backend == "transformer" else "artifacts/run/logistic_regression.joblib"
    path = Path(os.getenv("SENTIMENT_MODEL_PATH", default))
    if backend == "transformer":
        return TransformerPredictor(path)
    if backend == "classical":
        return ClassicalPredictor(path)
    raise ValueError("SENTIMENT_BACKEND must be classical or transformer")


def create_app(predictor: SentimentPredictor | None = None) -> FastAPI:
    app = FastAPI(title="English Sentiment Analysis API",
        description="Classify English text as negative, neutral, or positive.", version="1.0.0")
    def current():
        try:
            return predictor or build_predictor()
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail="Model unavailable") from error

    @app.get("/live")
    def live():
        return {"status": "alive"}

    @app.get("/", include_in_schema=False)
    def demo() -> FileResponse:
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        try:
            return {"status": "healthy", "backend": current().backend}
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/predict", response_model=Prediction)
    def predict(request: TextRequest) -> Prediction:
        if not request.text.strip():
            raise HTTPException(status_code=422, detail="Text must not be blank")
        return current().predict_many([request.text])[0]

    @app.post("/predict/batch", response_model=list[Prediction])
    def predict_batch(request: BatchRequest) -> list[Prediction]:
        if any(not text.strip() or len(text) > 1000 for text in request.texts):
            raise HTTPException(status_code=422, detail="Each text must contain 1 to 1000 characters")
        return current().predict_many(request.texts)
    return app


app = create_app()
