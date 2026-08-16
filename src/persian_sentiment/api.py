"""FastAPI service for Persian sentiment inference."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .constants import LABELS
from .preprocessing import normalize_persian_text


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    backend: str


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1_000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)


class SentimentPredictor(Protocol):
    backend: str

    def predict_many(self, texts: list[str]) -> list[Prediction]: ...


class ClassicalPredictor:
    backend = "logistic_regression"

    def __init__(self, model_path: Path) -> None:
        import joblib

        if not model_path.exists():
            raise FileNotFoundError(f"Classical model not found: {model_path}")
        self.model = joblib.load(model_path)

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        normalized = [normalize_persian_text(text) for text in texts]
        probabilities = self.model.predict_proba(normalized)
        classes = list(self.model.classes_)
        results = []
        for row in probabilities:
            scores = {label: float(row[classes.index(label)]) for label in LABELS}
            label = max(scores, key=scores.get)
            results.append(
                Prediction(
                    label=label,
                    confidence=scores[label],
                    probabilities=scores,
                    backend=self.backend,
                )
            )
        return results


class TransformerPredictor:
    backend = "persian_distilbert"

    def __init__(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install transformer dependencies to use this backend") from error
        if not model_path.exists():
            raise FileNotFoundError(f"Transformer model not found: {model_path}")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        normalized = [normalize_persian_text(text) for text in texts]
        encoded = self.tokenizer(
            normalized, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = self.torch.softmax(logits, dim=-1).cpu().numpy()
        results = []
        for row in probabilities:
            scores = {label: float(row[index]) for index, label in enumerate(LABELS)}
            label = max(scores, key=scores.get)
            results.append(
                Prediction(
                    label=label,
                    confidence=scores[label],
                    probabilities=scores,
                    backend=self.backend,
                )
            )
        return results


@lru_cache(maxsize=1)
def build_predictor() -> SentimentPredictor:
    backend = os.getenv("SENTIMENT_BACKEND", "classical").lower()
    if backend == "transformer":
        path = Path(os.getenv("SENTIMENT_MODEL_PATH", "artifacts/transformer_model"))
        return TransformerPredictor(path)
    if backend == "classical":
        path = Path(os.getenv("SENTIMENT_MODEL_PATH", "models/best_classical_model.joblib"))
        return ClassicalPredictor(path)
    raise ValueError("SENTIMENT_BACKEND must be 'classical' or 'transformer'")


def create_app(predictor: SentimentPredictor | None = None) -> FastAPI:
    application = FastAPI(
        title="Persian Sentiment Analysis API",
        description="Positive, negative, and neutral classification for Persian user text.",
        version="0.3.0",
    )

    def current_predictor() -> SentimentPredictor:
        return predictor or build_predictor()

    @application.get("/", include_in_schema=False)
    def demo() -> FileResponse:
        return FileResponse(Path(__file__).with_name("static") / "index.html")

    @application.get("/health")
    def health() -> dict[str, str]:
        try:
            backend = current_predictor().backend
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {"status": "healthy", "backend": backend}

    @application.post("/predict", response_model=Prediction)
    def predict(request: TextRequest) -> Prediction:
        return current_predictor().predict_many([request.text])[0]

    @application.post("/predict/batch", response_model=list[Prediction])
    def predict_batch(request: BatchRequest) -> list[Prediction]:
        if any(not text.strip() for text in request.texts):
            raise HTTPException(status_code=422, detail="Texts must not be blank")
        return current_predictor().predict_many(request.texts)

    return application


app = create_app()
