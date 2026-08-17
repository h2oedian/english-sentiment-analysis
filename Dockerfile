FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTIMENT_BACKEND=classical \
    SENTIMENT_MODEL_PATH=/app/models/best_classical_model.joblib

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY models ./models
RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "english_sentiment.api:app", "--host", "0.0.0.0", "--port", "8000"]
