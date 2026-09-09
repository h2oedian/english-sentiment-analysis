FROM python:3.11.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTIMENT_BACKEND=classical \
    SENTIMENT_MODEL_PATH=/models/model.joblib
WORKDIR /app
COPY requirements-api.lock requirements-build.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements-api.lock -r requirements-build.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps --no-build-isolation . \
    && useradd --uid 10001 --create-home sentiment
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "english_sentiment.api:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "32", "--timeout-keep-alive", "5"]
