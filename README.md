# English NLP & Sentiment Analysis System

[![CI](https://github.com/h2oedian/english-sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/h2oedian/english-sentiment-analysis/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end, reproducible English sentiment-analysis portfolio project. It classifies text as
**negative**, **neutral**, or **positive**, compares classical machine learning with a RoBERTa
transformer, and exposes the selected model through a tested API and browser interface.

## Highlights

- Official TweetEval train/validation/test splits with leakage checks
- Negation-aware English normalization (`don't` → `do not`)
- Word and character TF-IDF features
- Logistic Regression, calibrated Linear SVM, and Multinomial Naive Bayes
- Twitter-RoBERTa transformer benchmark
- Macro precision, recall, F1, weighted F1, per-class reports, and confusion matrices
- FastAPI endpoints, responsive English UI, Docker image, and GitHub Actions CI
- Saved, probability-calibrated deployment model

## Architecture

```text
TweetEval → validation/normalization → TF-IDF baselines ─┐
                                                        ├→ metrics + comparison report
TweetEval test → Twitter-RoBERTa evaluation ─────────────┘
                                                        ↓
                                      saved model → FastAPI → English web UI
```

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
english-sentiment prepare-data
english-sentiment train
pytest
uvicorn english_sentiment.api:app --reload
```

Open `http://127.0.0.1:8000` for the demo or `http://127.0.0.1:8000/docs` for OpenAPI docs.

## Dataset

The default benchmark is the sentiment subset of
[TweetEval](https://github.com/cardiffnlp/tweeteval), containing English tweets with three labels.
The project downloads the original files at runtime and does not redistribute the dataset.
Official splits are retained; classical models combine train and validation for final fitting and
are evaluated once on the test split.

The generated CSV schema is:

```csv
text,label,source_id,split
I love this!,positive,0a1b2c3d4e5f6789,train
```

## Experiments

Run the reproducible classical comparison:

```powershell
python -m pip install -e ".[experiment]"
english-sentiment prepare-data
english-sentiment train
```

Evaluate the CC-BY-4.0 licensed
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
on the same held-out test data:

```powershell
python -m pip install -e ".[transformer]"
english-sentiment evaluate-transformer
```

Generated files include JSON classification reports, confusion matrices, a CSV comparison table,
a comparison chart, and the selected classical model. Macro F1 is the primary metric because it
gives equal importance to all three classes despite class imbalance.

### Reproduced results

Results below were generated on all 12,284 examples in the official TweetEval test split:

| Model | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Twitter-RoBERTa | **0.718** | **0.731** | **0.722** | **0.719** |
| Logistic Regression | 0.601 | 0.619 | 0.608 | 0.611 |
| Multinomial Naive Bayes | 0.610 | 0.591 | 0.591 | 0.602 |
| Calibrated Linear SVM | 0.611 | 0.586 | 0.581 | 0.595 |

Twitter-RoBERTa improves macro F1 by 11.3 percentage points over the strongest classical baseline.
Exact machine-readable values are committed under `reports/`.

## API

```powershell
python -m pip install -e ".[api]"
uvicorn english_sentiment.api:app --reload
```

Endpoints:

- `GET /health` — service and backend status
- `POST /predict` — one text (maximum 1,000 characters)
- `POST /predict/batch` — up to 100 texts
- `GET /docs` — interactive API documentation

Example request:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post `
  -ContentType application/json -Body '{"text":"I do not like this product."}'
```

The compact classical model is the default. To serve a downloaded transformer artifact:

```powershell
$env:SENTIMENT_BACKEND = "transformer"
$env:SENTIMENT_MODEL_PATH = "artifacts/transformer_model"
uvicorn english_sentiment.api:app
```

## Docker

```powershell
docker build -t english-sentiment-api .
docker run --rm -p 8000:8000 english-sentiment-api
```

## Repository structure

```text
src/english_sentiment/  reusable package, API, and web UI
tests/                  unit and integration tests
data/raw/               downloaded data (Git-ignored)
models/                 compact deployment model
artifacts/              transformer files (Git-ignored)
reports/                metrics, plots, and experiment summaries
```

## Reproducibility and limitations

- Dataset splits are official and model configuration is committed.
- Raw data and large transformer weights are intentionally excluded from Git.
- Social-media sentiment is subjective; sarcasm, mixed opinions, and domain shifts remain hard.
- Predictions should not be used for high-stakes or individual-level decisions.

## License and citation

Project code is MIT licensed. TweetEval and pretrained-model terms apply separately. If you use the
benchmark, cite Barbieri et al., *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet
Classification* (Findings of EMNLP 2020). The RoBERTa model card specifies CC-BY-4.0.
