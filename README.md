# Persian NLP & Sentiment Analysis System

An end-to-end, reproducible Persian sentiment-analysis project that compares classical
machine-learning baselines with transformer models using macro F1, precision, and recall.

> Status: Phase 1 — classical baselines are implemented. Transformer training, API, and demo
> are planned next.

## Why this project?

Persian text contains practical normalization challenges such as Arabic/Persian character
variants, zero-width non-joiners, URLs, and informal writing. This project treats preprocessing,
data splitting, evaluation, and experiment reproducibility as first-class concerns.

## Current features

- Persian-aware text normalization without data leakage
- Stratified train/test split with a fixed random seed
- TF-IDF word and character features
- Logistic Regression, Linear SVM, and Multinomial Naive Bayes baselines
- Macro/weighted F1, precision, recall, classification reports, and confusion matrices
- Saved model artifact and machine-readable experiment summary
- Unit tests for critical normalization behavior

## Dataset format

Put a UTF-8 CSV file at `data/raw/reviews.csv` with these columns:

```csv
text,label
این محصول واقعاً عالی بود,positive
اصلاً راضی نبودم,negative
معمولی بود,neutral
```

Labels must be `positive`, `negative`, or `neutral`. Do not commit a dataset unless its license
allows redistribution; document its source and license in your final project.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
persian-sentiment train --data data/raw/reviews.csv
pytest
```

Outputs are written to `artifacts/` and `reports/`.

## Roadmap

- [x] Reproducible classical ML baselines
- [ ] ParsBERT/XLM-R transformer fine-tuning
- [ ] Unified experiment comparison table and error analysis
- [ ] FastAPI inference service and Docker image
- [ ] Streamlit or Gradio demo
- [ ] GitHub Actions CI and polished model card

## Repository structure

```text
src/persian_sentiment/  reusable Python package
tests/                  unit tests
data/raw/               local datasets (ignored by Git)
artifacts/               trained models (ignored by Git)
reports/                 metrics and plots
```

## License

MIT (code only). Dataset and pretrained model licenses must be reviewed separately.

