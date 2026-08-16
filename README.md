# Persian NLP & Sentiment Analysis System

An end-to-end, reproducible Persian sentiment-analysis project that compares classical
machine-learning baselines with transformer models using macro F1, precision, and recall.

> Status: Phase 3 — licensed data preparation, classical baselines, and Persian DistilBERT
> comparison are implemented. API and interactive demo are planned next.

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
- Reproducible ParsiNLU download with review-level aggregation and official splits

## Dataset format

The default dataset is [Persian Twitter Sentiment](https://huggingface.co/datasets/moali-mkh-2000/PersianTwitterDataset-SentimentAnalysis),
licensed under the Open Database License (ODbL). Download and prepare it with:

```powershell
persian-sentiment prepare-data
```

The source emotion labels are transparently mapped as `Happy → positive`, `Sad/Angry → negative`,
and `Neutral → neutral`; the ambiguous `Intense Emotions` class is excluded. A deterministic,
stratified 70/15/15 train/validation/test split is generated with random seed 42.

[ParsiNLU Sentiment](https://huggingface.co/datasets/persiannlp/parsinlu_sentiment) is also
supported as an optional source under CC BY-NC-SA 4.0:

```powershell
persian-sentiment prepare-data --source parsinlu
```

ParsiNLU aspect labels are collapsed into unambiguous review-level labels, but its neutral class
is very small; it is provided for research and robustness checks rather than the primary benchmark.

You can alternatively provide a UTF-8 CSV file at `data/raw/reviews.csv` with these columns:

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

## Baseline results

Results on the held-out Persian Twitter test split (441 examples):

| Model | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Persian DistilBERT (2 epochs) | **0.614** | **0.618** | **0.615** | **0.660** |
| Logistic Regression | 0.538 | 0.533 | 0.535 | 0.614 |
| Linear SVM | 0.528 | 0.519 | 0.522 | 0.600 |
| Multinomial Naive Bayes | 0.413 | 0.437 | 0.403 | 0.539 |

Macro F1 is the primary metric because the neutral class is smaller. Persian DistilBERT improves
macro F1 by 8.0 percentage points over the strongest classical baseline. The gap between macro and
weighted F1 still highlights class imbalance. Machine-readable reports, confusion matrices, and a
combined comparison chart are available in `reports/`.

## Transformer training

The transformer experiment fine-tunes the Apache-2.0 licensed
[`HooshvareLab/distilbert-fa-zwnj-base`](https://huggingface.co/HooshvareLab/distilbert-fa-zwnj-base).
It uses class-weighted cross-entropy, validation-based checkpoint selection, and the same held-out
test split as the classical baselines.

```powershell
python -m pip install -e ".[transformer]"
persian-sentiment train-transformer --data data/raw/reviews.csv
```

The default settings use two epochs, batch size 8, and maximum sequence length 128. Training runs
on CPU when CUDA is unavailable. Model checkpoints stay under the ignored `artifacts/` directory;
only compact evaluation reports are versioned. Interrupted or extended runs automatically resume
from the most recent checkpoint in the output directory.

## Roadmap

- [x] Licensed dataset pipeline and reproducible classical ML baselines
- [x] Persian DistilBERT transformer fine-tuning
- [x] Unified experiment comparison table and chart
- [ ] Qualitative error analysis
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
Dataset files are not redistributed by this repository. Persian Twitter data remains licensed
under [ODbL](https://opendatacommons.org/licenses/odbl/1-0/), while optional ParsiNLU data remains
under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Citation

For the default dataset, cite Elahimanesh, Mohammadkhani, and Kasaei, *Emotion Alignment:
Discovering the Gap Between Social Media and Real-World Sentiments in Persian Tweets and Images*
(2025), arXiv:2504.10662. For optional ParsiNLU, cite Khashabi et al. (2020), arXiv:2012.06154.
