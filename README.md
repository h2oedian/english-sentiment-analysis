# English sentiment analysis

A validation-first TweetEval experiment and FastAPI service. Python 3.11/3.12.

## Reproducible installation

```sh
python -m venv .venv
# Activate .venv for your shell first.
python -m pip install --require-hashes -r requirements-dev.lock -r requirements-build.lock
python -m pip install --no-deps --no-build-isolation -e .
pytest -q
```

Direct dependencies are pinned in pyproject.toml. Universal lockfiles pin transitive
versions and distribution hashes. Dataset downloads use commit
`4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`; the prepared CSV gets a SHA-256 manifest.
RoBERTa uses `FacebookAI/roberta-base` at revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`. Seeds are 42.
Exact floating-point reproducibility across hardware is not guaranteed.

## Experiment lifecycle

```sh
english-sentiment prepare-data
english-sentiment train
english-sentiment finetune-transformer --epochs 3 --batch-size 16
english-sentiment select
english-sentiment evaluate-test
```

All candidates share `--reports reports/run` and the same dataset. Use distinct
artifact/report directories for independent development experiments. Run all desired
candidates before `select`; it freezes the winner by **validation macro F1**, with
deterministic name ordering for ties. `train` fits TF-IDF and classifiers on train only:
Logistic Regression, calibrated Linear SVM, Multinomial NB, and a majority/prior baseline.
SVM calibration uses only training folds. There is no train+validation refit.

`finetune-transformer` trains the base RoBERTa classification head and backbone on train,
checks validation every epoch, and restores the best validation macro-F1 checkpoint.
It uses learning rate 2e-5, weight decay 0.01, maximum length 128, and dynamic padding.
This is a real fine-tuning implementation; no full benchmark scores from this new
protocol are claimed. Full RoBERTa training requires suitable compute and model downloads.
The same English normalization is used during training and serving.

Official split membership is required. Loading rejects empty text, invalid labels,
missing classes, source-ID overlap, and normalized/case-insensitive text overlap across
splits. Conflicting annotations within a split are retained as dataset ambiguity. Training
alone removes exact duplicate text/label pairs; validation and test retain their rows.
Custom data that violates these checks must be corrected before training.

`evaluate-test` evaluates **only the frozen winner**. It checks dataset and artifact
hashes, then exclusively creates `test_consumed.json` before inference. Another test
attempt, including after an interrupted evaluation, fails closed. The gate is scoped
to the experiment directory, not a tamper-proof global access control. Do not delete
it or create new runs to tune against the same test set. The original project already
used this test set for selection, so an unbiased new scientific claim needs a fresh
unseen holdout; a software fix cannot undo earlier exposure.

## Error analysis

Each validation candidate produces classification metrics, a labeled confusion matrix,
false-positive/false-negative examples in `validation_*_errors.csv`, and accuracy slices
with/without negation in `validation_*_analysis.json`. Inspect these for development.
The final winner produces equivalent `test_*` reports only after selection. Test error
analysis is descriptive; do not use it for another tuning cycle on the same test set.

## API and Docker

The default API reads `reports/run/selection.json` and serves its chosen artifact/backend.
To override it explicitly:

```sh
# Set SENTIMENT_BACKEND=classical and SENTIMENT_MODEL_PATH to the selected .joblib file.
uvicorn english_sentiment.api:app --host 127.0.0.1 --port 8000
```

For a transformer, set backend `transformer` and path to the saved transformer directory.
Transformer loading is offline and validates the label mapping. Load only trusted
joblib artifacts. `/live` is liveness; `/health` is readiness (503 when no model is
available). `/predict` accepts 1-1000 characters; `/predict/batch` accepts 1-100 texts.
Blank inputs are rejected. Model-load failures return 503 rather than leaking paths.

```sh
docker build -t english-sentiment-api .
docker run --rm -p 8000:8000 --mount type=bind,source=/absolute/selected-model.joblib,target=/models/model.joblib,readonly english-sentiment-api
```

The CPU classical image uses a fixed Python image tag, hash-locked dependencies,
non-root UID 10001, a readiness healthcheck, and bounded request concurrency. Model
artifacts are mounted read-only, not baked into the image. The image tag is versioned,
but not pinned to a registry digest. Transformer serving requires the transformer
extra/environment; it is not included in the compact Docker image.

## Historical results

Files in `reports/legacy/` and `models/best_classical_model.joblib` are historical outputs
from the old test-selected protocol. They are not current benchmark evidence, are not
loaded by the new default API, and are excluded from Docker. Retrain with the new pipeline.

## License

Code: MIT. Dataset and pretrained-model licenses apply separately. Cite Barbieri et al.,
*TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification* (2020).
