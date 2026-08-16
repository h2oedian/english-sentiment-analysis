"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import prepare_parsinlu, prepare_persian_twitter
from .training import train_classical_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Persian sentiment analysis experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-data", help="Download and prepare a Persian dataset")
    prepare.add_argument("--output", type=Path, default=Path("data/raw/reviews.csv"))
    prepare.add_argument(
        "--source", choices=["twitter", "parsinlu"], default="twitter", help="Dataset source"
    )
    train = subparsers.add_parser("train", help="Train and compare classical baselines")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    train.add_argument("--reports", type=Path, default=Path("reports"))
    train.add_argument("--test-size", type=float, default=0.2)
    transformer = subparsers.add_parser("train-transformer", help="Fine-tune Persian DistilBERT")
    transformer.add_argument("--data", type=Path, required=True)
    transformer.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    transformer.add_argument("--reports", type=Path, default=Path("reports"))
    transformer.add_argument("--model", default="HooshvareLab/distilbert-fa-zwnj-base")
    transformer.add_argument("--epochs", type=float, default=2.0)
    transformer.add_argument("--batch-size", type=int, default=8)
    transformer.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    if args.command == "prepare-data":
        if args.source == "twitter":
            frame = prepare_persian_twitter(args.output)
        else:
            frame = prepare_parsinlu(args.output)
        print(frame.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    elif args.command == "train":
        result = train_classical_models(args.data, args.artifacts, args.reports, args.test_size)
        print(json.dumps(result, indent=2))
    elif args.command == "train-transformer":
        from .transformer_training import train_transformer

        result = train_transformer(
            args.data,
            args.artifacts,
            args.reports,
            args.model,
            args.epochs,
            args.batch_size,
            args.max_length,
        )
        print(json.dumps(result, indent=2))



if __name__ == "__main__":
    main()
