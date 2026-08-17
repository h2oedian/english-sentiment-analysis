"""Project command-line interface."""

import argparse
import json
from pathlib import Path

from .dataset import prepare_tweeteval
from .training import train_classical_models


def main() -> None:
    parser = argparse.ArgumentParser(description="English sentiment analysis experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data", help="Download the TweetEval benchmark")
    prepare.add_argument("--output", type=Path, default=Path("data/raw/tweeteval.csv"))
    train = commands.add_parser("train", help="Train and compare classical baselines")
    train.add_argument("--data", type=Path, default=Path("data/raw/tweeteval.csv"))
    train.add_argument("--models", type=Path, default=Path("models"))
    train.add_argument("--reports", type=Path, default=Path("reports"))
    transformer = commands.add_parser("evaluate-transformer", help="Evaluate English RoBERTa")
    transformer.add_argument("--data", type=Path, default=Path("data/raw/tweeteval.csv"))
    transformer.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    transformer.add_argument("--reports", type=Path, default=Path("reports"))
    transformer.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    transformer.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.command == "prepare-data":
        frame = prepare_tweeteval(args.output)
        print(frame.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    elif args.command == "train":
        print(json.dumps(train_classical_models(args.data, args.models, args.reports), indent=2))
    else:
        from .transformer import evaluate_transformer
        metrics = evaluate_transformer(args.data, args.artifacts, args.reports,
                                       args.model, args.batch_size)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
