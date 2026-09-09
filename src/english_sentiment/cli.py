"""Explicit train -> select -> evaluate-test experiment lifecycle."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--output", type=Path, default=Path("data/raw/tweeteval.csv"))
    for command in ("train", "finetune-transformer", "select", "evaluate-test"):
        sub = commands.add_parser(command)
        sub.add_argument("--data", type=Path, default=Path("data/raw/tweeteval.csv"))
        sub.add_argument("--reports", type=Path, default=Path("reports/run"))
        sub.add_argument("--models", type=Path, default=Path("artifacts/run"))
        if command == "finetune-transformer":
            sub.add_argument("--epochs", type=int, default=3)
            sub.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.command == "prepare-data":
        from .dataset import prepare_tweeteval
        result = {"rows": len(prepare_tweeteval(args.output))}
    elif args.command == "train":
        from .training import train_classical_models
        result = train_classical_models(args.data, args.models, args.reports)
    elif args.command == "finetune-transformer":
        from .transformer import finetune_transformer
        result = finetune_transformer(args.data, args.models, args.reports,
                                      args.epochs, args.batch_size)
    elif args.command == "select":
        from .experiment import select_model
        result = select_model(args.reports)
    else:
        from .experiment import evaluate_test
        result = evaluate_test(args.data, args.reports)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
