"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training import train_classical_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Persian sentiment analysis experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train and compare classical baselines")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    train.add_argument("--reports", type=Path, default=Path("reports"))
    train.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    if args.command == "train":
        result = train_classical_models(args.data, args.artifacts, args.reports, args.test_size)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

