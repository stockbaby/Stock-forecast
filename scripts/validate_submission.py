from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_baseline import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a competition submission CSV.")
    parser.add_argument("--input", default="outputs/submissions/result.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-weight-sum", type=float, default=1.0)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")

    df = pd.read_csv(path, dtype={"stock_id": str})
    validate_submission(df, args.top_k, args.max_weight_sum)
    print("Submission is valid.")


if __name__ == "__main__":
    main()
