from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.construct import build_top_k_submission, evaluate_portfolio_strategy
from src.training.train_baseline import validate_submission


def _add_score_transform(df: pd.DataFrame, score_col: str, transform: str) -> tuple[pd.DataFrame, str]:
    if transform == "none":
        return df, score_col
    if transform == "date_zscore":
        out = df.copy()
        transformed_col = f"{score_col}_date_z"
        out[transformed_col] = out.groupby("date")[score_col].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 1e-8 else 1.0)
        )
        return out, transformed_col
    raise ValueError(f"Unsupported score transform: {transform}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process prediction scores into a submission.")
    parser.add_argument("--prediction-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metrics-path")
    parser.add_argument("--label-col", default="y_ret_a_stage_round1_open_open")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--score-transform", choices=["none", "date_zscore"], default="none")
    parser.add_argument("--strategy", default="softmax_t0.6")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-weight-sum", type=float, default=1.0)
    parser.add_argument("--industry-map-path")
    parser.add_argument("--max-per-industry", type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.prediction_path, dtype={"stock_id": str})
    df["date"] = pd.to_datetime(df["date"])
    if "label" in df.columns and args.label_col not in df.columns:
        df = df.rename(columns={"label": args.label_col})
    df, effective_score_col = _add_score_transform(df, args.score_col, args.score_transform)

    latest = df[df["date"] == df["date"].max()].copy()
    submission = build_top_k_submission(
        latest,
        score_col=effective_score_col,
        stock_col="stock_id",
        top_k=args.top_k,
        max_weight_sum=args.max_weight_sum,
        strategy=args.strategy,
        temperature=args.temperature,
        industry_map_path=args.industry_map_path,
        max_per_industry=args.max_per_industry,
    )
    validate_submission(submission, args.top_k, args.max_weight_sum)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    metrics = {
        "prediction_path": args.prediction_path,
        "output_path": str(output_path),
        "score_col": effective_score_col,
        "score_transform": args.score_transform,
        "strategy": args.strategy,
        "max_per_industry": args.max_per_industry,
        "submission_weight_sum": float(submission["weight"].sum()),
    }
    if args.label_col in df.columns:
        metrics["portfolio_eval"] = evaluate_portfolio_strategy(
            df,
            label_col=args.label_col,
            score_col=effective_score_col,
            strategy=args.strategy,
            top_k=args.top_k,
            max_weight_sum=args.max_weight_sum,
            temperature=args.temperature,
            industry_map_path=args.industry_map_path,
            max_per_industry=args.max_per_industry,
        )
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(submission.to_string(index=False))


if __name__ == "__main__":
    main()
