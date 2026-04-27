from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.industry_context import load_industry_map
from src.portfolio.construct import build_top_k_submission, evaluate_portfolio_strategy
from src.training.train_baseline import validate_submission


def add_relation_score(
    df: pd.DataFrame,
    industry_map_path: str,
    mode: str,
    alpha: float,
    source_score_col: str = "score",
) -> pd.DataFrame:
    out = df.copy()
    out["stock_id"] = out["stock_id"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("")
    out["date"] = pd.to_datetime(out["date"])

    industry_map = load_industry_map(industry_map_path)[["stock_id", "industry_name"]].copy()
    industry_map["stock_id"] = industry_map["stock_id"].astype(str).str.zfill(6)
    out = out.merge(industry_map, on="stock_id", how="left")
    out["industry_name"] = out["industry_name"].fillna("other")

    out["score_z"] = out.groupby("date")[source_score_col].transform(
        lambda s: (s - s.mean()) / (s.std() if s.std() > 1e-8 else 1.0)
    )
    out["industry_mean_score_z"] = out.groupby(["date", "industry_name"])["score_z"].transform("mean")
    out["industry_rank_score"] = out.groupby(["date", "industry_name"])["score_z"].rank(pct=True).fillna(0.5)
    out["global_rank_score"] = out.groupby("date")["score_z"].rank(pct=True).fillna(0.5)

    if mode == "smooth":
        out["score_relation"] = (1.0 - alpha) * out["score_z"] + alpha * out["industry_mean_score_z"]
    elif mode == "neutral":
        out["score_relation"] = out["score_z"] - alpha * out["industry_mean_score_z"]
    elif mode == "rank_mix":
        out["score_relation"] = (1.0 - alpha) * out["score_z"] + alpha * out["global_rank_score"]
    elif mode == "rank_ind_mix":
        out["score_relation"] = (1.0 - alpha) * out["score_z"] + alpha * out["industry_rank_score"]
    else:
        raise ValueError(f"Unsupported relation mode: {mode}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="HIST-style relation post-processing for prediction scores.")
    parser.add_argument("--prediction-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metrics-path")
    parser.add_argument("--industry-map-path", default="data/raw/hs300_stock_list.csv")
    parser.add_argument("--label-col", default="y_ret_a_stage_round1_open_open")
    parser.add_argument("--source-score-col", default="score")
    parser.add_argument("--mode", choices=["smooth", "neutral", "rank_mix", "rank_ind_mix"], default="rank_ind_mix")
    parser.add_argument("--alpha", type=float, default=-0.5)
    parser.add_argument("--strategy", default="softmax_t0.6")
    parser.add_argument("--max-per-industry", type=int)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-weight-sum", type=float, default=1.0)
    args = parser.parse_args()

    df = pd.read_csv(args.prediction_path, dtype={"stock_id": str})
    if "label" in df.columns and args.label_col not in df.columns:
        df = df.rename(columns={"label": args.label_col})
    scored = add_relation_score(
        df,
        industry_map_path=args.industry_map_path,
        mode=args.mode,
        alpha=args.alpha,
        source_score_col=args.source_score_col,
    )
    latest = scored[scored["date"] == scored["date"].max()].copy()
    submission = build_top_k_submission(
        latest,
        score_col="score_relation",
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
        "mode": args.mode,
        "alpha": args.alpha,
        "strategy": args.strategy,
        "max_per_industry": args.max_per_industry,
        "submission_weight_sum": float(submission["weight"].sum()),
    }
    if args.label_col in scored.columns:
        metrics["portfolio_eval"] = evaluate_portfolio_strategy(
            scored,
            label_col=args.label_col,
            score_col="score_relation",
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
