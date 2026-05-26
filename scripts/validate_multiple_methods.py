from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.construct import build_top_k_submission


DEFAULT_STRATEGIES = [
    "top1_weight",
    "confidence_topk",
    "top2_softmax",
    "dynamic_risk_budget",
    "top3_softmax",
    "proportional_positive_thr0.0",
]

DEFAULT_METHODS = [
    {
        "name": "master_official",
        "prediction_path": "outputs/predictions/master_alpha_official_rank_predictions.csv",
        "latest_prediction_path": "outputs/predictions/master_alpha_official_rank_latest_predictions.csv",
    },
    {
        "name": "stockmixer_fast",
        "prediction_path": "outputs/predictions/stockmixer_alpha_fast_predictions.csv",
        "latest_prediction_path": "outputs/predictions/stockmixer_alpha_fast_latest_predictions.csv",
    },
    {
        "name": "stockmixer_official",
        "prediction_path": "outputs/predictions/stockmixer_alpha_official_rank_predictions.csv",
        "latest_prediction_path": None,
    },
    {
        "name": "master_multiseed",
        "prediction_path": "outputs/predictions/master_multiseed/master_multiseed_predictions.csv",
        "latest_prediction_path": None,
    },
]


def _normalize_stock_id(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _parse_method_spec(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("Method spec must be name:prediction_path[:latest_prediction_path]")
    return {
        "name": parts[0],
        "prediction_path": parts[1],
        "latest_prediction_path": parts[2] if len(parts) == 3 and parts[2] else None,
    }


def _load_prediction(path: Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    if "label" in df.columns and label_col not in df.columns:
        df = df.rename(columns={"label": label_col})
    missing = {"stock_id", "date", "score"}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df.copy()
    out["stock_id"] = out["stock_id"].map(_normalize_stock_id)
    out["date"] = pd.to_datetime(out["date"])
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    if label_col in out.columns:
        out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    return out


def _evaluate_history(
    df: pd.DataFrame,
    label_col: str,
    strategy: str,
    top_k: int,
    max_weight_sum: float,
    temperature: float,
) -> tuple[dict, list[dict]]:
    daily_records = []
    for date, daily in df.dropna(subset=[label_col, "score"]).groupby("date"):
        submission = build_top_k_submission(
            daily,
            score_col="score",
            stock_col="stock_id",
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            strategy=strategy,
            temperature=temperature,
        )
        if submission.empty:
            daily_return = 0.0
            stocks = []
            weights = []
        else:
            merged = submission.merge(daily[["stock_id", label_col]], on="stock_id", how="left")
            daily_return = float((merged["weight"] * merged[label_col]).sum())
            stocks = submission["stock_id"].tolist()
            weights = submission["weight"].round(10).tolist()
        daily_records.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "return": daily_return,
                "stocks": stocks,
                "weights": weights,
            }
        )

    returns = np.asarray([row["return"] for row in daily_records], dtype=float)
    if len(returns) == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "var": math.nan,
            "neg_rate": math.nan,
            "min": math.nan,
            "p05": math.nan,
            "p10": math.nan,
            "max": math.nan,
            "max_drawdown": math.nan,
        }, daily_records
    equity = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(running_peak, 1e-12) - 1.0
    return {
        "n": int(len(returns)),
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "var": float(returns.var()),
        "neg_rate": float((returns < 0).mean()),
        "min": float(returns.min()),
        "p05": float(np.quantile(returns, 0.05)),
        "p10": float(np.quantile(returns, 0.10)),
        "max": float(returns.max()),
        "max_drawdown": float(drawdowns.min()),
    }, daily_records


def _evaluate_latest(
    latest_path: str | None,
    strategy: str,
    top_k: int,
    max_weight_sum: float,
    temperature: float,
    realized_return_path: str | None,
) -> tuple[float, str]:
    if not latest_path or not Path(latest_path).exists():
        return math.nan, ""
    latest = _load_prediction(Path(latest_path), label_col="__unused_label__")
    latest = latest[latest["date"] == latest["date"].max()].copy()
    submission = build_top_k_submission(
        latest,
        score_col="score",
        stock_col="stock_id",
        top_k=top_k,
        max_weight_sum=max_weight_sum,
        strategy=strategy,
        temperature=temperature,
    )
    submission_text = ";".join(f"{row.stock_id}:{row.weight:.4f}" for row in submission.itertuples())
    if not realized_return_path or not Path(realized_return_path).exists():
        return math.nan, submission_text
    realized = pd.read_csv(realized_return_path, dtype={"stock_id": str})
    realized["stock_id"] = realized["stock_id"].map(_normalize_stock_id)
    ret_col = "ret" if "ret" in realized.columns else "real_return"
    merged = submission.merge(realized[["stock_id", ret_col]], on="stock_id", how="left")
    return float((merged["weight"] * merged[ret_col]).sum()), submission_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multiple model/portfolio methods on the same window.")
    parser.add_argument("--method", action="append", help="name:prediction_path[:latest_prediction_path]")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--label-col", default="y_ret_a_stage_round1_open_open")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-weight-sum", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--realized-return-path", default="outputs/real_a_stage_0427_0430_returns.csv")
    parser.add_argument("--output-dir", default="outputs/method_validation")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    methods = [_parse_method_spec(item) for item in args.method] if args.method else DEFAULT_METHODS
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    records: dict[str, list[dict]] = {}
    for method in methods:
        prediction_path = Path(method["prediction_path"])
        if not prediction_path.exists():
            continue
        prediction = _load_prediction(prediction_path, args.label_col)
        if args.label_col not in prediction.columns:
            continue
        for strategy in strategies:
            metrics, daily_records = _evaluate_history(
                prediction,
                label_col=args.label_col,
                strategy=strategy,
                top_k=args.top_k,
                max_weight_sum=args.max_weight_sum,
                temperature=args.temperature,
            )
            latest_score, latest_submission = _evaluate_latest(
                method.get("latest_prediction_path"),
                strategy=strategy,
                top_k=args.top_k,
                max_weight_sum=args.max_weight_sum,
                temperature=args.temperature,
                realized_return_path=args.realized_return_path,
            )
            row = {
                "method": method["name"],
                "strategy": strategy,
                **metrics,
                "latest_a_score": latest_score,
                "latest_submission": latest_submission,
            }
            rows.append(row)
            records[f"{method['name']}::{strategy}"] = daily_records

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(["mean", "std"], ascending=[False, True])
    leaderboard.to_csv(output_dir / "multi_method_validation.csv", index=False)
    (output_dir / "daily_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(leaderboard.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
