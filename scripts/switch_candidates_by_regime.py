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

from src.portfolio.construct import build_top_k_submission, evaluate_portfolio_strategy
from src.training.train_baseline import validate_submission


def _normalize_stock_id(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _parse_model_spec(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError("Model specs must be name:validation_prediction_path:latest_prediction_path")
    return parts[0], Path(parts[1]), Path(parts[2])


def _load_prediction(path: Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    if "label" in df.columns and label_col not in df.columns:
        df = df.rename(columns={"label": label_col})
    if "score" not in df.columns:
        raise ValueError(f"{path} missing score column.")
    out = df.copy()
    out["stock_id"] = out["stock_id"].map(_normalize_stock_id)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _evaluate_recent(
    df: pd.DataFrame,
    label_col: str,
    strategy: str,
    window: int,
    top_k: int,
    max_weight_sum: float,
    temperature: float,
) -> dict:
    labeled = df.dropna(subset=[label_col, "score"]).copy()
    dates = sorted(pd.to_datetime(labeled["date"]).drop_duplicates().tolist())
    recent_dates = dates[-window:] if len(dates) > window else dates
    recent = labeled[labeled["date"].isin(recent_dates)].copy()
    metrics = evaluate_portfolio_strategy(
        recent,
        label_col=label_col,
        score_col="score",
        strategy=strategy,
        top_k=top_k,
        max_weight_sum=max_weight_sum,
        temperature=temperature,
    )
    return metrics


def _switch_walk_forward(
    model_frames: dict[str, pd.DataFrame],
    label_col: str,
    strategy: str,
    selection_window: int,
    min_history_days: int,
    top_k: int,
    max_weight_sum: float,
    temperature: float,
) -> dict:
    common_dates = None
    for frame in model_frames.values():
        dates = set(pd.to_datetime(frame.dropna(subset=[label_col])["date"]).drop_duplicates().tolist())
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    dates = sorted(common_dates or [])
    records = []
    for idx, date in enumerate(dates):
        history_dates = dates[max(0, idx - selection_window) : idx]
        if len(history_dates) < min_history_days:
            continue
        leaderboard = []
        for name, frame in model_frames.items():
            history = frame[frame["date"].isin(history_dates)].copy()
            metrics = evaluate_portfolio_strategy(
                history,
                label_col=label_col,
                score_col="score",
                strategy=strategy,
                top_k=top_k,
                max_weight_sum=max_weight_sum,
                temperature=temperature,
            )
            leaderboard.append({"name": name, **metrics})
        leaderboard.sort(key=lambda item: -math.inf if np.isnan(item["mean_return"]) else item["mean_return"], reverse=True)
        selected = leaderboard[0]["name"]
        daily = model_frames[selected][model_frames[selected]["date"] == date].copy()
        submission = build_top_k_submission(
            daily,
            score_col="score",
            stock_col="stock_id",
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            strategy=strategy,
            temperature=temperature,
        )
        validate_submission(submission, top_k, max_weight_sum)
        merged = submission.merge(daily[["stock_id", label_col]], on="stock_id", how="left")
        records.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "selected_model": selected,
                "return": float((merged["weight"] * merged[label_col]).sum()),
                "history_mean_return": leaderboard[0]["mean_return"],
                "stocks": submission["stock_id"].tolist(),
                "weights": submission["weight"].round(10).tolist(),
            }
        )
    returns = [row["return"] for row in records]
    return {
        "selection_window": selection_window,
        "num_days": len(records),
        "mean_return": float(np.mean(returns)) if returns else math.nan,
        "std_return": float(np.std(returns)) if returns else math.nan,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Switch between model candidates by recent online-window performance.")
    parser.add_argument("--model", action="append", required=True, help="name:validation_prediction_path:latest_prediction_path")
    parser.add_argument("--output-dir", default="outputs/candidate_switch")
    parser.add_argument("--label-col", default="y_ret_a_stage_round1_open_open")
    parser.add_argument("--strategy", default="top1_weight")
    parser.add_argument("--selection-windows", default="20,40")
    parser.add_argument("--min-history-days", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-weight-sum", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [_parse_model_spec(item) for item in args.model]
    model_frames = {name: _load_prediction(pred_path, args.label_col) for name, pred_path, _ in specs}
    latest_frames = {name: _load_prediction(latest_path, args.label_col) for name, _, latest_path in specs}

    walk_forward = [
        _switch_walk_forward(
            model_frames,
            label_col=args.label_col,
            strategy=args.strategy,
            selection_window=int(window),
            min_history_days=args.min_history_days,
            top_k=args.top_k,
            max_weight_sum=args.max_weight_sum,
            temperature=args.temperature,
        )
        for window in args.selection_windows.split(",")
        if window.strip()
    ]
    best_window = max(walk_forward, key=lambda item: -math.inf if np.isnan(item["mean_return"]) else item["mean_return"])
    latest_leaderboard = []
    for name, frame in model_frames.items():
        latest_leaderboard.append(
            {
                "name": name,
                **_evaluate_recent(
                    frame,
                    label_col=args.label_col,
                    strategy=args.strategy,
                    window=int(best_window["selection_window"]),
                    top_k=args.top_k,
                    max_weight_sum=args.max_weight_sum,
                    temperature=args.temperature,
                ),
            }
        )
    latest_leaderboard.sort(key=lambda item: -math.inf if np.isnan(item["mean_return"]) else item["mean_return"], reverse=True)
    selected_name = latest_leaderboard[0]["name"]
    latest_daily = latest_frames[selected_name]
    latest_daily = latest_daily[latest_daily["date"] == latest_daily["date"].max()].copy()
    submission = build_top_k_submission(
        latest_daily,
        score_col="score",
        stock_col="stock_id",
        top_k=args.top_k,
        max_weight_sum=args.max_weight_sum,
        strategy=args.strategy,
        temperature=args.temperature,
    )
    validate_submission(submission, args.top_k, args.max_weight_sum)
    submission_path = output_dir / "latest_submission.csv"
    submission.to_csv(submission_path, index=False)
    summary = {
        "strategy": args.strategy,
        "walk_forward": walk_forward,
        "selected_window": best_window["selection_window"],
        "latest_leaderboard": latest_leaderboard,
        "selected_model": selected_name,
        "latest_submission_path": str(submission_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        **{key: value for key, value in summary.items() if key != "walk_forward"},
        "walk_forward": [{key: value for key, value in item.items() if key != "records"} for item in walk_forward],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
