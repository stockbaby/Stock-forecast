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


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_prediction(path: Path, label_col: str, score_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    if "label" in df.columns and label_col not in df.columns:
        df = df.rename(columns={"label": label_col})
    missing = {"stock_id", "date", score_col}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df.copy()
    out["stock_id"] = out["stock_id"].map(_normalize_stock_id)
    out["date"] = pd.to_datetime(out["date"])
    if label_col in out.columns:
        out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    return out


def _strategy_grid(strategies: list[str], max_per_industry_values: list[int | None]) -> list[dict]:
    return [
        {"strategy": strategy, "max_per_industry": max_per_industry}
        for strategy in strategies
        for max_per_industry in max_per_industry_values
    ]


def _selection_value(metrics: dict, selection_score: str, risk_weight: float, negative_weight: float) -> float:
    mean_return = float(metrics.get("mean_return", math.nan))
    if np.isnan(mean_return):
        return -math.inf
    if selection_score == "mean_return":
        return mean_return
    if selection_score == "risk_adjusted":
        p05 = float(metrics.get("p05_return", 0.0))
        negative_rate = float(metrics.get("negative_rate", 0.0))
        return mean_return + risk_weight * p05 - negative_weight * negative_rate
    raise ValueError(f"Unsupported selection score: {selection_score}")


def _return_summary(returns: list[float]) -> dict:
    if not returns:
        return {
            "mean_return": math.nan,
            "std_return": math.nan,
            "p05_return": math.nan,
            "negative_rate": math.nan,
            "max_drawdown": math.nan,
        }
    return_array = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + return_array)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(running_peak, 1e-12) - 1.0
    return {
        "mean_return": float(np.mean(return_array)),
        "std_return": float(np.std(return_array)),
        "p05_return": float(np.quantile(return_array, 0.05)),
        "negative_rate": float(np.mean(return_array < 0.0)),
        "max_drawdown": float(np.min(drawdowns)),
    }


def _evaluate_one_day(
    daily: pd.DataFrame,
    label_col: str,
    score_col: str,
    strategy: str,
    top_k: int,
    max_weight_sum: float,
    temperature: float,
    industry_map_path: str | None,
    max_per_industry: int | None,
) -> tuple[float, pd.DataFrame]:
    submission = build_top_k_submission(
        daily,
        score_col=score_col,
        stock_col="stock_id",
        top_k=top_k,
        max_weight_sum=max_weight_sum,
        strategy=strategy,
        temperature=temperature,
        industry_map_path=industry_map_path,
        max_per_industry=max_per_industry,
    )
    validate_submission(submission, top_k, max_weight_sum)
    merged = submission.merge(daily[["stock_id", label_col]], on="stock_id", how="left")
    score = float((merged["weight"] * merged[label_col]).sum()) if not merged.empty else 0.0
    return score, submission


def _choose_best_spec(
    history: pd.DataFrame,
    specs: list[dict],
    label_col: str,
    score_col: str,
    top_k: int,
    max_weight_sum: float,
    temperature: float,
    industry_map_path: str | None,
    selection_score: str,
    risk_weight: float,
    negative_weight: float,
) -> tuple[dict, list[dict]]:
    rows = []
    for spec in specs:
        metrics = evaluate_portfolio_strategy(
            history,
            label_col=label_col,
            score_col=score_col,
            strategy=spec["strategy"],
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            temperature=temperature,
            industry_map_path=industry_map_path,
            max_per_industry=spec["max_per_industry"],
        )
        rows.append(
            {
                **spec,
                **metrics,
                "selection_value": _selection_value(metrics, selection_score, risk_weight, negative_weight),
            }
        )
    rows.sort(key=lambda item: item["selection_value"], reverse=True)
    return rows[0], rows


def walk_forward_simulation(
    df: pd.DataFrame,
    specs: list[dict],
    label_col: str,
    score_col: str,
    selection_window: int,
    min_history_days: int,
    top_k: int,
    max_weight_sum: float,
    temperature: float,
    industry_map_path: str | None,
    selection_score: str,
    risk_weight: float,
    negative_weight: float,
) -> dict:
    labeled = df.dropna(subset=[label_col, score_col]).copy()
    dates = sorted(pd.to_datetime(labeled["date"]).drop_duplicates().tolist())
    records = []
    for idx, date in enumerate(dates):
        history_dates = dates[max(0, idx - selection_window) : idx]
        if len(history_dates) < min_history_days:
            continue
        history = labeled[labeled["date"].isin(history_dates)].copy()
        daily = labeled[labeled["date"] == date].copy()
        best, _ = _choose_best_spec(
            history=history,
            specs=specs,
            label_col=label_col,
            score_col=score_col,
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            temperature=temperature,
            industry_map_path=industry_map_path,
            selection_score=selection_score,
            risk_weight=risk_weight,
            negative_weight=negative_weight,
        )
        score, submission = _evaluate_one_day(
            daily=daily,
            label_col=label_col,
            score_col=score_col,
            strategy=best["strategy"],
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            temperature=temperature,
            industry_map_path=industry_map_path,
            max_per_industry=best["max_per_industry"],
        )
        records.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "return": score,
                "selected_strategy": best["strategy"],
                "selected_max_per_industry": best["max_per_industry"],
                "history_mean_return": best["mean_return"],
                "stocks": submission["stock_id"].tolist(),
                "weights": submission["weight"].round(10).tolist(),
            }
        )
    returns = [row["return"] for row in records]
    return {
        "selection_window": selection_window,
        "min_history_days": min_history_days,
        "num_days": len(records),
        **_return_summary(returns),
        "records": records,
    }


def split_holdout_simulation(
    df: pd.DataFrame,
    specs: list[dict],
    label_col: str,
    score_col: str,
    top_k: int,
    max_weight_sum: float,
    temperature: float,
    industry_map_path: str | None,
    selection_score: str,
    risk_weight: float,
    negative_weight: float,
) -> dict:
    labeled = df.dropna(subset=[label_col, score_col]).copy()
    dates = sorted(pd.to_datetime(labeled["date"]).drop_duplicates().tolist())
    split_idx = len(dates) // 2
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    train = labeled[labeled["date"].isin(train_dates)].copy()
    test = labeled[labeled["date"].isin(test_dates)].copy()
    best, leaderboard = _choose_best_spec(
        history=train,
        specs=specs,
        label_col=label_col,
        score_col=score_col,
        top_k=top_k,
        max_weight_sum=max_weight_sum,
        temperature=temperature,
        industry_map_path=industry_map_path,
        selection_score=selection_score,
        risk_weight=risk_weight,
        negative_weight=negative_weight,
    )
    test_metrics = evaluate_portfolio_strategy(
        test,
        label_col=label_col,
        score_col=score_col,
        strategy=best["strategy"],
        top_k=top_k,
        max_weight_sum=max_weight_sum,
        temperature=temperature,
        industry_map_path=industry_map_path,
        max_per_industry=best["max_per_industry"],
    )
    return {
        "train_start": str(pd.Timestamp(train_dates[0]).date()) if train_dates else None,
        "train_end": str(pd.Timestamp(train_dates[-1]).date()) if train_dates else None,
        "test_start": str(pd.Timestamp(test_dates[0]).date()) if test_dates else None,
        "test_end": str(pd.Timestamp(test_dates[-1]).date()) if test_dates else None,
        "selected": best,
        "test_metrics": test_metrics,
        "leaderboard": leaderboard,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward online-window simulation for T-date submissions.")
    parser.add_argument("--prediction-path", required=True)
    parser.add_argument("--latest-prediction-path")
    parser.add_argument("--output-dir", default="outputs/online_window_sim")
    parser.add_argument("--label-col", default="y_ret_a_stage_round1_open_open")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--strategies", default="top1_weight,top2_softmax,confidence_topk,dynamic_risk_budget,softmax_t0.6,proportional_positive_thr0.0")
    parser.add_argument("--max-per-industry", default="none")
    parser.add_argument("--selection-windows", default="20,40,60,90")
    parser.add_argument("--selection-score", choices=["mean_return", "risk_adjusted"], default="risk_adjusted")
    parser.add_argument("--risk-weight", type=float, default=0.25)
    parser.add_argument("--negative-weight", type=float, default=0.01)
    parser.add_argument("--min-history-days", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-weight-sum", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--industry-map-path", default="data/raw/hs300_stock_list.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = _load_prediction(Path(args.prediction_path), args.label_col, args.score_col)
    max_per_industry_values: list[int | None] = [
        None if item.lower() in {"none", "null", "na"} else int(item)
        for item in _parse_csv_list(args.max_per_industry)
    ]
    specs = _strategy_grid(_parse_csv_list(args.strategies), max_per_industry_values)

    split = split_holdout_simulation(
        prediction,
        specs=specs,
        label_col=args.label_col,
        score_col=args.score_col,
        top_k=args.top_k,
        max_weight_sum=args.max_weight_sum,
        temperature=args.temperature,
        industry_map_path=args.industry_map_path,
        selection_score=args.selection_score,
        risk_weight=args.risk_weight,
        negative_weight=args.negative_weight,
    )
    walk_forward = [
        walk_forward_simulation(
            prediction,
            specs=specs,
            label_col=args.label_col,
            score_col=args.score_col,
            selection_window=int(window),
            min_history_days=args.min_history_days,
            top_k=args.top_k,
            max_weight_sum=args.max_weight_sum,
            temperature=args.temperature,
            industry_map_path=args.industry_map_path,
            selection_score=args.selection_score,
            risk_weight=args.risk_weight,
            negative_weight=args.negative_weight,
        )
        for window in _parse_csv_list(args.selection_windows)
    ]
    best_window = max(
        walk_forward,
        key=lambda item: _selection_value(item, args.selection_score, args.risk_weight, args.negative_weight),
    )
    latest_submission_path = None
    if args.latest_prediction_path:
        latest = _load_prediction(Path(args.latest_prediction_path), args.label_col, args.score_col)
        history_dates = sorted(pd.to_datetime(prediction.dropna(subset=[args.label_col])["date"]).drop_duplicates().tolist())
        latest_history_dates = history_dates[-int(best_window["selection_window"]) :]
        history = prediction[prediction["date"].isin(latest_history_dates)].dropna(subset=[args.label_col]).copy()
        best_latest, latest_leaderboard = _choose_best_spec(
            history=history,
            specs=specs,
            label_col=args.label_col,
            score_col=args.score_col,
            top_k=args.top_k,
            max_weight_sum=args.max_weight_sum,
            temperature=args.temperature,
            industry_map_path=args.industry_map_path,
            selection_score=args.selection_score,
            risk_weight=args.risk_weight,
            negative_weight=args.negative_weight,
        )
        latest_daily = latest[latest["date"] == latest["date"].max()].copy()
        submission = build_top_k_submission(
            latest_daily,
            score_col=args.score_col,
            stock_col="stock_id",
            top_k=args.top_k,
            max_weight_sum=args.max_weight_sum,
            strategy=best_latest["strategy"],
            temperature=args.temperature,
            industry_map_path=args.industry_map_path,
            max_per_industry=best_latest["max_per_industry"],
        )
        validate_submission(submission, args.top_k, args.max_weight_sum)
        latest_submission_path = output_dir / "latest_submission.csv"
        submission.to_csv(latest_submission_path, index=False)
    else:
        best_latest = None
        latest_leaderboard = []

    summary = {
        "prediction_path": args.prediction_path,
        "latest_prediction_path": args.latest_prediction_path,
        "selection_score": args.selection_score,
        "risk_weight": args.risk_weight,
        "negative_weight": args.negative_weight,
        "split_holdout": split,
        "walk_forward": walk_forward,
        "selected_window": best_window["selection_window"],
        "latest_selected": best_latest,
        "latest_leaderboard": latest_leaderboard,
        "latest_submission_path": str(latest_submission_path) if latest_submission_path else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        "prediction_path": summary["prediction_path"],
        "latest_prediction_path": summary["latest_prediction_path"],
        "split_holdout": summary["split_holdout"],
        "walk_forward": [
            {key: value for key, value in item.items() if key != "records"}
            for item in summary["walk_forward"]
        ],
        "selected_window": summary["selected_window"],
        "latest_selected": summary["latest_selected"],
        "latest_submission_path": summary["latest_submission_path"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
