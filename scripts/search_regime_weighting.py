from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _float_grid(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    scaled = x / max(float(temperature), 1e-6)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / exp.sum()


def _daily_returns(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    top_k: int,
    base_temp: float,
    high_risk_temp: float,
    low_risk_temp: float,
    risk_high: float,
    risk_low: float,
) -> tuple[np.ndarray, list[dict]]:
    returns: list[float] = []
    latest_rows: list[dict] = []
    for date, group in df.groupby("date", sort=True):
        ranked = group.sort_values(score_col, ascending=False).head(top_k).copy()
        if ranked.empty:
            returns.append(0.0)
            continue
        risk = float(ranked["regime_risk"].iloc[0]) if "regime_risk" in ranked.columns else 0.0
        temp = base_temp
        if risk >= risk_high:
            temp = high_risk_temp
        elif risk <= risk_low:
            temp = low_risk_temp
        weights = _softmax(ranked[score_col].to_numpy(dtype=float), temp)
        returns.append(float((weights * ranked[label_col].to_numpy(dtype=float)).sum()))
        if date == df["date"].max():
            for (_, row), weight in zip(ranked.iterrows(), weights):
                latest_rows.append({"stock_id": str(row["stock_id"]).zfill(6), "weight": float(weight)})
    return np.asarray(returns, dtype=float), latest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Search regime-aware softmax temperatures for a fixed score.")
    parser.add_argument("--scored-path", required=True)
    parser.add_argument("--output-dir", default="outputs/regime_weighting_search")
    parser.add_argument("--label-col", default="y_ret_a_stage_round1_open_open")
    parser.add_argument("--score-col", default="score_relation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--base-temps", default="0.55,0.6,0.65,0.7")
    parser.add_argument("--high-risk-temps", default="0.7,0.8,0.9,1.0")
    parser.add_argument("--low-risk-temps", default="0.45,0.5,0.55,0.6")
    parser.add_argument("--risk-highs", default="-0.5,0,0.5,1.0")
    parser.add_argument("--risk-lows", default="-1.0,-0.5,0")
    parser.add_argument("--baseline-mean", type=float, default=0.022160136791895574)
    parser.add_argument("--baseline-windows", default="0.03286045452471061,0.0085537555697401,0.018287859325639203,0.022509880174329443")
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.scored_path, dtype={"stock_id": str})
    df["date"] = pd.to_datetime(df["date"])
    windows = [20, 40, 60, 90]
    baseline_windows = dict(zip(windows, _float_grid(args.baseline_windows)))
    unique_dates = sorted(df["date"].unique())
    rows: list[dict] = []

    for base_temp, high_temp, low_temp, risk_high, risk_low in itertools.product(
        _float_grid(args.base_temps),
        _float_grid(args.high_risk_temps),
        _float_grid(args.low_risk_temps),
        _float_grid(args.risk_highs),
        _float_grid(args.risk_lows),
    ):
        if risk_low >= risk_high:
            continue
        daily, latest_rows = _daily_returns(
            df,
            label_col=args.label_col,
            score_col=args.score_col,
            top_k=args.top_k,
            base_temp=base_temp,
            high_risk_temp=high_temp,
            low_risk_temp=low_temp,
            risk_high=risk_high,
            risk_low=risk_low,
        )
        row = {
            "base_temp": base_temp,
            "high_risk_temp": high_temp,
            "low_risk_temp": low_temp,
            "risk_high": risk_high,
            "risk_low": risk_low,
            "mean_return": float(daily.mean()),
            "std_return": float(daily.std()),
            "num_days": int(len(daily)),
            "latest_submission": latest_rows,
        }
        stable = row["mean_return"] >= args.baseline_mean
        min_diff = row["mean_return"] - args.baseline_mean
        for window in windows:
            win_df = df[df["date"].isin(unique_dates[-window:])]
            win_daily, _ = _daily_returns(
                win_df,
                label_col=args.label_col,
                score_col=args.score_col,
                top_k=args.top_k,
                base_temp=base_temp,
                high_risk_temp=high_temp,
                low_risk_temp=low_temp,
                risk_high=risk_high,
                risk_low=risk_low,
            )
            win_mean = float(win_daily.mean())
            diff = win_mean - baseline_windows[window]
            row[f"mean_{window}d"] = win_mean
            row[f"diff_{window}d"] = diff
            stable = stable and diff >= -1e-12
            min_diff = min(min_diff, diff)
        row["min_diff"] = float(min_diff)
        row["stable"] = bool(stable)
        rows.append(row)

    rows.sort(key=lambda item: (item["stable"], item["mean_return"], item["min_diff"]), reverse=True)
    stable_rows = [row for row in rows if row["stable"]]
    (output_dir / "leaderboard.json").write_text(json.dumps(rows[: args.top_n], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "stable_leaderboard.json").write_text(json.dumps(stable_rows[: args.top_n], ensure_ascii=False, indent=2), encoding="utf-8")
    best = stable_rows[0] if stable_rows else rows[0]
    pd.DataFrame(best["latest_submission"]).to_csv(output_dir / "best_submission.csv", index=False)
    (output_dir / "best_report.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": {k: v for k, v in best.items() if k != "latest_submission"}, "num_stable": len(stable_rows)}, ensure_ascii=False, indent=2))
    print(pd.DataFrame(best["latest_submission"]).to_string(index=False))


if __name__ == "__main__":
    main()
