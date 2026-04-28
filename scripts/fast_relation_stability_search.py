from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _float_grid(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def _normalize_stock_id(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _safe_z_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    grouped = df.groupby("date")[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return ((df[col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _load_frame(path: str, uncertainty_path: str | None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    df["stock_id"] = df["stock_id"].map(_normalize_stock_id)
    df["date"] = pd.to_datetime(df["date"])
    if uncertainty_path:
        unc = pd.read_csv(uncertainty_path, dtype={"stock_id": str})
        unc["stock_id"] = unc["stock_id"].map(_normalize_stock_id)
        unc["date"] = pd.to_datetime(unc["date"])
        df = df.merge(unc[["stock_id", "date", "score_std"]], on=["stock_id", "date"], how="left")
    if "score_std" not in df.columns:
        df["score_std"] = 0.0
    df["score_std"] = df["score_std"].fillna(0.0)
    df["score_std_z"] = _safe_z_by_date(df, "score_std")
    df["score_std_rank"] = df.groupby("date")["score_std"].rank(pct=True).fillna(0.5)
    daily = df.groupby("date").agg(market_vol=("volatility_20", "mean"), market_beta=("beta_20", "mean"))
    daily["market_vol_z"] = (daily["market_vol"] - daily["market_vol"].mean()) / daily["market_vol"].std()
    daily["market_beta_z"] = (daily["market_beta"] - daily["market_beta"].mean()) / daily["market_beta"].std()
    daily["regime_risk"] = daily["market_vol_z"].fillna(0.0) + 0.25 * daily["market_beta_z"].fillna(0.0)
    df = df.merge(daily[["regime_risk"]], on="date", how="left")
    df["regime_risk"] = df["regime_risk"].fillna(0.0)
    return df.sort_values(["date", "stock_id"]).reset_index(drop=True)


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    y = x / max(float(temp), 1e-6)
    y = y - np.max(y)
    exp = np.exp(y)
    return exp / exp.sum()


def _portfolio_returns(df: pd.DataFrame, score: np.ndarray, label: np.ndarray, top_k: int, temp: float) -> np.ndarray:
    returns: list[float] = []
    start = 0
    dates = df["date"].to_numpy()
    n = len(df)
    while start < n:
        end = start + 1
        while end < n and dates[end] == dates[start]:
            end += 1
        daily_score = score[start:end]
        daily_label = label[start:end]
        if len(daily_score) <= top_k:
            idx = np.argsort(daily_score)[::-1]
        else:
            idx = np.argpartition(daily_score, -top_k)[-top_k:]
            idx = idx[np.argsort(daily_score[idx])[::-1]]
        weights = _softmax(daily_score[idx], temp)
        returns.append(float(np.sum(weights * daily_label[idx])))
        start = end
    return np.asarray(returns, dtype=float)


def _score(df: pd.DataFrame, params: dict[str, float]) -> np.ndarray:
    s = (1.0 - params["alpha"]) * df["score_z"].to_numpy(float)
    s += params["alpha"] * df["industry_rank_score"].to_numpy(float)
    s += params["beta_alpha"] * df["beta_20_rank"].fillna(0.5).to_numpy(float)
    s += params["vol_alpha"] * df["volatility_20_rank"].fillna(0.5).to_numpy(float)
    s += params["liquidity_alpha"] * df["liquidity_20_rank"].fillna(0.5).to_numpy(float)
    s += params["corr_alpha"] * df["corr_peer_score"].fillna(0.0).to_numpy(float)
    s -= params["uncertainty_alpha"] * df["score_std_z"].to_numpy(float)
    s -= params["uncertainty_rank_alpha"] * df["score_std_rank"].to_numpy(float)
    risk_mask = (df["regime_risk"].to_numpy(float) > 0.0).astype(float)
    s += params["regime_risk_alpha"] * risk_mask * df["volatility_20_rank"].fillna(0.5).to_numpy(float)
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast relation parameter search with window stability constraints.")
    parser.add_argument("--prepared-scored-path", required=True)
    parser.add_argument("--uncertainty-path", default="outputs/predictions/master_multiseed/master_multiseed_predictions.csv")
    parser.add_argument("--output-dir", default="outputs/relation_fast_search")
    parser.add_argument("--label-col", default="y_ret_a_stage_round1_open_open")
    parser.add_argument("--alphas", default="-0.45,-0.4,-0.35,-0.3,-0.25")
    parser.add_argument("--beta-alphas", default="0,0.025,0.05,0.075")
    parser.add_argument("--vol-alphas", default="-0.2,-0.175,-0.15,-0.125,-0.1")
    parser.add_argument("--liquidity-alphas", default="0.05,0.075,0.1,0.125,0.15")
    parser.add_argument("--corr-alphas", default="0.075,0.1,0.125")
    parser.add_argument("--uncertainty-alphas", default="0,0.01,0.02")
    parser.add_argument("--uncertainty-rank-alphas", default="0,0.01,0.02")
    parser.add_argument("--regime-risk-alphas", default="0,-0.025,-0.05,-0.075,-0.1")
    parser.add_argument("--baseline-windows", default="0.03279055830196935,0.008530621776006903,0.01821118421265041,0.02244215218813285")
    parser.add_argument("--baseline-mean", type=float, default=0.02209556318009597)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_frame(args.prepared_scored_path, args.uncertainty_path)
    label = df[args.label_col].to_numpy(float)
    unique_dates = sorted(df["date"].unique())
    windows = [20, 40, 60, 90]
    baseline_windows = dict(zip(windows, _float_grid(args.baseline_windows)))
    date_array = df["date"].to_numpy()
    masks = {window: np.isin(date_array, unique_dates[-window:]) for window in windows}
    rows: list[dict] = []
    stable_rows: list[dict] = []

    for values in itertools.product(
        _float_grid(args.alphas),
        _float_grid(args.beta_alphas),
        _float_grid(args.vol_alphas),
        _float_grid(args.liquidity_alphas),
        _float_grid(args.corr_alphas),
        _float_grid(args.uncertainty_alphas),
        _float_grid(args.uncertainty_rank_alphas),
        _float_grid(args.regime_risk_alphas),
    ):
        params = {
            "alpha": values[0],
            "beta_alpha": values[1],
            "vol_alpha": values[2],
            "liquidity_alpha": values[3],
            "corr_alpha": values[4],
            "uncertainty_alpha": values[5],
            "uncertainty_rank_alpha": values[6],
            "regime_risk_alpha": values[7],
        }
        score = _score(df, params)
        daily = _portfolio_returns(df, score, label, args.top_k, args.temperature)
        row = {
            **params,
            "mean_return": float(daily.mean()),
            "std_return": float(daily.std()),
            "num_days": int(len(daily)),
        }
        stable = row["mean_return"] >= args.baseline_mean
        min_diff = row["mean_return"] - args.baseline_mean
        for window in windows:
            win_df = df[masks[window]]
            win_score = score[masks[window]]
            win_label = label[masks[window]]
            win_daily = _portfolio_returns(win_df, win_score, win_label, args.top_k, args.temperature)
            win_mean = float(win_daily.mean())
            diff = win_mean - baseline_windows[window]
            row[f"mean_{window}d"] = win_mean
            row[f"diff_{window}d"] = diff
            stable = stable and diff >= -1e-12
            min_diff = min(min_diff, diff)
        row["min_diff"] = float(min_diff)
        rows.append(row)
        if stable:
            stable_rows.append(row)

    rows.sort(key=lambda item: (item["mean_return"], item["min_diff"]), reverse=True)
    stable_rows.sort(key=lambda item: (item["mean_return"], item["min_diff"]), reverse=True)
    (output_dir / "search_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "stable_leaderboard.json").write_text(
        json.dumps(stable_rows[: args.top_n], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "leaderboard.json").write_text(json.dumps(rows[: args.top_n], ensure_ascii=False, indent=2), encoding="utf-8")

    best = stable_rows[0] if stable_rows else rows[0]
    best_params = {key: best[key] for key in [
        "alpha",
        "beta_alpha",
        "vol_alpha",
        "liquidity_alpha",
        "corr_alpha",
        "uncertainty_alpha",
        "uncertainty_rank_alpha",
        "regime_risk_alpha",
    ]}
    df["score_relation"] = _score(df, best_params)
    df.to_csv(output_dir / "best_scored_predictions.csv", index=False)
    latest = df[df["date"] == df["date"].max()].sort_values("score_relation", ascending=False).head(args.top_k).copy()
    weights = _softmax(latest["score_relation"].to_numpy(float), args.temperature)
    submission = pd.DataFrame({"stock_id": latest["stock_id"].astype(str), "weight": weights})
    submission.to_csv(output_dir / "best_submission.csv", index=False)
    print(json.dumps({"best": best, "num_stable": len(stable_rows)}, ensure_ascii=False, indent=2))
    print(submission.to_string(index=False))


if __name__ == "__main__":
    main()
