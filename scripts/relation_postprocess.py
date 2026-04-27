from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.industry_context import load_industry_map
from src.portfolio.construct import build_top_k_submission, evaluate_portfolio_strategy
from src.training.train_baseline import validate_submission


def _normalize_stock_id(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _safe_zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    grouped = df.groupby("date")[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return ((df[col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_rank_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].rank(pct=True).fillna(0.5)


def _load_relation_feature_frame(raw_dir: str, dates: pd.Series, corr_window: int) -> pd.DataFrame:
    from src.data.io import load_price_data

    price = load_price_data(raw_dir)
    price["stock_id"] = price["stock_id"].map(_normalize_stock_id)
    price["date"] = pd.to_datetime(price["date"])
    price = price.sort_values(["stock_id", "date"]).copy()
    g = price.groupby("stock_id", group_keys=False)
    price["ret_1"] = g["close"].pct_change(1)
    market_ret = price.groupby("date")["ret_1"].mean().rename("market_ret_1")
    price = price.merge(market_ret, on="date", how="left")
    g = price.groupby("stock_id", group_keys=False)
    for window in sorted({20, 60, corr_window}):
        cov = g.apply(lambda x: x["ret_1"].rolling(window).cov(x["market_ret_1"])).reset_index(level=0, drop=True)
        var = price.groupby("stock_id")["market_ret_1"].rolling(window).var().reset_index(level=0, drop=True)
        price[f"beta_{window}"] = cov / var.replace(0, np.nan)
    price["volatility_20"] = g["ret_1"].rolling(20).std().reset_index(level=0, drop=True)
    liquidity_source = price["amount"] if "amount" in price.columns else price["volume"]
    price["liquidity_20"] = g[liquidity_source.name].rolling(20).mean().reset_index(level=0, drop=True)

    needed_start = pd.to_datetime(dates).min() - pd.Timedelta(days=max(100, corr_window * 3))
    needed_dates = set(pd.to_datetime(dates).dt.normalize())
    frame = price[price["date"] >= needed_start].copy()
    frame["is_eval_date"] = frame["date"].dt.normalize().isin(needed_dates)
    feature_cols = ["stock_id", "date", "ret_1", "is_eval_date", "beta_20", "beta_60", "volatility_20", "liquidity_20"]
    return frame[[col for col in feature_cols if col in frame.columns]]


def _add_style_relation_features(scored: pd.DataFrame, raw_dir: str | None, corr_window: int, corr_top_n: int) -> pd.DataFrame:
    if not raw_dir:
        return scored
    feature_frame = _load_relation_feature_frame(raw_dir, scored["date"], corr_window=corr_window)
    latest_features = feature_frame[feature_frame["is_eval_date"]].drop(columns=["is_eval_date", "ret_1"], errors="ignore")
    out = scored.merge(latest_features, on=["stock_id", "date"], how="left")
    for col in ["beta_20", "beta_60", "volatility_20", "liquidity_20"]:
        if col in out.columns:
            out[f"{col}_rank"] = _safe_rank_by_date(out, col)
            out[f"{col}_z"] = _safe_zscore_by_date(out, col)

    if corr_top_n <= 0:
        out["corr_peer_score"] = 0.0
        return out

    history = feature_frame[["stock_id", "date", "ret_1"]].dropna().copy()
    score_lookup = out[["stock_id", "date", "score_z"]].copy()
    peer_rows: list[pd.DataFrame] = []
    for date, daily_scores in score_lookup.groupby("date"):
        hist = history[history["date"] < date].tail(corr_window * max(350, daily_scores["stock_id"].nunique()))
        if hist.empty:
            continue
        pivot = hist.pivot_table(index="date", columns="stock_id", values="ret_1").tail(corr_window)
        stocks = [stock for stock in daily_scores["stock_id"].tolist() if stock in pivot.columns]
        if len(stocks) < 2:
            continue
        corr = pivot[stocks].corr(min_periods=max(10, corr_window // 3)).fillna(0.0)
        score_map = daily_scores.set_index("stock_id")["score_z"]
        values = []
        for stock in stocks:
            peers = corr[stock].drop(index=stock, errors="ignore").sort_values(ascending=False).head(corr_top_n)
            peers = peers[peers > 0]
            if peers.empty:
                values.append({"stock_id": stock, "date": date, "corr_peer_score": 0.0})
                continue
            weights = peers / peers.sum()
            peer_score = float((score_map.reindex(weights.index).fillna(0.0) * weights).sum())
            values.append({"stock_id": stock, "date": date, "corr_peer_score": peer_score})
        peer_rows.append(pd.DataFrame(values))
    if peer_rows:
        peer_df = pd.concat(peer_rows, ignore_index=True)
        out = out.merge(peer_df, on=["stock_id", "date"], how="left")
    out["corr_peer_score"] = out.get("corr_peer_score", 0.0)
    out["corr_peer_score"] = out["corr_peer_score"].fillna(0.0)
    return out


def add_relation_score(
    df: pd.DataFrame,
    industry_map_path: str,
    mode: str,
    alpha: float,
    source_score_col: str = "score",
    raw_dir: str | None = None,
    beta_alpha: float = 0.0,
    vol_alpha: float = 0.0,
    liquidity_alpha: float = 0.0,
    corr_alpha: float = 0.0,
    corr_window: int = 60,
    corr_top_n: int = 8,
) -> pd.DataFrame:
    out = df.copy()
    out["stock_id"] = out["stock_id"].map(_normalize_stock_id)
    out["date"] = pd.to_datetime(out["date"])

    industry_map = load_industry_map(industry_map_path)[["stock_id", "industry_name"]].copy()
    industry_map["stock_id"] = industry_map["stock_id"].astype(str).str.zfill(6)
    out = out.merge(industry_map, on="stock_id", how="left")
    out["industry_name"] = out["industry_name"].fillna("other")

    out["score_z"] = _safe_zscore_by_date(out, source_score_col)
    out["industry_mean_score_z"] = out.groupby(["date", "industry_name"])["score_z"].transform("mean")
    out["industry_rank_score"] = out.groupby(["date", "industry_name"])["score_z"].rank(pct=True).fillna(0.5)
    out["global_rank_score"] = out.groupby("date")["score_z"].rank(pct=True).fillna(0.5)
    out = _add_style_relation_features(out, raw_dir=raw_dir, corr_window=corr_window, corr_top_n=corr_top_n)

    if mode == "smooth":
        out["score_relation"] = (1.0 - alpha) * out["score_z"] + alpha * out["industry_mean_score_z"]
    elif mode == "neutral":
        out["score_relation"] = out["score_z"] - alpha * out["industry_mean_score_z"]
    elif mode == "rank_mix":
        out["score_relation"] = (1.0 - alpha) * out["score_z"] + alpha * out["global_rank_score"]
    elif mode == "rank_ind_mix":
        out["score_relation"] = (1.0 - alpha) * out["score_z"] + alpha * out["industry_rank_score"]
    elif mode == "multi_peer":
        out["score_relation"] = (1.0 - alpha) * out["score_z"] + alpha * out["industry_rank_score"]
        if "beta_20_rank" in out.columns:
            out["score_relation"] += beta_alpha * out["beta_20_rank"].fillna(0.5)
        if "volatility_20_rank" in out.columns:
            out["score_relation"] += vol_alpha * out["volatility_20_rank"].fillna(0.5)
        if "liquidity_20_rank" in out.columns:
            out["score_relation"] += liquidity_alpha * out["liquidity_20_rank"].fillna(0.5)
        out["score_relation"] += corr_alpha * out["corr_peer_score"].fillna(0.0)
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
    parser.add_argument("--mode", choices=["smooth", "neutral", "rank_mix", "rank_ind_mix", "multi_peer"], default="rank_ind_mix")
    parser.add_argument("--alpha", type=float, default=-0.5)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--beta-alpha", type=float, default=0.0)
    parser.add_argument("--vol-alpha", type=float, default=0.0)
    parser.add_argument("--liquidity-alpha", type=float, default=0.0)
    parser.add_argument("--corr-alpha", type=float, default=0.0)
    parser.add_argument("--corr-window", type=int, default=60)
    parser.add_argument("--corr-top-n", type=int, default=8)
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
        raw_dir=args.raw_dir if args.mode == "multi_peer" else None,
        beta_alpha=args.beta_alpha,
        vol_alpha=args.vol_alpha,
        liquidity_alpha=args.liquidity_alpha,
        corr_alpha=args.corr_alpha,
        corr_window=args.corr_window,
        corr_top_n=args.corr_top_n,
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
        "beta_alpha": args.beta_alpha,
        "vol_alpha": args.vol_alpha,
        "liquidity_alpha": args.liquidity_alpha,
        "corr_alpha": args.corr_alpha,
        "corr_window": args.corr_window,
        "corr_top_n": args.corr_top_n,
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
