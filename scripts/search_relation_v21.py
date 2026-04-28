from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.construct import build_top_k_submission, evaluate_portfolio_strategy
from src.training.metrics import precision_at_k, rank_ic, top_k_portfolio_return
from src.training.train_baseline import save_dataframe, validate_submission
from src.training.validation import evaluate_recent_windows


def _float_grid(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def _int_or_none_grid(text: str) -> list[int | None]:
    values: list[int | None] = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        values.append(None if token in {"none", "null", "na"} else int(token))
    return values


def _normalize_stock_id(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _safe_z_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    grouped = df.groupby("date")[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return ((df[col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    scaled = values / temp
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    denom = exp.sum()
    if denom <= 0:
        return np.repeat(1.0 / len(values), len(values))
    return exp / denom


def _add_uncertainty(base: pd.DataFrame, uncertainty_path: str | None) -> pd.DataFrame:
    out = base.copy()
    if uncertainty_path:
        unc = pd.read_csv(uncertainty_path, dtype={"stock_id": str})
        unc["stock_id"] = unc["stock_id"].map(_normalize_stock_id)
        unc["date"] = pd.to_datetime(unc["date"])
        keep = [col for col in ["stock_id", "date", "score_std"] if col in unc.columns]
        out = out.merge(unc[keep], on=["stock_id", "date"], how="left")
    if "score_std" not in out.columns:
        out["score_std"] = 0.0
    out["score_std"] = out["score_std"].fillna(0.0)
    out["score_std_z"] = _safe_z_by_date(out, "score_std")
    out["score_std_rank"] = out.groupby("date")["score_std"].rank(pct=True).fillna(0.5)
    return out


def _add_regime(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    daily = out.groupby("date").agg(
        market_vol=("volatility_20", "mean"),
        market_beta=("beta_20", "mean"),
        market_liquidity=("liquidity_20", "mean"),
        market_corr=("corr_peer_score", "mean"),
    )
    for col in daily.columns:
        daily[f"{col}_z"] = (daily[col] - daily[col].mean()) / (daily[col].std() if daily[col].std() > 1e-8 else 1.0)
    daily["regime_risk"] = daily["market_vol_z"].fillna(0.0) + 0.25 * daily["market_beta_z"].fillna(0.0)
    daily["regime_trend_proxy"] = daily["market_corr_z"].fillna(0.0) + 0.25 * daily["market_liquidity_z"].fillna(0.0)
    out = out.merge(daily[["regime_risk", "regime_trend_proxy"]], on="date", how="left")
    out["regime_risk"] = out["regime_risk"].fillna(0.0)
    out["regime_trend_proxy"] = out["regime_trend_proxy"].fillna(0.0)
    return out


def _build_score(df: pd.DataFrame, params: dict) -> pd.Series:
    score = (1.0 - params["alpha"]) * df["score_z"] + params["alpha"] * df["industry_rank_score"]
    score = score + params["beta_alpha"] * df.get("beta_20_rank", 0.5)
    score = score + params["vol_alpha"] * df.get("volatility_20_rank", 0.5)
    score = score + params["liquidity_alpha"] * df.get("liquidity_20_rank", 0.5)
    score = score + params["corr_alpha"] * df.get("corr_peer_score", 0.0)
    score = score - params["uncertainty_alpha"] * df.get("score_std_z", 0.0)
    score = score - params["uncertainty_rank_alpha"] * df.get("score_std_rank", 0.5)
    if params["regime_risk_alpha"]:
        risk_scale = (df.get("regime_risk", 0.0) > 0).astype(float)
        score = score + params["regime_risk_alpha"] * risk_scale * df.get("volatility_20_rank", 0.5)
    return score


def _evaluate_regime_strategy(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    top_k: int,
    max_weight_sum: float,
    base_temperature: float,
    risk_temperature: float,
    trend_temperature: float,
) -> dict:
    daily_returns: list[float] = []
    for _, group in df[[label_col, score_col, "date", "stock_id", "regime_risk", "regime_trend_proxy"]].dropna().groupby("date"):
        ranked = group.sort_values(score_col, ascending=False).head(top_k).copy()
        if ranked.empty:
            daily_returns.append(0.0)
            continue
        risk = float(ranked["regime_risk"].iloc[0])
        trend = float(ranked["regime_trend_proxy"].iloc[0])
        temp = base_temperature
        if risk > 0.5:
            temp = risk_temperature
        elif trend > 0.5:
            temp = trend_temperature
        weights = _softmax(ranked[score_col].to_numpy(dtype=float), temp) * max_weight_sum
        daily_returns.append(float((weights * ranked[label_col].to_numpy(dtype=float)).sum()))
    return {
        "strategy": "regime_softmax",
        "mean_return": float(np.mean(daily_returns)) if daily_returns else float("nan"),
        "std_return": float(np.std(daily_returns)) if daily_returns else float("nan"),
        "num_days": int(len(daily_returns)),
    }


def _evaluate_candidate(
    df: pd.DataFrame,
    label_col: str,
    strategy: str,
    cap: int | None,
    top_k: int,
    max_weight_sum: float,
    temperature: float,
) -> dict:
    if strategy == "regime_softmax":
        result = _evaluate_regime_strategy(
            df,
            label_col=label_col,
            score_col="score_relation",
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            base_temperature=temperature,
            risk_temperature=0.9,
            trend_temperature=0.5,
        )
    else:
        result = evaluate_portfolio_strategy(
            df,
            label_col=label_col,
            score_col="score_relation",
            strategy=strategy,
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            temperature=temperature,
            max_per_industry=cap,
            industry_map_path="data/raw/hs300_stock_list.csv",
        )
    return result


def _split_metrics(df: pd.DataFrame, label_col: str, strategy: str, cap: int | None, top_k: int, max_weight_sum: float, temperature: float) -> dict:
    dates = sorted(pd.to_datetime(df["date"]).unique())
    mid = len(dates) // 2
    first = df[df["date"].isin(dates[:mid])].copy()
    second = df[df["date"].isin(dates[mid:])].copy()
    first_eval = _evaluate_candidate(first, label_col, strategy, cap, top_k, max_weight_sum, temperature)
    second_eval = _evaluate_candidate(second, label_col, strategy, cap, top_k, max_weight_sum, temperature)
    return {
        "first_half_mean": float(first_eval["mean_return"]),
        "second_half_mean": float(second_eval["mean_return"]),
        "split_min_mean": float(min(first_eval["mean_return"], second_eval["mean_return"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search uncertainty-aware relation 2.1 candidates.")
    parser.add_argument("--prepared-scored-path", required=True)
    parser.add_argument("--uncertainty-path", default="outputs/predictions/master_multiseed/master_multiseed_predictions.csv")
    parser.add_argument("--output-dir", default="outputs/relation_v21_search")
    parser.add_argument("--label-col", default="y_ret_a_stage_round1_open_open")
    parser.add_argument("--alphas", default="-0.4,-0.35,-0.3")
    parser.add_argument("--beta-alphas", default="0,0.05")
    parser.add_argument("--vol-alphas", default="-0.2,-0.15,-0.1")
    parser.add_argument("--liquidity-alphas", default="0.05,0.1,0.15")
    parser.add_argument("--corr-alphas", default="0.05,0.1,0.15")
    parser.add_argument("--uncertainty-alphas", default="0,0.03,0.06,0.1")
    parser.add_argument("--uncertainty-rank-alphas", default="0,0.03,0.06")
    parser.add_argument("--regime-risk-alphas", default="0,-0.05,-0.1")
    parser.add_argument("--strategies", nargs="*", default=["softmax_t0.6", "regime_softmax"])
    parser.add_argument("--caps", default="none")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-weight-sum", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.prepared_scored_path, dtype={"stock_id": str})
    base["stock_id"] = base["stock_id"].map(_normalize_stock_id)
    base["date"] = pd.to_datetime(base["date"])
    base = _add_uncertainty(base, args.uncertainty_path)
    base = _add_regime(base)

    rows: list[dict] = []
    caps = _int_or_none_grid(args.caps)
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
        candidate = base.copy()
        candidate["score_relation"] = _build_score(candidate, params)
        for strategy in args.strategies:
            for cap in caps:
                portfolio_eval = _evaluate_candidate(
                    candidate,
                    label_col=args.label_col,
                    strategy=strategy,
                    cap=cap,
                    top_k=args.top_k,
                    max_weight_sum=args.max_weight_sum,
                    temperature=args.temperature,
                )
                split = _split_metrics(
                    candidate,
                    label_col=args.label_col,
                    strategy=strategy,
                    cap=cap,
                    top_k=args.top_k,
                    max_weight_sum=args.max_weight_sum,
                    temperature=args.temperature,
                )
                row = {
                    **params,
                    "strategy": strategy,
                    "max_per_industry": cap,
                    "mean_return": float(portfolio_eval["mean_return"]),
                    "std_return": float(portfolio_eval.get("std_return", 0.0)),
                    "num_days": int(portfolio_eval["num_days"]),
                    "rank_ic": rank_ic(candidate, args.label_col, "score_relation"),
                    "precision_at_k": precision_at_k(candidate, args.label_col, "score_relation", args.top_k),
                    "top_k_portfolio_return": top_k_portfolio_return(candidate, args.label_col, "score_relation", args.top_k),
                    **split,
                }
                row["robust_score"] = row["mean_return"] + 0.3 * row["split_min_mean"] - 0.05 * row["std_return"]
                rows.append(row)

    rows.sort(key=lambda item: (item["robust_score"], item["mean_return"]), reverse=True)
    (output_dir / "search_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "leaderboard.json").write_text(json.dumps(rows[: args.top_n], ensure_ascii=False, indent=2), encoding="utf-8")

    best = rows[0]
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
    best_scored = base.copy()
    best_scored["score_relation"] = _build_score(best_scored, best_params)
    latest = best_scored[best_scored["date"] == best_scored["date"].max()].copy()
    submission = build_top_k_submission(
        latest,
        score_col="score_relation",
        stock_col="stock_id",
        top_k=args.top_k,
        max_weight_sum=args.max_weight_sum,
        strategy="softmax_t0.6" if best["strategy"] == "regime_softmax" else best["strategy"],
        temperature=args.temperature,
        max_per_industry=best["max_per_industry"],
    )
    validate_submission(submission, args.top_k, args.max_weight_sum)
    report = {
        "best": best,
        "windows": evaluate_recent_windows(
            best_scored,
            label_col=args.label_col,
            score_col="score_relation",
            top_k=args.top_k,
            max_weight_sum=args.max_weight_sum,
            strategy="softmax_t0.6" if best["strategy"] == "regime_softmax" else best["strategy"],
            temperature=args.temperature,
            windows=[20, 40, 60, 90],
        ),
    }
    save_dataframe(best_scored, output_dir / "best_scored_predictions.csv")
    save_dataframe(submission, output_dir / "best_submission.csv")
    (output_dir / "best_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows[: args.top_n], ensure_ascii=False, indent=2))
    print(submission.to_string(index=False))


if __name__ == "__main__":
    main()
