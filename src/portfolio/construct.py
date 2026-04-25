from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _normalize_stock_id(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if len(x) == 0:
        return x
    temp = max(float(temperature), 1e-6)
    scaled = x / temp
    scaled = scaled - np.max(scaled)
    exp_x = np.exp(scaled)
    denom = exp_x.sum()
    if denom <= 0:
        return np.repeat(1.0 / len(x), len(x))
    return exp_x / denom


def _candidate_frame(
    latest_df: pd.DataFrame,
    score_col: str,
    stock_col: str,
    top_k: int,
) -> pd.DataFrame:
    top = latest_df.sort_values(score_col, ascending=False).head(top_k).copy()
    if top.empty:
        return pd.DataFrame(columns=[stock_col, score_col])
    top[stock_col] = top[stock_col].map(_normalize_stock_id)
    return top


def _weights_from_strategy(
    candidates: pd.DataFrame,
    score_col: str,
    strategy: str,
    max_weight_sum: float,
    temperature: float = 1.0,
) -> np.ndarray:
    scores = candidates[score_col].to_numpy(dtype=float)

    if strategy == "equal_weight":
        return np.repeat(max_weight_sum / len(candidates), len(candidates))

    if strategy == "top3_equal":
        subset = min(3, len(candidates))
        weights = np.zeros(len(candidates), dtype=float)
        weights[:subset] = max_weight_sum / subset
        return weights

    if strategy == "softmax":
        return _softmax(scores, temperature=temperature) * max_weight_sum

    if strategy == "positive_only":
        positive = np.clip(scores, a_min=0.0, a_max=None)
        if positive.sum() <= 0:
            return np.zeros(len(candidates), dtype=float)
        return positive / positive.sum() * max_weight_sum

    if strategy == "positive_softmax":
        mask = scores > 0
        if not mask.any():
            return np.zeros(len(candidates), dtype=float)
        weights = np.zeros(len(candidates), dtype=float)
        weights[mask] = _softmax(scores[mask], temperature=temperature) * max_weight_sum
        return weights

    if strategy == "proportional_positive":
        positive_scores = np.clip(scores, a_min=0.0, a_max=None)
        if positive_scores.sum() == 0:
            return np.repeat(max_weight_sum / len(candidates), len(candidates))
        return positive_scores / positive_scores.sum() * max_weight_sum

    raise ValueError(f"Unsupported portfolio strategy: {strategy}")


def build_top_k_submission(
    latest_df: pd.DataFrame,
    score_col: str = "score",
    stock_col: str = "stock_id",
    top_k: int = 5,
    max_weight_sum: float = 1.0,
    strategy: str = "proportional_positive",
    temperature: float = 1.0,
) -> pd.DataFrame:
    candidates = _candidate_frame(latest_df, score_col=score_col, stock_col=stock_col, top_k=top_k)
    if candidates.empty:
        return pd.DataFrame(columns=["stock_id", "weight"])

    weights = _weights_from_strategy(
        candidates,
        score_col=score_col,
        strategy=strategy,
        max_weight_sum=max_weight_sum,
        temperature=temperature,
    )
    submission = pd.DataFrame(
        {
            "stock_id": candidates[stock_col].astype(str),
            "weight": weights,
        }
    )
    submission = submission[submission["weight"] > 0].copy()
    return submission


def evaluate_portfolio_strategy(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    strategy: str,
    top_k: int,
    max_weight_sum: float,
    temperature: float = 1.0,
) -> dict:
    daily_returns: list[float] = []

    for _, group in df[[label_col, score_col, "date", "stock_id"]].dropna().groupby("date"):
        submission = build_top_k_submission(
            group,
            score_col=score_col,
            stock_col="stock_id",
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            strategy=strategy,
            temperature=temperature,
        )
        if submission.empty:
            daily_returns.append(0.0)
            continue

        merged = submission.merge(
            group[["stock_id", label_col]].assign(stock_id=lambda x: x["stock_id"].map(_normalize_stock_id)),
            on="stock_id",
            how="left",
        )
        merged["weighted_return"] = merged["weight"] * merged[label_col]
        daily_returns.append(float(merged["weighted_return"].sum()))

    if not daily_returns:
        return {"strategy": strategy, "mean_return": math.nan, "num_days": 0}

    return {
        "strategy": strategy,
        "mean_return": float(np.mean(daily_returns)),
        "std_return": float(np.std(daily_returns)),
        "num_days": int(len(daily_returns)),
    }


def select_best_portfolio_strategy(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    strategies: list[str],
    top_k: int,
    max_weight_sum: float,
    temperature: float = 1.0,
) -> tuple[str, list[dict]]:
    results = [
        evaluate_portfolio_strategy(
            df,
            label_col=label_col,
            score_col=score_col,
            strategy=strategy,
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            temperature=temperature,
        )
        for strategy in strategies
    ]
    best = max(results, key=lambda item: item["mean_return"])
    return best["strategy"], results
