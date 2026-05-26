from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.industry_context import load_industry_map

PORTFOLIO_CONTEXT_COLUMNS = [
    "score_std",
    "regime_trend",
    "regime_vol_ratio",
    "regime_drawdown",
    "regime_score",
    "regime_is_trending",
    "regime_is_high_vol",
    "index_ret_5",
    "index_ret_10",
    "index_ret_20",
    "index_drawdown_20",
    "ret_5",
    "ret_20",
    "volume_ratio_5",
    "volume_ratio_20",
    "amount_ratio_5",
    "amount_ratio_20",
    "volume_breakout_5",
    "volume_breakout_20",
    "industry_name",
    "industry_id",
    "industry_collective_momentum",
    "industry_collective_volume_confirm",
]


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


def _first_numeric(candidates: pd.DataFrame, columns: list[str], default: float = 0.0, mode: str = "first") -> float:
    for col in columns:
        if col not in candidates.columns:
            continue
        values = pd.to_numeric(candidates[col], errors="coerce").dropna()
        if values.empty:
            continue
        if mode == "mean":
            return float(values.mean())
        return float(values.iloc[0])
    return default


def _market_state_inputs(candidates: pd.DataFrame) -> dict[str, float]:
    high_vol = _first_numeric(candidates, ["regime_is_high_vol"], default=0.0)
    vol_ratio = _first_numeric(candidates, ["regime_vol_ratio"], default=1.0)
    if "regime_is_high_vol" not in candidates.columns and vol_ratio:
        high_vol = float(np.clip(vol_ratio - 1.0, 0.0, 2.0))

    drawdown = _first_numeric(candidates, ["regime_drawdown", "index_drawdown_20"], default=0.0)
    drawdown_risk = float(np.clip(abs(min(drawdown, 0.0)) * 5.0, 0.0, 1.5))

    trend = _first_numeric(candidates, ["regime_trend", "regime_score"], default=0.0)
    index_momentum = _first_numeric(candidates, ["index_ret_20", "index_ret_10", "index_ret_5"], default=0.0)
    stock_momentum = _first_numeric(candidates, ["ret_20", "ret_5"], default=0.0, mode="mean")
    momentum = float(np.tanh(3.0 * trend + 5.0 * index_momentum + 2.0 * stock_momentum))

    volume_breakout = _first_numeric(
        candidates,
        ["volume_breakout_20", "volume_breakout_5", "volume_ratio_20", "volume_ratio_5", "amount_ratio_20", "amount_ratio_5"],
        default=1.0,
        mode="mean",
    )
    volume_pressure = float(np.clip(volume_breakout - 1.0, 0.0, 2.0))

    industry_concentration = 0.0
    for col in ["industry_name", "industry_id"]:
        if col in candidates.columns and len(candidates) > 0:
            concentration = candidates[col].astype(str).value_counts(normalize=True).max()
            industry_concentration = float(concentration)
            break

    risk = float(np.clip(0.45 * high_vol + 0.35 * drawdown_risk + 0.15 * volume_pressure + 0.20 * industry_concentration - 0.30 * max(momentum, 0.0), 0.0, 2.0))
    return {
        "market_risk": risk,
        "market_momentum": momentum,
        "market_high_vol": float(high_vol),
        "market_drawdown_risk": drawdown_risk,
        "volume_pressure": volume_pressure,
        "industry_concentration": industry_concentration,
    }


def _candidate_frame(
    latest_df: pd.DataFrame,
    score_col: str,
    stock_col: str,
    top_k: int,
    min_score: float | None = None,
    min_score_z: float | None = None,
    industry_map_path: str | None = None,
    max_per_industry: int | None = None,
) -> pd.DataFrame:
    ranked = latest_df.sort_values(score_col, ascending=False).copy()
    if min_score is not None:
        ranked = ranked[ranked[score_col] >= min_score]
    if min_score_z is not None and not ranked.empty:
        scores = ranked[score_col].to_numpy(dtype=float)
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        std = std if std > 1e-8 else 1.0
        ranked["score_z"] = (ranked[score_col] - mean) / std
        ranked = ranked[ranked["score_z"] >= min_score_z]
    if industry_map_path and max_per_industry is not None and Path(industry_map_path).exists():
        industry_map = load_industry_map(industry_map_path)[["stock_id", "industry_name"]].copy()
        industry_map["stock_id"] = industry_map["stock_id"].map(_normalize_stock_id)
        ranked = ranked.copy()
        ranked[stock_col] = ranked[stock_col].map(_normalize_stock_id)
        ranked = ranked.merge(industry_map, left_on=stock_col, right_on="stock_id", how="left")
        ranked["industry_name"] = ranked["industry_name"].fillna("other")
        selected_parts: list[pd.DataFrame] = []
        counts: dict[str, int] = {}
        for _, row in ranked.iterrows():
            industry = str(row["industry_name"])
            if counts.get(industry, 0) >= max_per_industry:
                continue
            selected_parts.append(pd.DataFrame([row]))
            counts[industry] = counts.get(industry, 0) + 1
            if len(selected_parts) >= top_k:
                break
        if selected_parts:
            ranked = pd.concat(selected_parts, ignore_index=True)
        else:
            ranked = ranked.head(0)
    top = ranked.head(top_k).copy()
    if top.empty:
        return pd.DataFrame(columns=[stock_col, score_col])
    top[stock_col] = top[stock_col].map(_normalize_stock_id)
    return top


def _parse_strategy_spec(
    strategy: str,
    default_temperature: float,
) -> tuple[str, float, float | None, float | None]:
    base_strategy = strategy
    temperature = default_temperature
    min_score: float | None = None
    min_score_z: float | None = None

    temp_match = re.search(r"_t(-?\d+(?:\.\d+)?)", strategy)
    if temp_match:
        temperature = float(temp_match.group(1))
        base_strategy = base_strategy.replace(temp_match.group(0), "")

    thr_match = re.search(r"_thr(-?\d+(?:\.\d+)?)", base_strategy)
    if thr_match:
        min_score = float(thr_match.group(1))
        base_strategy = base_strategy.replace(thr_match.group(0), "")

    thrz_match = re.search(r"_zthr(-?\d+(?:\.\d+)?)", base_strategy)
    if thrz_match:
        min_score_z = float(thrz_match.group(1))
        base_strategy = base_strategy.replace(thrz_match.group(0), "")

    return base_strategy, temperature, min_score, min_score_z


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

    if strategy == "top1_weight":
        weights = np.zeros(len(candidates), dtype=float)
        weights[0] = max_weight_sum
        return weights

    if strategy == "top2_softmax":
        subset = min(2, len(candidates))
        weights = np.zeros(len(candidates), dtype=float)
        weights[:subset] = _softmax(scores[:subset], temperature=temperature) * max_weight_sum
        return weights

    if strategy == "top3_softmax":
        subset = min(3, len(candidates))
        weights = np.zeros(len(candidates), dtype=float)
        weights[:subset] = _softmax(scores[:subset], temperature=temperature) * max_weight_sum
        return weights

    if strategy == "confidence_topk":
        if len(candidates) == 1:
            return np.array([max_weight_sum], dtype=float)
        score_std = float(np.std(scores))
        scale = score_std if score_std > 1e-8 else 1.0
        margin_12 = float((scores[0] - scores[1]) / scale)
        top_strength = float((scores[0] - np.mean(scores)) / scale)
        uncertainty_penalty = 0.0
        if "score_std" in candidates.columns:
            std_values = candidates["score_std"].to_numpy(dtype=float)
            std_scale = float(np.std(std_values))
            if std_scale > 1e-8:
                uncertainty_penalty = float((std_values[0] - np.mean(std_values)) / std_scale)
        confidence = 0.55 * margin_12 + 0.30 * top_strength - 0.25 * uncertainty_penalty
        top1_weight = max_weight_sum * float(np.clip(0.55 + 0.18 * confidence, 0.35, 0.95))
        weights = np.zeros(len(candidates), dtype=float)
        weights[0] = top1_weight
        if len(candidates) > 1 and top1_weight < max_weight_sum:
            tail_count = min(4, len(candidates) - 1)
            weights[1 : 1 + tail_count] = (
                _softmax(scores[1 : 1 + tail_count], temperature=temperature) * (max_weight_sum - top1_weight)
            )
        return weights

    if strategy == "dynamic_risk_budget":
        if len(candidates) == 1:
            return np.array([max_weight_sum], dtype=float)
        score_scale = float(np.std(scores))
        score_scale = score_scale if score_scale > 1e-8 else 1.0
        margin_12 = float((scores[0] - scores[1]) / score_scale)
        top_strength = float((scores[0] - np.mean(scores)) / score_scale)
        uncertainty_z = 0.0
        if "score_std" in candidates.columns:
            std_values = candidates["score_std"].to_numpy(dtype=float)
            std_scale = float(np.std(std_values))
            if std_scale > 1e-8:
                uncertainty_z = float((std_values[0] - np.mean(std_values)) / std_scale)

        market_state = _market_state_inputs(candidates)
        market_risk = market_state["market_risk"]
        market_momentum = market_state["market_momentum"]
        strong_signal = margin_12 >= 1.0 and top_strength >= 0.75 and uncertainty_z <= 0.5
        weak_signal = margin_12 <= 0.35 or uncertainty_z >= 1.0 or market_risk >= 1.0

        if strong_signal and market_risk < 0.65 and market_momentum > -0.25:
            weights = np.zeros(len(candidates), dtype=float)
            weights[0] = max_weight_sum
            return weights
        if weak_signal:
            subset = min(2, len(candidates))
            weights = np.zeros(len(candidates), dtype=float)
            weights[:subset] = _softmax(scores[:subset], temperature=temperature) * max_weight_sum
            return weights

        confidence = 0.60 * margin_12 + 0.25 * top_strength - 0.35 * uncertainty_z - 0.30 * market_risk + 0.15 * market_momentum
        top1_weight = max_weight_sum * float(np.clip(0.50 + 0.16 * confidence, 0.35, 0.88))
        weights = np.zeros(len(candidates), dtype=float)
        weights[0] = top1_weight
        tail_count = min(4, len(candidates) - 1)
        weights[1 : 1 + tail_count] = (
            _softmax(scores[1 : 1 + tail_count], temperature=temperature) * (max_weight_sum - top1_weight)
        )
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


def _max_drawdown(returns: list[float]) -> float:
    if not returns:
        return math.nan
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(running_peak, 1e-12) - 1.0
    return float(np.min(drawdowns))


def build_top_k_submission(
    latest_df: pd.DataFrame,
    score_col: str = "score",
    stock_col: str = "stock_id",
    top_k: int = 5,
    max_weight_sum: float = 1.0,
    strategy: str = "proportional_positive",
    temperature: float = 1.0,
    industry_map_path: str | None = None,
    max_per_industry: int | None = None,
) -> pd.DataFrame:
    base_strategy, effective_temperature, min_score, min_score_z = _parse_strategy_spec(strategy, temperature)
    candidates = _candidate_frame(
        latest_df,
        score_col=score_col,
        stock_col=stock_col,
        top_k=top_k,
        min_score=min_score,
        min_score_z=min_score_z,
        industry_map_path=industry_map_path,
        max_per_industry=max_per_industry,
    )
    if candidates.empty:
        return pd.DataFrame(columns=["stock_id", "weight"])

    weights = _weights_from_strategy(
        candidates,
        score_col=score_col,
        strategy=base_strategy,
        max_weight_sum=max_weight_sum,
        temperature=effective_temperature,
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
    industry_map_path: str | None = None,
    max_per_industry: int | None = None,
) -> dict:
    daily_returns: list[float] = []
    required_cols = [label_col, score_col, "date", "stock_id"]
    optional_cols = [col for col in PORTFOLIO_CONTEXT_COLUMNS if col in df.columns and col not in required_cols]
    eval_cols = [*required_cols, *optional_cols]

    for _, group in df[eval_cols].dropna(subset=[label_col, score_col, "date", "stock_id"]).groupby("date"):
        submission = build_top_k_submission(
            group,
            score_col=score_col,
            stock_col="stock_id",
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            strategy=strategy,
            temperature=temperature,
            industry_map_path=industry_map_path,
            max_per_industry=max_per_industry,
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

    return_array = np.asarray(daily_returns, dtype=float)
    return {
        "strategy": strategy,
        "mean_return": float(np.mean(return_array)),
        "std_return": float(np.std(return_array)),
        "p05_return": float(np.quantile(return_array, 0.05)),
        "negative_rate": float(np.mean(return_array < 0.0)),
        "max_drawdown": _max_drawdown(daily_returns),
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
    industry_map_path: str | None = None,
    max_per_industry: int | None = None,
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
            industry_map_path=industry_map_path,
            max_per_industry=max_per_industry,
        )
        for strategy in strategies
    ]
    best = max(results, key=lambda item: item["mean_return"])
    return best["strategy"], results
