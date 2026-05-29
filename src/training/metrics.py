from __future__ import annotations

import math

import pandas as pd


def rank_ic(df: pd.DataFrame, label_col: str, score_col: str) -> float:
    daily_values: list[float] = []
    for _, group in df[[label_col, score_col, "date"]].dropna().groupby("date"):
        if len(group) < 2:
            continue
        corr = group[label_col].corr(group[score_col], method="spearman")
        if pd.notna(corr):
            daily_values.append(float(corr))
    if not daily_values:
        return math.nan
    return float(sum(daily_values) / len(daily_values))


def precision_at_k(df: pd.DataFrame, label_col: str, score_col: str, k: int) -> float:
    hits: list[float] = []
    for _, group in df[[label_col, score_col, "date"]].dropna().groupby("date"):
        if len(group) < k:
            continue
        pred_top = set(group.nlargest(k, score_col).index)
        true_top = set(group.nlargest(k, label_col).index)
        hits.append(len(pred_top & true_top) / k)
    if not hits:
        return math.nan
    return float(sum(hits) / len(hits))


def top_hit_rate(df: pd.DataFrame, label_col: str, score_col: str, true_top_k: int = 1, pred_top_k: int = 1) -> float:
    hits: list[float] = []
    for _, group in df[[label_col, score_col, "date"]].dropna().groupby("date"):
        if len(group) < max(true_top_k, pred_top_k):
            continue
        true_top = set(group.nlargest(true_top_k, label_col).index)
        pred_top = set(group.nlargest(pred_top_k, score_col).index)
        hits.append(1.0 if true_top & pred_top else 0.0)
    if not hits:
        return math.nan
    return float(sum(hits) / len(hits))


def top1_margin_z(df: pd.DataFrame, score_col: str) -> float:
    margins: list[float] = []
    for _, group in df[[score_col, "date"]].dropna().groupby("date"):
        if len(group) < 2:
            continue
        scores = group[score_col].sort_values(ascending=False).to_numpy(dtype=float)
        std = float(scores.std())
        scale = std if std > 1e-8 else 1.0
        margins.append(float((scores[0] - scores[1]) / scale))
    if not margins:
        return math.nan
    return float(sum(margins) / len(margins))


def top1_portfolio_return(df: pd.DataFrame, label_col: str, score_col: str) -> float:
    returns: list[float] = []
    for _, group in df[[label_col, score_col, "date"]].dropna().groupby("date"):
        if len(group) < 1:
            continue
        top = group.nlargest(1, score_col)
        returns.append(float(top[label_col].iloc[0]))
    if not returns:
        return math.nan
    return float(sum(returns) / len(returns))


def top_k_portfolio_return(df: pd.DataFrame, label_col: str, score_col: str, k: int) -> float:
    returns: list[float] = []
    for _, group in df[[label_col, score_col, "date"]].dropna().groupby("date"):
        if len(group) < k:
            continue
        top = group.nlargest(k, score_col)
        returns.append(float(top[label_col].mean()))
    if not returns:
        return math.nan
    return float(sum(returns) / len(returns))
