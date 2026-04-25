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
