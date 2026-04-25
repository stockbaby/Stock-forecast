from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.portfolio.construct import evaluate_portfolio_strategy
from src.training.metrics import precision_at_k, rank_ic, top_k_portfolio_return


@dataclass
class WindowMetric:
    window_days: int
    start_date: str
    end_date: str
    rank_ic: float
    precision_at_k: float
    top_k_portfolio_return: float
    strategy_mean_return: float


def evaluate_recent_windows(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    top_k: int,
    max_weight_sum: float,
    strategy: str,
    temperature: float = 1.0,
    windows: list[int] | tuple[int, ...] = (20, 40, 60, 90),
) -> list[dict]:
    out: list[dict] = []
    unique_dates = sorted(pd.to_datetime(df["date"]).dropna().unique())
    for window_days in windows:
        if len(unique_dates) < window_days:
            continue
        start_date = unique_dates[-window_days]
        window_df = df[pd.to_datetime(df["date"]) >= start_date].copy()
        if window_df.empty:
            continue
        portfolio_eval = evaluate_portfolio_strategy(
            window_df,
            label_col=label_col,
            score_col=score_col,
            strategy=strategy,
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            temperature=temperature,
        )
        out.append(
            {
                "window_days": int(window_days),
                "start_date": str(pd.Timestamp(start_date).date()),
                "end_date": str(pd.Timestamp(window_df["date"].max()).date()),
                "rank_ic": rank_ic(window_df, label_col, score_col),
                "precision_at_k": precision_at_k(window_df, label_col, score_col, top_k),
                "top_k_portfolio_return": top_k_portfolio_return(window_df, label_col, score_col, top_k),
                "strategy_mean_return": portfolio_eval["mean_return"],
            }
        )
    return out
