from __future__ import annotations

import numpy as np
import pandas as pd


def build_top_k_submission(
    latest_df: pd.DataFrame,
    score_col: str = "score",
    stock_col: str = "stock_id",
    top_k: int = 5,
    max_weight_sum: float = 1.0,
) -> pd.DataFrame:
    top = latest_df.sort_values(score_col, ascending=False).head(top_k).copy()
    if top.empty:
        return pd.DataFrame(columns=["stock_id", "weight"])

    positive_scores = np.clip(top[score_col].to_numpy(dtype=float), a_min=0.0, a_max=None)
    if positive_scores.sum() == 0:
        weights = np.repeat(max_weight_sum / len(top), len(top))
    else:
        weights = positive_scores / positive_scores.sum() * max_weight_sum

    submission = pd.DataFrame({"stock_id": top[stock_col].astype(str), "weight": weights})
    return submission
