from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_price_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("stock_id", group_keys=False)

    out["ret_1"] = g["close"].pct_change(1)
    out["open_to_close"] = (out["close"] / out["open"]) - 1.0
    out["high_to_low"] = (out["high"] / out["low"]) - 1.0
    out["amplitude"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)

    for window in windows:
        out[f"ret_{window}"] = g["close"].pct_change(window)
        out[f"volatility_{window}"] = g["ret_1"].rolling(window).std().reset_index(level=0, drop=True)
        out[f"ma_ratio_{window}"] = (
            out["close"] / g["close"].rolling(window).mean().reset_index(level=0, drop=True)
        ) - 1.0
        out[f"volume_ratio_{window}"] = (
            out["volume"] / g["volume"].rolling(window).mean().reset_index(level=0, drop=True)
        ) - 1.0

    return out


def add_cross_sectional_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        grouped = out.groupby("date")[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        out[f"{col}_cs_z"] = (out[col] - mean) / std
        out[f"{col}_cs_rank"] = grouped.rank(pct=True)
    return out
