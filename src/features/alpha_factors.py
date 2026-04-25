from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_price_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("stock_id", group_keys=False)
    prev_close = g["close"].shift(1)
    intraday_range = (out["high"] - out["low"]).replace(0, np.nan)

    out["ret_1"] = g["close"].pct_change(1)
    out["open_to_close"] = (out["close"] / out["open"]) - 1.0
    out["high_to_low"] = (out["high"] / out["low"]) - 1.0
    out["amplitude"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["gap_open"] = (out["open"] / prev_close) - 1.0
    out["close_vs_prev_close"] = (out["close"] / prev_close) - 1.0
    out["body_ratio"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    out["upper_shadow_ratio"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["close"].replace(0, np.nan)
    out["lower_shadow_ratio"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["close"].replace(0, np.nan)
    out["close_location"] = (out["close"] - out["low"]) / intraday_range
    out["open_location"] = (out["open"] - out["low"]) / intraday_range

    if "amount" in out.columns:
        out["amount_per_volume"] = out["amount"] / out["volume"].replace(0, np.nan)
    if "turnover_rate_pct" in out.columns:
        out["turnover_rate"] = out["turnover_rate_pct"] / 100.0
    if "pct_chg" in out.columns:
        out["pct_chg_decimal"] = out["pct_chg"] / 100.0
    if "amplitude_pct" in out.columns:
        out["amplitude_pct_decimal"] = out["amplitude_pct"] / 100.0

    ema_12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema_26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    out["ema_12_ratio"] = (out["close"] / ema_12) - 1.0
    out["ema_26_ratio"] = (out["close"] / ema_26) - 1.0
    out["macd_line"] = (ema_12 - ema_26) / out["close"].replace(0, np.nan)
    out["signal_line"] = out.groupby("stock_id")["macd_line"].transform(
        lambda s: s.ewm(span=9, adjust=False).mean()
    )
    out["macd_hist"] = out["macd_line"] - out["signal_line"]

    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(out["stock_id"]).transform(lambda s: s.rolling(14).mean())
    avg_loss = loss.groupby(out["stock_id"]).transform(lambda s: s.rolling(14).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 1 - (1 / (1 + rs))

    for window in windows:
        out[f"ret_{window}"] = g["close"].pct_change(window)
        out[f"volatility_{window}"] = g["ret_1"].rolling(window).std().reset_index(level=0, drop=True)
        out[f"ret_mean_{window}"] = g["ret_1"].rolling(window).mean().reset_index(level=0, drop=True)
        out[f"ret_std_{window}"] = g["ret_1"].rolling(window).std().reset_index(level=0, drop=True)
        out[f"ret_skew_{window}"] = g["ret_1"].rolling(window).skew().reset_index(level=0, drop=True)
        out[f"ma_ratio_{window}"] = (
            out["close"] / g["close"].rolling(window).mean().reset_index(level=0, drop=True)
        ) - 1.0
        out[f"volume_ratio_{window}"] = (
            out["volume"] / g["volume"].rolling(window).mean().reset_index(level=0, drop=True)
        ) - 1.0
        out[f"high_ratio_{window}"] = (
            out["high"] / g["high"].rolling(window).max().reset_index(level=0, drop=True)
        ) - 1.0
        out[f"low_ratio_{window}"] = (
            out["low"] / g["low"].rolling(window).min().reset_index(level=0, drop=True)
        ) - 1.0
        out[f"price_position_{window}"] = (
            (out["close"] - g["low"].rolling(window).min().reset_index(level=0, drop=True))
            / (
                g["high"].rolling(window).max().reset_index(level=0, drop=True)
                - g["low"].rolling(window).min().reset_index(level=0, drop=True)
            ).replace(0, np.nan)
        )
        out[f"ret_to_vol_{window}"] = out[f"ret_{window}"] / out[f"volatility_{window}"].replace(0, np.nan)

        if "amount" in out.columns:
            out[f"amount_ratio_{window}"] = (
                out["amount"] / g["amount"].rolling(window).mean().reset_index(level=0, drop=True)
            ) - 1.0
        if "turnover_rate" in out.columns:
            out[f"turnover_ratio_{window}"] = (
                out["turnover_rate"] / g["turnover_rate"].rolling(window).mean().reset_index(level=0, drop=True)
            ) - 1.0
        if "pct_chg_decimal" in out.columns:
            out[f"pct_chg_mean_{window}"] = (
                g["pct_chg_decimal"].rolling(window).mean().reset_index(level=0, drop=True)
            )
        out[f"volume_volatility_{window}"] = g["volume"].rolling(window).std().reset_index(level=0, drop=True) / g[
            "volume"
        ].rolling(window).mean().reset_index(level=0, drop=True).replace(0, np.nan)

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
