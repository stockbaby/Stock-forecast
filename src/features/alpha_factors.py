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
        rolling_high = g["high"].rolling(window).max().reset_index(level=0, drop=True)
        rolling_volume = g["volume"].rolling(window).mean().reset_index(level=0, drop=True)
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
        out[f"high_ratio_{window}"] = (out["high"] / rolling_high) - 1.0
        out[f"close_to_high_{window}"] = (out["close"] / rolling_high) - 1.0
        out[f"breakout_strength_{window}"] = np.maximum(out[f"close_to_high_{window}"], 0.0)
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
        out[f"volume_volatility_{window}"] = (
            g["volume"].rolling(window).std().reset_index(level=0, drop=True) / rolling_volume.replace(0, np.nan)
        )
        out[f"volume_breakout_{window}"] = out[f"volume_ratio_{window}"] * out["ret_1"].clip(lower=0.0)
        out[f"momentum_accel_{window}"] = out[f"ret_{window}"] - out[f"ret_mean_{window}"] * window

    if {3, 5}.issubset(set(windows)):
        out["short_momentum_3_5"] = out["ret_3"] + out["ret_5"]
        out["short_momentum_accel_3_5"] = out["ret_3"] - out["ret_5"]
        out["short_volume_momentum_3_5"] = out["short_momentum_3_5"] * out["volume_ratio_3"].clip(lower=0.0)
        if "amount_ratio_3" in out.columns and "amount_ratio_5" in out.columns:
            out["amount_accel_3_5"] = out["amount_ratio_3"] - out["amount_ratio_5"]
        if "turnover_ratio_3" in out.columns and "turnover_ratio_5" in out.columns:
            out["turnover_accel_3_5"] = out["turnover_ratio_3"] - out["turnover_ratio_5"]
    if {5, 20}.issubset(set(windows)):
        out["trend_alignment_5_20"] = out["ret_5"] + out["ret_20"]
        out["trend_accel_5_20"] = out["ret_5"] - out["ret_20"]
        out["breakout_volume_confirm_20"] = out["breakout_strength_20"] * out["volume_ratio_5"].clip(lower=0.0)
    if "pct_chg_decimal" in out.columns:
        out["limit_up_proximity"] = (out["pct_chg_decimal"] / 0.10).clip(lower=-1.0, upper=1.5)
    if {"ret_3", "volume_ratio_3"}.issubset(out.columns):
        liquidity = out["amount_ratio_5"] if "amount_ratio_5" in out.columns else out["volume_ratio_5"]
        out["fragile_rally_penalty"] = out["ret_3"].clip(lower=0.0) * (-liquidity).clip(lower=0.0)

    return out


def add_cross_sectional_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    feature_map: dict[str, pd.Series] = {}
    for col in feature_cols:
        grouped = out.groupby("date")[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        feature_map[f"{col}_cs_z"] = (out[col] - mean) / std
        feature_map[f"{col}_cs_rank"] = grouped.rank(pct=True)

    if feature_map:
        out = pd.concat([out, pd.DataFrame(feature_map, index=out.index)], axis=1)
    return out


def build_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "price": [],
        "volume": [],
        "volatility": [],
        "market": [],
        "cross_sectional": [],
        "style": [],
        "other": [],
    }

    volume_tokens = ("volume", "amount", "turnover")
    volatility_tokens = ("volatility", "amplitude", "shadow", "body", "ret_std", "ret_skew", "ret_to_vol", "rsi")
    market_tokens = ("index_", "market_", "beta_", "regime_", "excess_", "idio_", "alpha_")
    style_tokens = ("style_", "bucket", "group_", "theme_", "industry_collective")
    cross_tokens = ("_cs_z", "_cs_rank")

    for col in feature_cols:
        if any(token in col for token in cross_tokens):
            groups["cross_sectional"].append(col)
        elif any(token in col for token in market_tokens):
            groups["market"].append(col)
        elif any(token in col for token in style_tokens):
            groups["style"].append(col)
        elif any(token in col for token in volume_tokens):
            groups["volume"].append(col)
        elif any(token in col for token in volatility_tokens):
            groups["volatility"].append(col)
        elif any(
            token in col
            for token in (
                "open",
                "close",
                "high",
                "low",
                "ret_",
                "ma_ratio",
                "ema_",
                "macd",
                "price_position",
                "momentum",
                "breakout",
                "trend_",
            )
        ):
            groups["price"].append(col)
        else:
            groups["other"].append(col)

    return {name: cols for name, cols in groups.items() if cols}
