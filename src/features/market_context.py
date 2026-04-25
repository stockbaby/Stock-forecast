from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_market_index_frame(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Market index file not found: {file_path}")

    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Failed to decode market index file: {last_error}")

    rename_map = {
        "datetime": "date",
        "trade_date": "date",
        "\u65e5\u671f": "date",
        "\u5f00\u76d8": "index_open",
        "\u6536\u76d8": "index_close",
        "\u6700\u9ad8": "index_high",
        "\u6700\u4f4e": "index_low",
        "open": "index_open",
        "close": "index_close",
        "high": "index_high",
        "low": "index_low",
        "volume": "index_volume",
        "\u6210\u4ea4\u91cf": "index_volume",
        "amount": "index_amount",
        "\u6210\u4ea4\u989d": "index_amount",
    }
    out = df.rename(columns=rename_map).copy()
    required = ["date", "index_open", "index_close", "index_high", "index_low"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required market index columns: {missing}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in [c for c in out.columns if c.startswith("index_")]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "index_open", "index_close"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


def add_market_index_features(stock_df: pd.DataFrame, index_df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    idx = index_df.sort_values("date").copy()
    idx["index_ret_1"] = idx["index_close"].pct_change(1)
    idx["index_open_to_close"] = (idx["index_close"] / idx["index_open"]) - 1.0
    idx["index_high_to_low"] = (idx["index_high"] / idx["index_low"]) - 1.0

    for window in windows:
        idx[f"index_ret_{window}"] = idx["index_close"].pct_change(window)
        idx[f"index_ma_ratio_{window}"] = (
            idx["index_close"] / idx["index_close"].rolling(window).mean()
        ) - 1.0
        idx[f"index_volatility_{window}"] = idx["index_ret_1"].rolling(window).std()
        idx[f"index_drawdown_{window}"] = (
            idx["index_close"] / idx["index_close"].rolling(window).max()
        ) - 1.0

    short_window = min(windows) if windows else 5
    long_window = max(windows) if windows else 20
    idx["regime_trend"] = idx[f"index_ma_ratio_{short_window}"] - idx[f"index_ma_ratio_{long_window}"]
    idx["regime_vol_ratio"] = idx[f"index_volatility_{short_window}"] / idx[f"index_volatility_{long_window}"].replace(
        0, np.nan
    )
    idx["regime_drawdown"] = idx[f"index_drawdown_{long_window}"]
    idx["regime_score"] = idx["regime_trend"] / idx["regime_vol_ratio"].replace(0, np.nan)
    idx["regime_is_trending"] = (idx["regime_trend"] > 0).astype(float)
    idx["regime_is_high_vol"] = (idx["regime_vol_ratio"] > 1.0).astype(float)

    out = stock_df.merge(idx, on="date", how="left")
    out["stock_excess_ret_1"] = out["ret_1"] - out["index_ret_1"]

    for window in windows:
        out[f"stock_excess_ret_{window}"] = out[f"ret_{window}"] - out[f"index_ret_{window}"]

    g = out.groupby("stock_id", group_keys=False)
    for window in sorted(set(windows + [20, 60])):
        cov = g.apply(
            lambda group: group["ret_1"].rolling(window).cov(group["index_ret_1"])
        ).reset_index(level=0, drop=True)
        var = g["index_ret_1"].rolling(window).var().reset_index(level=0, drop=True)
        out[f"beta_{window}"] = cov / var.replace(0, np.nan)
        out[f"idio_ret_{window}"] = out["ret_1"] - out[f"beta_{window}"] * out["index_ret_1"]

    liquidity_source = out["amount"] if "amount" in out.columns else out["volume"]
    vol_source = out["volatility_20"] if "volatility_20" in out.columns else out["ret_1"].abs()
    beta_source = out["beta_20"]

    def _bucket(series: pd.Series, q: int = 3) -> pd.Series:
        ranks = series.rank(pct=True, method="average")
        buckets = np.floor((ranks.fillna(0.5) * q).clip(upper=q - 1e-6)).astype(int)
        return buckets

    out["style_liquidity_bucket"] = out.groupby("date")[liquidity_source.name].transform(_bucket)
    out["style_vol_bucket"] = out.groupby("date")[vol_source.name].transform(_bucket)
    out["style_beta_bucket"] = out.groupby("date")[beta_source.name].transform(_bucket)
    out["style_bucket"] = (
        out["style_liquidity_bucket"].astype(str)
        + "_"
        + out["style_vol_bucket"].astype(str)
        + "_"
        + out["style_beta_bucket"].astype(str)
    )

    style_feature_map: dict[str, pd.Series] = {}
    style_feature_cols = [
        "ret_1",
        "ret_5",
        "ret_10",
        "ret_20",
        "ma_ratio_5",
        "ma_ratio_20",
        "stock_excess_ret_1",
        "stock_excess_ret_5",
        "beta_20",
        "idio_ret_20",
    ]
    for col in [feature for feature in style_feature_cols if feature in out.columns]:
        grouped = out.groupby(["date", "style_bucket"])[col]
        style_mean = grouped.transform("mean")
        style_std = grouped.transform("std").replace(0, np.nan)
        style_feature_map[f"style_excess_{col}"] = out[col] - style_mean
        style_feature_map[f"style_z_{col}"] = (out[col] - style_mean) / style_std
        style_feature_map[f"style_rank_{col}"] = grouped.rank(pct=True)

    if style_feature_map:
        out = pd.concat([out, pd.DataFrame(style_feature_map, index=out.index)], axis=1)

    return out
