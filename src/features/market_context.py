from __future__ import annotations

from pathlib import Path

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
    out = stock_df.merge(index_df, on="date", how="left")
    out["index_ret_1"] = out["index_close"].pct_change(1)
    out["stock_excess_ret_1"] = out["ret_1"] - out["index_ret_1"]
    out["index_open_to_close"] = (out["index_close"] / out["index_open"]) - 1.0
    out["index_high_to_low"] = (out["index_high"] / out["index_low"]) - 1.0

    for window in windows:
        out[f"index_ret_{window}"] = out["index_close"].pct_change(window)
        out[f"index_ma_ratio_{window}"] = (
            out["index_close"] / out["index_close"].rolling(window).mean()
        ) - 1.0
        out[f"stock_excess_ret_{window}"] = out[f"ret_{window}"] - out[f"index_ret_{window}"]

    return out
