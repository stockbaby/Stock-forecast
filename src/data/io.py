from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REQUIRED_PRICE_COLUMNS = ["date", "stock_id", "open", "high", "low", "close", "volume"]
OPTIONAL_NUMERIC_COLUMNS = [
    "amount",
    "amplitude_pct",
    "change_amount",
    "turnover_rate_pct",
    "pct_chg",
]


def discover_csv_files(raw_dir: str | Path) -> list[Path]:
    raw_path = Path(raw_dir)
    candidates = sorted(raw_path.glob("*.csv"))
    excluded_names = {
        "hs300_stock_list.csv",
        "failed_stocks.csv",
        "failed_stocks_akshare.csv",
    }
    return [path for path in candidates if path.name not in excluded_names]


def load_price_data(raw_dir: str | Path) -> pd.DataFrame:
    files = discover_csv_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No CSV files were found under {Path(raw_dir).resolve()}. "
            "Put raw daily stock files into data/raw first."
        )

    frames = [_read_single_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    return normalize_price_frame(df)


def _read_single_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"Failed to decode {path} with supported encodings. Last error: {last_error}")


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "datetime": "date",
        "trade_date": "date",
        "\u65e5\u671f": "date",
        "code": "stock_id",
        "ts_code": "stock_id",
        "\u80a1\u7968\u4ee3\u7801": "stock_id",
        "vol": "volume",
        "\u5f00\u76d8": "open",
        "\u6536\u76d8": "close",
        "\u6700\u9ad8": "high",
        "\u6700\u4f4e": "low",
        "\u6210\u4ea4\u91cf": "volume",
        "\u6210\u4ea4\u989d": "amount",
        "\u632f\u5e45": "amplitude_pct",
        "\u6da8\u8dcc\u989d": "change_amount",
        "\u6362\u624b\u7387": "turnover_rate_pct",
        "\u6da8\u8dcc\u5e45": "pct_chg",
    }
    out = df.rename(columns=rename_map).copy()
    out = _coalesce_duplicate_columns(out)

    missing = [col for col in REQUIRED_PRICE_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["stock_id"] = out["stock_id"].map(_normalize_stock_id)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    optional_cols = [col for col in OPTIONAL_NUMERIC_COLUMNS if col in out.columns]
    for col in numeric_cols + optional_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "stock_id", "open", "high", "low", "close"])
    out = out.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return out


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df

    deduped = pd.DataFrame(index=df.index)
    seen: set[str] = set()
    for column in df.columns:
        if column in seen:
            continue
        seen.add(column)
        same_name = df.loc[:, df.columns == column]
        if same_name.shape[1] == 1:
            deduped[column] = same_name.iloc[:, 0]
        else:
            deduped[column] = same_name.bfill(axis=1).iloc[:, 0]
    return deduped


def _normalize_stock_id(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    match = re.search(r"(\d{6})", text)
    if match:
        return match.group(1)

    if text.isdigit():
        return text.zfill(6)

    return None


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
