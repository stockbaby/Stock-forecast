from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import baostock as bs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "baostock is required for this script. Install dependencies from environment.yml or requirements.txt."
    ) from exc


def format_output_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.map(lambda x: f"{x.year}/{x.month}/{x.day}" if pd.notna(x) else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch HS300 index history from baostock.")
    parser.add_argument("--output-path", default="data/raw/hs300_index.csv")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2026-04-24")
    args = parser.parse_args()

    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"baostock login failed: {result.error_msg}")

    try:
        fields = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
        rs = bs.query_history_k_data_plus(
            "sh.000300",
            fields,
            start_date=args.start_date,
            end_date=args.end_date,
            frequency="d",
            adjustflag="3",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"failed to fetch hs300 index: {rs.error_msg}")

        rows: list[list[str]] = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = format_output_date(df["date"])
        out = df.rename(
            columns={
                "date": "\u65e5\u671f",
                "open": "\u5f00\u76d8",
                "close": "\u6536\u76d8",
                "high": "\u6700\u9ad8",
                "low": "\u6700\u4f4e",
                "volume": "\u6210\u4ea4\u91cf",
                "amount": "\u6210\u4ea4\u989d",
                "pctChg": "\u6da8\u8dcc\u5e45",
            }
        )[
            ["\u65e5\u671f", "\u5f00\u76d8", "\u6536\u76d8", "\u6700\u9ad8", "\u6700\u4f4e", "\u6210\u4ea4\u91cf", "\u6210\u4ea4\u989d", "\u6da8\u8dcc\u5e45"]
        ]
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved HS300 index file to: {output_path}")
    finally:
        bs.logout()
