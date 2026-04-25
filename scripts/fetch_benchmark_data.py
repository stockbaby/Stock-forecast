from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
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


COL_STOCK_CODE = "\u80a1\u7968\u4ee3\u7801"
COL_STOCK_NAME = "\u80a1\u7968\u540d\u79f0"
COL_DATE = "\u65e5\u671f"
COL_OPEN = "\u5f00\u76d8"
COL_CLOSE = "\u6536\u76d8"
COL_HIGH = "\u6700\u9ad8"
COL_LOW = "\u6700\u4f4e"
COL_VOLUME = "\u6210\u4ea4\u91cf"
COL_AMOUNT = "\u6210\u4ea4\u989d"
COL_AMPLITUDE = "\u632f\u5e45"
COL_CHANGE_AMOUNT = "\u6da8\u8dcc\u989d"
COL_TURNOVER = "\u6362\u624b\u7387"
COL_PCT_CHG = "\u6da8\u8dcc\u5e45"


@dataclass
class FetchConfig:
    data_dir: Path
    output_name: str
    start_date: str
    end_date: str
    sleep_every: int
    sleep_seconds: float

    @property
    def output_path(self) -> Path:
        return self.data_dir / self.output_name

    @property
    def hs300_list_path(self) -> Path:
        return self.data_dir / "hs300_stock_list.csv"

    @property
    def failed_path(self) -> Path:
        return self.data_dir / "failed_stocks.csv"


def parse_args() -> FetchConfig:
    parser = argparse.ArgumentParser(
        description="Fetch CSI300 benchmark stock history from baostock with incremental update support."
    )
    parser.add_argument("--data-dir", default="data/raw", help="Directory for output CSV files.")
    parser.add_argument("--output-name", default="stock_data.csv", help="Main history CSV filename.")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default="2026-04-24", help="End date in YYYY-MM-DD format.")
    parser.add_argument("--sleep-every", type=int, default=10, help="Sleep after every N successful stocks.")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Sleep duration between request batches.")
    args = parser.parse_args()

    return FetchConfig(
        data_dir=Path(args.data_dir),
        output_name=args.output_name,
        start_date=args.start_date,
        end_date=args.end_date,
        sleep_every=args.sleep_every,
        sleep_seconds=args.sleep_seconds,
    )


def login() -> None:
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"baostock login failed: {result.error_msg}")
    print("baostock login succeeded")


def logout() -> None:
    bs.logout()
    print("baostock logout completed")


def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to decode {path}. Last error: {last_error}")


def get_hs300_stocks() -> pd.DataFrame:
    result = bs.query_hs300_stocks()
    if result.error_code != "0":
        raise RuntimeError(f"Failed to query CSI300 constituents: {result.error_msg}")

    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())

    df = pd.DataFrame(rows, columns=result.fields)
    if df.empty:
        raise RuntimeError("CSI300 constituent list is empty.")
    return df


def normalize_stock_code(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("sh.") or text.startswith("sz."):
        text = text[3:]
    return text.zfill(6)


def format_output_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.map(lambda x: f"{x.year}/{x.month}/{x.day}" if pd.notna(x) else None)


def get_stock_history(bs_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
    result = bs.query_history_k_data_plus(
        bs_code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="1",
    )
    if result.error_code != "0":
        raise RuntimeError(f"Failed to query {bs_code}: {result.error_msg}")

    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=result.fields)
    numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[COL_AMPLITUDE] = ((df["high"] - df["low"]) / df["preclose"] * 100).round(2)
    df[COL_CHANGE_AMOUNT] = (df["close"] - df["preclose"]).round(2)
    df["date"] = format_output_date(df["date"])
    df["code"] = df["code"].map(normalize_stock_code)

    renamed = df.rename(
        columns={
            "code": COL_STOCK_CODE,
            "date": COL_DATE,
            "open": COL_OPEN,
            "close": COL_CLOSE,
            "high": COL_HIGH,
            "low": COL_LOW,
            "volume": COL_VOLUME,
            "amount": COL_AMOUNT,
            "turn": COL_TURNOVER,
            "pctChg": COL_PCT_CHG,
        }
    )
    columns = [
        COL_STOCK_CODE,
        COL_DATE,
        COL_OPEN,
        COL_CLOSE,
        COL_HIGH,
        COL_LOW,
        COL_VOLUME,
        COL_AMOUNT,
        COL_AMPLITUDE,
        COL_CHANGE_AMOUNT,
        COL_TURNOVER,
        COL_PCT_CHG,
    ]
    return renamed[columns]


def get_existing_stocks(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        df = read_csv_auto(output_path)
        if COL_STOCK_CODE in df.columns:
            return {normalize_stock_code(v) for v in df[COL_STOCK_CODE].dropna().unique()}
    except Exception:
        return set()
    return set()


def filter_data_by_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty or COL_DATE not in df.columns:
        return df

    out = df.copy()
    out["_date_dt"] = pd.to_datetime(out[COL_DATE], format="%Y/%m/%d", errors="coerce")
    out = out.dropna(subset=["_date_dt"])
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    out = out[(out["_date_dt"] >= start_dt) & (out["_date_dt"] <= end_dt)].copy()
    return out.drop(columns=["_date_dt"])


def get_stock_date_range(output_path: Path, stock_code: str, start_date: str, end_date: str) -> tuple[str | None, str | None]:
    if not output_path.exists():
        return None, None

    try:
        df = read_csv_auto(output_path)
        if COL_STOCK_CODE not in df.columns or COL_DATE not in df.columns:
            return None, None

        stock_df = df[df[COL_STOCK_CODE].map(normalize_stock_code) == stock_code].copy()
        if stock_df.empty:
            return None, None

        stock_df["_date_dt"] = pd.to_datetime(stock_df[COL_DATE], format="%Y/%m/%d", errors="coerce")
        stock_df = stock_df.dropna(subset=["_date_dt"])
        if stock_df.empty:
            return None, None

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        stock_df = stock_df[(stock_df["_date_dt"] >= start_dt) & (stock_df["_date_dt"] <= end_dt)]
        if stock_df.empty:
            return None, None

        return (
            stock_df["_date_dt"].min().strftime("%Y-%m-%d"),
            stock_df["_date_dt"].max().strftime("%Y-%m-%d"),
        )
    except Exception as exc:
        print(f"warning: failed to inspect existing range for {stock_code}: {exc}")
        return None, None


def merge_stock_data(existing_df: pd.DataFrame, new_df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    if new_df is None or new_df.empty:
        return existing_df

    existing = existing_df.copy()
    existing["_stock_code"] = existing[COL_STOCK_CODE].map(normalize_stock_code)

    other_df = existing[existing["_stock_code"] != stock_code].drop(columns=["_stock_code"])
    stock_existing = existing[existing["_stock_code"] == stock_code].drop(columns=["_stock_code"])

    if not stock_existing.empty:
        old_part = stock_existing.copy()
        new_part = new_df.copy()
        old_part["_date_dt"] = pd.to_datetime(old_part[COL_DATE], format="%Y/%m/%d", errors="coerce")
        new_part["_date_dt"] = pd.to_datetime(new_part[COL_DATE], format="%Y/%m/%d", errors="coerce")

        combined = pd.concat([old_part, new_part], ignore_index=True)
        combined = combined.drop_duplicates(subset=["_date_dt"], keep="last")
        combined = combined.sort_values("_date_dt").drop(columns=["_date_dt"])
    else:
        combined = new_df

    return pd.concat([other_df, combined], ignore_index=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def verify_stock_blocks(df: pd.DataFrame) -> bool:
    if df.empty or COL_STOCK_CODE not in df.columns:
        return True
    grouped = df.groupby(COL_STOCK_CODE).apply(lambda x: x.index.max() - x.index.min() + 1)
    return int(grouped.sum()) == int(len(df))


def main() -> None:
    config = parse_args()
    config.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target range: {config.start_date} to {config.end_date}")
    print(f"Output file: {config.output_path}")
    print("=" * 60)

    existing_stocks = get_existing_stocks(config.output_path)
    if existing_stocks:
        print(f"Found existing data for {len(existing_stocks)} stocks; incremental update mode enabled.")

    login()
    try:
        hs300_df = get_hs300_stocks()
        save_dataframe(hs300_df, config.hs300_list_path)
        print(f"Saved constituent list to: {config.hs300_list_path}")

        existing_df: pd.DataFrame | None = None
        if config.output_path.exists() and existing_stocks:
            try:
                existing_df = read_csv_auto(config.output_path)
                raw_len = len(existing_df)
                existing_df = filter_data_by_date_range(existing_df, config.start_date, config.end_date)
                print(f"Loaded existing data: {len(existing_df)} rows")
                if len(existing_df) != raw_len:
                    print(f"Filtered old data to target window: {raw_len} -> {len(existing_df)}")
            except Exception as exc:
                print(f"warning: failed to load existing data: {exc}")

        hs300_df["pure_code"] = hs300_df["code"].map(normalize_stock_code)

        failed_stocks: list[tuple[str, str]] = []
        success_count = 0
        new_stock_count = 0
        incremental_count = 0
        total_new_records = 0
        total = len(hs300_df)

        for idx, row in hs300_df.iterrows():
            bs_code = row["code"]
            stock_name = row.get("code_name", "")
            pure_code = row["pure_code"]

            existing_min, existing_max = get_stock_date_range(
                config.output_path, pure_code, config.start_date, config.end_date
            )

            if existing_min and existing_max:
                need_early = existing_min > config.start_date
                need_late = existing_max < config.end_date

                if not need_early and not need_late:
                    print(
                        f"[{idx + 1}/{total}] {bs_code} {stock_name} - complete "
                        f"({existing_min} to {existing_max}), skip"
                    )
                    continue

                print(f"[{idx + 1}/{total}] {bs_code} {stock_name} - incremental update")
                print(f"  existing range: {existing_min} to {existing_max}")
                fetch_ranges: list[tuple[str, str, str]] = []
                if need_early:
                    early_end = (pd.to_datetime(existing_min) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    fetch_ranges.append((config.start_date, early_end, "early"))
                if need_late:
                    late_start = max(
                        pd.to_datetime(config.start_date),
                        pd.to_datetime(existing_max) + pd.Timedelta(days=1),
                    ).strftime("%Y-%m-%d")
                    fetch_ranges.append((late_start, config.end_date, "late"))
            else:
                print(f"[{idx + 1}/{total}] {bs_code} {stock_name} - full fetch")
                fetch_ranges = [(config.start_date, config.end_date, "full")]

            try:
                chunks: list[pd.DataFrame] = []
                for fetch_start, fetch_end, label in fetch_ranges:
                    print(f"  fetching {label}: {fetch_start} to {fetch_end}")
                    frame = get_stock_history(bs_code, fetch_start, fetch_end)
                    if frame is not None and not frame.empty:
                        chunks.append(frame)

                if not chunks:
                    print("  no new rows")
                    continue

                new_data = pd.concat(chunks, ignore_index=True)
                if existing_df is not None and not existing_df.empty:
                    existing_df = merge_stock_data(existing_df, new_data, pure_code)
                    save_dataframe(existing_df, config.output_path)
                    incremental_count += 1
                else:
                    save_dataframe(new_data, config.output_path)
                    existing_df = new_data
                    new_stock_count += 1

                total_new_records += len(new_data)
                success_count += 1
                print(f"  success: +{len(new_data)} rows")
            except Exception as exc:
                print(f"  failed: {exc}")
                failed_stocks.append((pure_code, stock_name))

            if config.sleep_every > 0 and success_count > 0 and success_count % config.sleep_every == 0:
                print(f"Sleeping {config.sleep_seconds} seconds after {success_count} successful stocks")
                time.sleep(config.sleep_seconds)

        print("=" * 60)
        print("Fetch finished")
        print(f"  new stocks: {new_stock_count}")
        print(f"  incremental stocks: {incremental_count}")
        print(f"  failed stocks: {len(failed_stocks)}")
        print(f"  new rows: {total_new_records}")

        if config.output_path.exists():
            df = read_csv_auto(config.output_path)
            print("File summary:")
            print(f"  size_mb: {config.output_path.stat().st_size / 1024 / 1024:.2f}")
            print(f"  total_rows: {len(df)}")
            if COL_STOCK_CODE in df.columns:
                print(f"  unique_stocks: {df[COL_STOCK_CODE].nunique()}")
            if COL_DATE in df.columns and len(df) > 0:
                print(f"  date_range: {df[COL_DATE].min()} to {df[COL_DATE].max()}")
            print(f"  contiguous_stock_blocks: {'yes' if verify_stock_blocks(df) else 'warning'}")

        if failed_stocks:
            failed_df = pd.DataFrame(failed_stocks, columns=[COL_STOCK_CODE, COL_STOCK_NAME])
            save_dataframe(failed_df, config.failed_path)
            print(f"Saved failed stock list to: {config.failed_path}")
    finally:
        logout()


if __name__ == "__main__":
    main()
