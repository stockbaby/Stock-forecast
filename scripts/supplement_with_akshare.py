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
    import akshare as ak
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "akshare is required for this script. Install dependencies from environment.yml or requirements.txt."
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
class SupplementConfig:
    input_path: Path
    output_path: Path
    hs300_list_path: Path | None
    start_date: str
    end_date: str
    adjust: str
    sleep_every: int
    sleep_seconds: float
    volume_multiplier: int


def parse_args() -> SupplementConfig:
    parser = argparse.ArgumentParser(
        description="Supplement a benchmark stock_data.csv with missing tail data from AkShare."
    )
    parser.add_argument("--input-path", default="data/raw/stock_data.csv")
    parser.add_argument("--output-path", default="data/raw/stock_data.csv")
    parser.add_argument("--hs300-list-path", default="data/raw/hs300_stock_list.csv")
    parser.add_argument("--start-date", default="2026-02-10", help="Supplement start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default="2026-04-24", help="Supplement end date in YYYY-MM-DD format.")
    parser.add_argument("--adjust", default="hfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--sleep-every", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument(
        "--volume-multiplier",
        type=int,
        default=100,
        help="AkShare stock_zh_a_hist volume is treated as lots; multiply by 100 to align with baostock share volume.",
    )
    args = parser.parse_args()

    hs300_list_path = Path(args.hs300_list_path) if args.hs300_list_path else None
    return SupplementConfig(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        hs300_list_path=hs300_list_path,
        start_date=args.start_date,
        end_date=args.end_date,
        adjust=args.adjust,
        sleep_every=args.sleep_every,
        sleep_seconds=args.sleep_seconds,
        volume_multiplier=args.volume_multiplier,
    )


def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to decode {path}. Last error: {last_error}")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_stock_code(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("sh.") or text.startswith("sz."):
        text = text[3:]
    return text.zfill(6)


def read_target_stock_codes(existing_df: pd.DataFrame, hs300_list_path: Path | None) -> list[str]:
    if hs300_list_path and hs300_list_path.exists():
        hs300_df = read_csv_auto(hs300_list_path)
        if "code" in hs300_df.columns:
            return sorted({normalize_stock_code(code) for code in hs300_df["code"].dropna()})

    if COL_STOCK_CODE not in existing_df.columns:
        raise ValueError(f"Input file must contain column {COL_STOCK_CODE}")
    return sorted({normalize_stock_code(code) for code in existing_df[COL_STOCK_CODE].dropna()})


def get_existing_tail_date(df: pd.DataFrame, stock_code: str) -> pd.Timestamp | None:
    stock_df = df[df[COL_STOCK_CODE].map(normalize_stock_code) == stock_code].copy()
    if stock_df.empty:
        return None
    stock_df["_date_dt"] = pd.to_datetime(stock_df[COL_DATE], format="%Y/%m/%d", errors="coerce")
    stock_df = stock_df.dropna(subset=["_date_dt"])
    if stock_df.empty:
        return None
    return pd.Timestamp(stock_df["_date_dt"].max())


def format_output_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.map(lambda x: f"{x.year}/{x.month}/{x.day}" if pd.notna(x) else None)


def fetch_akshare_history(
    stock_code: str,
    start_date: str,
    end_date: str,
    adjust: str,
    volume_multiplier: int,
) -> pd.DataFrame | None:
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust=adjust,
    )
    if df is None or df.empty:
        return None

    out = df.copy()
    out[COL_STOCK_CODE] = stock_code
    out[COL_DATE] = format_output_date(out[COL_DATE] if COL_DATE in out.columns else out["日期"])

    numeric_cols = [
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
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out[COL_VOLUME] = out[COL_VOLUME] * volume_multiplier

    ordered_columns = [
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
    return out[ordered_columns]


def merge_stock_data(existing_df: pd.DataFrame, new_df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
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


def verify_stock_blocks(df: pd.DataFrame) -> bool:
    if df.empty or COL_STOCK_CODE not in df.columns:
        return True
    grouped = df.groupby(COL_STOCK_CODE).apply(lambda x: x.index.max() - x.index.min() + 1)
    return int(grouped.sum()) == int(len(df))


def main() -> None:
    config = parse_args()
    if not config.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {config.input_path}")

    existing_df = read_csv_auto(config.input_path)
    stock_codes = read_target_stock_codes(existing_df, config.hs300_list_path)

    print(f"Input file: {config.input_path}")
    print(f"Output file: {config.output_path}")
    print(f"Supplement window: {config.start_date} to {config.end_date}")
    print(f"Target stocks: {len(stock_codes)}")
    print("=" * 60)

    success_count = 0
    supplemented_count = 0
    skipped_count = 0
    failed: list[tuple[str, str]] = []
    total_new_rows = 0

    for idx, stock_code in enumerate(stock_codes, start=1):
        tail_dt = get_existing_tail_date(existing_df, stock_code)
        effective_start = pd.to_datetime(config.start_date)
        if tail_dt is not None:
            effective_start = max(effective_start, tail_dt + pd.Timedelta(days=1))

        effective_end = pd.to_datetime(config.end_date)
        if effective_start > effective_end:
            print(f"[{idx}/{len(stock_codes)}] {stock_code} - already covered, skip")
            skipped_count += 1
            continue

        start_str = effective_start.strftime("%Y-%m-%d")
        end_str = effective_end.strftime("%Y-%m-%d")
        print(f"[{idx}/{len(stock_codes)}] {stock_code} - supplement {start_str} to {end_str}")

        try:
            frame = fetch_akshare_history(
                stock_code=stock_code,
                start_date=start_str,
                end_date=end_str,
                adjust=config.adjust,
                volume_multiplier=config.volume_multiplier,
            )
            if frame is None or frame.empty:
                print("  no new rows")
                skipped_count += 1
                continue

            existing_df = merge_stock_data(existing_df, frame, stock_code)
            save_dataframe(existing_df, config.output_path)
            success_count += 1
            supplemented_count += 1
            total_new_rows += len(frame)
            print(f"  success: +{len(frame)} rows")
        except Exception as exc:
            print(f"  failed: {exc}")
            failed.append((stock_code, str(exc)))

        if config.sleep_every > 0 and success_count > 0 and success_count % config.sleep_every == 0:
            print(f"Sleeping {config.sleep_seconds} seconds after {success_count} successful supplements")
            time.sleep(config.sleep_seconds)

    print("=" * 60)
    print("AkShare supplement finished")
    print(f"  supplemented stocks: {supplemented_count}")
    print(f"  skipped stocks: {skipped_count}")
    print(f"  failed stocks: {len(failed)}")
    print(f"  new rows: {total_new_rows}")

    final_df = read_csv_auto(config.output_path)
    print("File summary:")
    print(f"  total_rows: {len(final_df)}")
    print(f"  unique_stocks: {final_df[COL_STOCK_CODE].nunique()}")
    print(f"  date_range: {final_df[COL_DATE].min()} to {final_df[COL_DATE].max()}")
    print(f"  contiguous_stock_blocks: {'yes' if verify_stock_blocks(final_df) else 'warning'}")

    if failed:
        failed_df = pd.DataFrame(failed, columns=[COL_STOCK_CODE, "error"])
        failed_path = config.output_path.parent / "failed_stocks_akshare.csv"
        save_dataframe(failed_df, failed_path)
        print(f"Saved AkShare failure list to: {failed_path}")


if __name__ == "__main__":
    main()
