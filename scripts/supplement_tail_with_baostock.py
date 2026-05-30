from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import baostock as bs
except ImportError as exc:  # pragma: no cover
    raise SystemExit("baostock is required in the active Python environment.") from exc


COL_STOCK_CODE = "\u80a1\u7968\u4ee3\u7801"
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


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype={COL_STOCK_CODE: str})
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype={COL_STOCK_CODE: str})


def _norm_stock(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith(("sh.", "sz.")):
        text = text[3:]
    return text.zfill(6)


def _bs_code(stock_id: str) -> str:
    return f"sh.{stock_id}" if stock_id.startswith(("6", "9")) else f"sz.{stock_id}"


def _format_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.map(lambda x: f"{x.year}/{x.month}/{x.day}" if pd.notna(x) else None)


def _fetch(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
    rs = bs.query_history_k_data_plus(
        _bs_code(stock_id),
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="1",
    )
    if rs.error_code != "0":
        raise RuntimeError(rs.error_msg)
    rows: list[list[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rs.fields)
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out = pd.DataFrame(
        {
            COL_STOCK_CODE: stock_id,
            COL_DATE: _format_date(df["date"]),
            COL_OPEN: df["open"],
            COL_CLOSE: df["close"],
            COL_HIGH: df["high"],
            COL_LOW: df["low"],
            COL_VOLUME: df["volume"],
            COL_AMOUNT: df["amount"],
            COL_AMPLITUDE: ((df["high"] - df["low"]) / df["preclose"] * 100).round(2),
            COL_CHANGE_AMOUNT: (df["close"] - df["preclose"]).round(2),
            COL_TURNOVER: df["turn"],
            COL_PCT_CHG: df["pctChg"],
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a recent baostock tail to an existing benchmark stock_data.csv.")
    parser.add_argument("--input-path", default="data/raw/stock_data.csv")
    parser.add_argument("--output-path", default="data/raw_latestA_20260529/stock_data.csv")
    parser.add_argument("--hs300-list-path", default="data/raw/hs300_stock_list.csv")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--sleep-every", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()

    source = _read_csv(Path(args.input_path))
    source[COL_STOCK_CODE] = source[COL_STOCK_CODE].map(_norm_stock)
    source["_date_dt"] = pd.to_datetime(source[COL_DATE], errors="coerce")

    stock_list = _read_csv(Path(args.hs300_list_path))
    code_col = "code" if "code" in stock_list.columns else stock_list.columns[1]
    stocks = sorted({_norm_stock(code) for code in stock_list[code_col].dropna()})

    print(f"Input rows: {len(source)}")
    print(f"Target stocks: {len(stocks)}")
    print(f"Tail window: {args.start_date} to {args.end_date}")

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_msg}")

    frames = [source.drop(columns=["_date_dt"])]
    failed: list[tuple[str, str]] = []
    try:
        for idx, stock in enumerate(stocks, start=1):
            stock_dates = source.loc[source[COL_STOCK_CODE] == stock, "_date_dt"]
            tail = stock_dates.max() if not stock_dates.empty else pd.NaT
            fetch_start = pd.Timestamp(args.start_date)
            if pd.notna(tail):
                fetch_start = max(fetch_start, pd.Timestamp(tail) + pd.Timedelta(days=1))
            if fetch_start > pd.Timestamp(args.end_date):
                print(f"[{idx}/{len(stocks)}] {stock} already complete")
                continue
            try:
                print(f"[{idx}/{len(stocks)}] {stock} fetch {fetch_start.date()} to {args.end_date}")
                part = _fetch(stock, fetch_start.strftime("%Y-%m-%d"), args.end_date)
                if not part.empty:
                    frames.append(part)
                    print(f"  +{len(part)} rows")
                else:
                    print("  no rows")
            except Exception as exc:
                failed.append((stock, str(exc)))
                print(f"  failed: {exc}")
            if args.sleep_every > 0 and idx % args.sleep_every == 0:
                time.sleep(args.sleep_seconds)
    finally:
        bs.logout()

    out = pd.concat(frames, ignore_index=True)
    out[COL_STOCK_CODE] = out[COL_STOCK_CODE].map(_norm_stock)
    out["_date_dt"] = pd.to_datetime(out[COL_DATE], errors="coerce")
    out = out.dropna(subset=["_date_dt"])
    out = out.drop_duplicates(subset=[COL_STOCK_CODE, "_date_dt"], keep="last")
    out = out.sort_values([COL_STOCK_CODE, "_date_dt"]).drop(columns=["_date_dt"])

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    if failed:
        pd.DataFrame(failed, columns=["stock_id", "error"]).to_csv(output_path.parent / "failed_baostock_tail.csv", index=False)

    check = out.copy()
    check["_date_dt"] = pd.to_datetime(check[COL_DATE], errors="coerce")
    coverage = check[check["_date_dt"] >= pd.Timestamp(args.start_date)].groupby("_date_dt")[COL_STOCK_CODE].nunique()
    print("Coverage:")
    print(coverage.to_string())
    print(f"Failed stocks: {len(failed)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
