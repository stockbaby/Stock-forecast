from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


COL_STOCK_CODE = "\u80a1\u7968\u4ee3\u7801"
COL_DATE = "\u65e5\u671f"
COL_OPEN = "\u5f00\u76d8"


@dataclass
class EvalConfig:
    data_path: Path
    submission_path: Path
    baseline_submission_path: Path | None
    trade_date: str
    buy_offset: int
    sell_offset: int
    sell_fallback_offset: int | None


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate a submission return using the competition formula and optionally compare with a baseline."
    )
    parser.add_argument("--data-path", default="data/raw/stock_data.csv")
    parser.add_argument("--submission-path", required=True)
    parser.add_argument("--baseline-submission-path", default=None)
    parser.add_argument("--trade-date", required=True, help="Reference trade date T in YYYY-MM-DD format.")
    parser.add_argument("--buy-offset", type=int, default=1)
    parser.add_argument("--sell-offset", type=int, default=5)
    parser.add_argument("--sell-fallback-offset", type=int, default=None)
    args = parser.parse_args()
    return EvalConfig(
        data_path=Path(args.data_path),
        submission_path=Path(args.submission_path),
        baseline_submission_path=Path(args.baseline_submission_path) if args.baseline_submission_path else None,
        trade_date=args.trade_date,
        buy_offset=args.buy_offset,
        sell_offset=args.sell_offset,
        sell_fallback_offset=args.sell_fallback_offset,
    )


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to decode {path}. Last error: {last_error}")


def normalize_stock_code(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("sh.") or text.startswith("sz."):
        text = text[3:]
    if "." in text:
        text = text.split(".")[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6)


def load_price_frame(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)
    out = df.copy()
    out[COL_STOCK_CODE] = out[COL_STOCK_CODE].map(normalize_stock_code)
    out[COL_DATE] = pd.to_datetime(out[COL_DATE], format="%Y/%m/%d", errors="coerce")
    out[COL_OPEN] = pd.to_numeric(out[COL_OPEN], errors="coerce")
    out = out.dropna(subset=[COL_STOCK_CODE, COL_DATE, COL_OPEN])
    return out[[COL_STOCK_CODE, COL_DATE, COL_OPEN]].copy()


def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    df["stock_id"] = df["stock_id"].map(normalize_stock_code)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["stock_id", "weight"])
    return df[["stock_id", "weight"]].copy()


def get_trading_dates(price_df: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(price_df[COL_DATE].drop_duplicates().tolist())


def resolve_trade_dates(
    price_df: pd.DataFrame,
    trade_date: str,
    buy_offset: int,
    sell_offset: int,
    sell_fallback_offset: int | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    trading_dates = get_trading_dates(price_df)
    t_date = pd.to_datetime(trade_date)
    if t_date not in trading_dates:
        raise ValueError(f"Trade date {trade_date} not found in data.")

    base_idx = trading_dates.index(t_date)
    buy_idx = base_idx + buy_offset
    sell_idx = base_idx + sell_offset
    if buy_idx >= len(trading_dates):
        raise ValueError("Buy date is beyond available trading dates.")

    if sell_idx >= len(trading_dates):
        if sell_fallback_offset is None:
            raise ValueError("Sell date is beyond available trading dates and no fallback is provided.")
        sell_idx = base_idx + sell_fallback_offset
        if sell_idx >= len(trading_dates):
            raise ValueError("Fallback sell date is also beyond available trading dates.")

    return trading_dates[buy_idx], trading_dates[sell_idx]


def evaluate_submission(
    price_df: pd.DataFrame,
    submission_df: pd.DataFrame,
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
) -> dict:
    buy_open = price_df[price_df[COL_DATE] == buy_date][[COL_STOCK_CODE, COL_OPEN]].rename(columns={COL_OPEN: "buy_open"})
    sell_open = price_df[price_df[COL_DATE] == sell_date][[COL_STOCK_CODE, COL_OPEN]].rename(columns={COL_OPEN: "sell_open"})

    merged = submission_df.merge(buy_open, left_on="stock_id", right_on=COL_STOCK_CODE, how="left")
    merged = merged.merge(sell_open, left_on="stock_id", right_on=COL_STOCK_CODE, how="left", suffixes=("", "_sell"))
    merged = merged.drop(columns=[col for col in merged.columns if col.startswith(COL_STOCK_CODE)])

    if merged["buy_open"].isna().any() or merged["sell_open"].isna().any():
        missing = merged[merged["buy_open"].isna() | merged["sell_open"].isna()]["stock_id"].tolist()
        raise ValueError(f"Missing buy/sell price for stocks: {missing}")

    merged["single_return"] = (merged["sell_open"] - merged["buy_open"]) / merged["buy_open"]
    merged["weighted_return"] = merged["weight"] * merged["single_return"]
    total_return = float(merged["weighted_return"].sum())
    cash_weight = float(1.0 - merged["weight"].sum())

    return {
        "buy_date": buy_date.strftime("%Y-%m-%d"),
        "sell_date": sell_date.strftime("%Y-%m-%d"),
        "cash_weight": cash_weight,
        "portfolio_return": total_return,
        "details": merged[["stock_id", "weight", "buy_open", "sell_open", "single_return", "weighted_return"]]
        .sort_values("weighted_return", ascending=False)
        .to_dict(orient="records"),
    }


def main() -> None:
    config = parse_args()
    price_df = load_price_frame(config.data_path)
    buy_date, sell_date = resolve_trade_dates(
        price_df,
        config.trade_date,
        config.buy_offset,
        config.sell_offset,
        config.sell_fallback_offset,
    )

    submission_df = load_submission(config.submission_path)
    result = {"candidate": evaluate_submission(price_df, submission_df, buy_date, sell_date)}

    if config.baseline_submission_path is not None:
        baseline_df = load_submission(config.baseline_submission_path)
        baseline_result = evaluate_submission(price_df, baseline_df, buy_date, sell_date)
        result["baseline"] = baseline_result
        result["return_diff_vs_baseline"] = (
            result["candidate"]["portfolio_return"] - baseline_result["portfolio_return"]
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
