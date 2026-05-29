"""Evaluate the official THU baseline checkpoint on recent holdout windows.

The official baseline emits a Top5 equal-weight submission. This script keeps
the published checkpoint fixed and, for each T date, feeds prediction data
cut off at T only. That avoids using post-T data for training or inference.
"""

from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


DEFAULT_DATES = ["2026-04-17", "2026-04-24", "2026-04-30", "2026-05-08", "2026-05-15"]


def _next_trading_dates(raw: pd.DataFrame, date: str, horizon: int = 5) -> tuple[str, str]:
    dates = sorted(pd.to_datetime(raw["日期"]).dt.strftime("%Y-%m-%d").unique())
    if date not in dates:
        raise ValueError(f"{date} is not present in raw data")
    idx = dates.index(date)
    buy_idx = idx + 1
    sell_idx = idx + horizon
    if sell_idx >= len(dates):
        raise ValueError(f"Not enough future dates after {date} for T+{horizon}")
    return dates[buy_idx], dates[sell_idx]


def _return_for_stock(raw: pd.DataFrame, stock_id: str, buy_date: str, sell_date: str) -> float | None:
    sid = str(stock_id).zfill(6)
    stock = raw[raw["股票代码"].astype(str).str.zfill(6) == sid]
    buy = stock.loc[stock["日期"] == buy_date, "开盘"]
    sell = stock.loc[stock["日期"] == sell_date, "开盘"]
    if buy.empty or sell.empty:
        return None
    buy_price = float(buy.iloc[0])
    sell_price = float(sell.iloc[0])
    if abs(buy_price) < 1e-12:
        return None
    return (sell_price - buy_price) / buy_price


def _summarize(series: pd.Series) -> dict[str, float | int]:
    returns = series.dropna().astype(float)
    if returns.empty:
        return {
            "n": 0,
            "hit_rate": float("nan"),
            "mean_return": float("nan"),
            "p05_return": float("nan"),
            "negative_rate": float("nan"),
            "max_drawdown": float("nan"),
        }
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "n": int(len(returns)),
        "hit_rate": float((returns > 0).mean()),
        "mean_return": float(returns.mean()),
        "p05_return": float(returns.quantile(0.05)),
        "negative_rate": float((returns < 0).mean()),
        "max_drawdown": float(drawdown.min()),
    }


def run_official_predict(
    official_root: Path,
    data_path: Path,
    model_dir: Path,
    work_dir: Path,
) -> Path:
    src_dir = official_root / "code" / "src"
    sys.path.insert(0, str(src_dir))

    config_mod = importlib.import_module("config")
    config_mod.config["data_path"] = str(data_path)
    config_mod.config["output_dir"] = str(model_dir)

    predict_mod = importlib.import_module("predict")
    predict_mod.config["data_path"] = str(data_path)
    predict_mod.config["output_dir"] = str(model_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "output").mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        predict_mod.main()
    finally:
        os.chdir(old_cwd)
        sys.path = [p for p in sys.path if p != str(src_dir)]

    return work_dir / "output" / "result.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default="external/THU-BDC2026")
    parser.add_argument("--raw-root", default="outputs/recent_holdout_matrix_strict_full/raw_cutoffs")
    parser.add_argument("--full-raw", default="data/raw/stock_data.csv")
    parser.add_argument("--out-dir", default="outputs/official_baseline_holdout")
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    args = parser.parse_args()

    official_root = Path(args.official_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    model_dir = official_root / "model" / "60_158+39"
    full_raw = pd.read_csv(args.full_raw, dtype={"股票代码": str})
    full_raw["股票代码"] = full_raw["股票代码"].astype(str).str.zfill(6)
    full_raw["日期"] = pd.to_datetime(full_raw["日期"]).dt.strftime("%Y-%m-%d")

    rows = []
    for date in args.dates:
        tag = date.replace("-", "")
        work_dir = out_dir / tag
        raw_file = raw_root / tag / "stock_data.csv"
        if not raw_file.exists():
            raise FileNotFoundError(raw_file)
        data_path = work_dir / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        train_file = data_path / "train.csv"
        if not train_file.exists():
            try:
                os.link(raw_file, train_file)
            except OSError:
                shutil.copy2(raw_file, train_file)

        result_path = run_official_predict(official_root, data_path, model_dir, work_dir)
        pred = pd.read_csv(result_path, dtype={"stock_id": str})
        pred["stock_id"] = pred["stock_id"].astype(str).str.zfill(6)

        buy_date, sell_date = _next_trading_dates(full_raw, date)
        returns = [
            _return_for_stock(full_raw, sid, buy_date, sell_date)
            for sid in pred["stock_id"].tolist()
        ]
        weighted_return = sum(
            float(w) * float(r)
            for w, r in zip(pred["weight"].tolist(), returns)
            if r is not None
        )
        top1_return = returns[0] if returns else None
        rows.append(
            {
                "date": date,
                "buy_date": buy_date,
                "sell_date": sell_date,
                "official_top1": pred["stock_id"].iloc[0],
                "official_top1_return": top1_return,
                "official_top5": ";".join(pred["stock_id"].tolist()),
                "official_top5_weights": ";".join(str(x) for x in pred["weight"].tolist()),
                "official_top5_returns": ";".join("" if r is None else f"{r:.10f}" for r in returns),
                "official_top5_equal_return": weighted_return,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.DataFrame(rows)
    matrix.to_csv(out_dir / "official_baseline_holdout_matrix.csv", index=False)

    summary = pd.DataFrame(
        [
            {"source": "official_baseline_top1", **_summarize(matrix["official_top1_return"])},
            {"source": "official_baseline_top5_equal", **_summarize(matrix["official_top5_equal_return"])},
        ]
    )
    summary.to_csv(out_dir / "official_baseline_summary.csv", index=False)
    print(matrix.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
