from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.construct import build_top_k_submission
from src.training.train_baseline import validate_submission


DEFAULT_MODELS = {
    "baseline": "baseline/baseline_seed_42_latest_predictions.csv",
    "master": "master/master_seed_42_latest_predictions.csv",
    "master_ms": "master/master_seed_*_latest_predictions.csv",
    "stockmixer": "stockmixer_official/stockmixer_official_seed_42_latest_predictions.csv",
    "stockmixer_ms": "stockmixer_official/stockmixer_official_seed_*_latest_predictions.csv",
    "timexer": "timexer/timexer_seed_42_latest_predictions.csv",
    "timexer_ms": "timexer/timexer_seed_*_latest_predictions.csv",
}


def _norm_stock(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"股票代码": str, "stock_id": str})
    rename = {"股票代码": "stock_id", "日期": "date", "开盘": "open", "收盘": "close"}
    df = df.rename(columns=rename)
    df["stock_id"] = df["stock_id"].map(_norm_stock)
    df["date"] = pd.to_datetime(df["date"])
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    return df.dropna(subset=["stock_id", "date", "open"])


def _date_returns(price: pd.DataFrame, buy_date: str, sell_date: str) -> pd.DataFrame:
    buy = price[price["date"] == pd.to_datetime(buy_date)][["stock_id", "open"]].rename(columns={"open": "buy_open"})
    sell = price[price["date"] == pd.to_datetime(sell_date)][["stock_id", "open"]].rename(columns={"open": "sell_open"})
    ret = buy.merge(sell, on="stock_id", how="inner")
    ret["label"] = ret["sell_open"] / ret["buy_open"] - 1.0
    return ret[["stock_id", "label"]]


def _read_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    if "score" not in df.columns and "score_mean" in df.columns:
        df = df.rename(columns={"score_mean": "score"})
    out = df[["stock_id", "date", "score"]].copy()
    out["stock_id"] = out["stock_id"].map(_norm_stock)
    out["date"] = pd.to_datetime(out["date"])
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out.dropna(subset=["score"])


def _load_model_prediction(root: Path, date_key: str, pattern: str) -> pd.DataFrame | None:
    base = root / date_key
    paths = sorted(base.glob(pattern))
    if not paths:
        return None
    frames = [_read_prediction(path).rename(columns={"score": f"score_{idx}"}) for idx, path in enumerate(paths)]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["stock_id", "date"], how="inner")
    score_cols = [col for col in merged.columns if col.startswith("score_")]
    merged["score"] = merged[score_cols].mean(axis=1)
    return merged[["stock_id", "date", "score"]]


def _transform(frame: pd.DataFrame, col: str, mode: str) -> pd.Series:
    if mode == "rank":
        return frame[col].rank(pct=True)
    if mode == "zscore":
        std = frame[col].std()
        scale = std if pd.notna(std) and std > 1e-8 else 1.0
        return ((frame[col] - frame[col].mean()) / scale).fillna(0.0)
    raise ValueError(mode)


def _weight_grid(names: list[str], step: float, max_models: int) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    specs: list[dict[str, float]] = []
    for size in range(1, min(max_models, len(names)) + 1):
        for subset in itertools.combinations(names, size):
            for values in itertools.product(range(units + 1), repeat=size):
                if sum(values) != units or max(values) == 0:
                    continue
                spec = {name: 0.0 for name in names}
                for name, value in zip(subset, values):
                    spec[name] = value / units
                specs.append(spec)
    return list({tuple(sorted(s.items())): s for s in specs}.values())


def _build_daily_frames(root: Path, dates: list[str], model_specs: dict[str, str], price: pd.DataFrame, matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    daily: dict[str, pd.DataFrame] = {}
    for date_key in dates:
        rows = matrix[matrix["date"] == pd.to_datetime(date_key)]
        if rows.empty:
            continue
        buy_date = str(pd.to_datetime(rows.iloc[0]["buy_date"]).date())
        sell_date = str(pd.to_datetime(rows.iloc[0]["sell_date"]).date())
        label = _date_returns(price, buy_date, sell_date)
        merged = label.copy()
        for name, pattern in model_specs.items():
            pred = _load_model_prediction(root, date_key.replace("-", ""), pattern)
            if pred is None:
                continue
            pred = pred.rename(columns={"score": f"score_{name}"})
            merged = merged.merge(pred[["stock_id", f"score_{name}"]], on="stock_id", how="inner")
        if len([c for c in merged.columns if c.startswith("score_")]) >= 2:
            daily[date_key] = merged
    return daily


def _score_daily(
    daily: pd.DataFrame,
    weights: dict[str, float],
    transform: str,
    strategy: str,
    top_k: int,
    temperature: float,
) -> tuple[float, pd.DataFrame]:
    frame = daily[["stock_id", "label"]].copy()
    frame["score"] = 0.0
    active = []
    for name, weight in weights.items():
        col = f"score_{name}"
        if weight and col in daily.columns:
            frame["score"] += float(weight) * _transform(daily, col, transform)
            active.append(name)
    if not active:
        return math.nan, pd.DataFrame()
    sub = build_top_k_submission(
        frame,
        score_col="score",
        stock_col="stock_id",
        top_k=top_k,
        strategy=strategy,
        temperature=temperature,
        max_weight_sum=1.0,
    )
    validate_submission(sub, top_k=min(top_k, len(sub)), max_weight_sum=1.0)
    realized = sub.merge(frame[["stock_id", "label"]], on="stock_id", how="left")
    return float((realized["weight"] * realized["label"]).sum()), sub


def _eval_spec(daily_frames: dict[str, pd.DataFrame], weights: dict[str, float], transform: str, strategy: str, top_k: int, temperature: float) -> dict:
    records = []
    for date, frame in daily_frames.items():
        ret, sub = _score_daily(frame, weights, transform, strategy, top_k, temperature)
        if pd.notna(ret):
            records.append({"date": date, "return": ret, "top1": sub.iloc[0]["stock_id"] if not sub.empty else ""})
    returns = np.asarray([r["return"] for r in records], dtype=float)
    if len(returns) == 0:
        return {"n": 0, "mean_return": math.nan, "p05_return": math.nan, "std_return": math.nan, "robust_score": -math.inf, "records": records}
    return {
        "n": int(len(returns)),
        "mean_return": float(returns.mean()),
        "p05_return": float(np.quantile(returns, 0.05)),
        "std_return": float(returns.std()),
        "negative_rate": float((returns < 0.0).mean()),
        "robust_score": float(returns.mean() + 0.50 * np.quantile(returns, 0.05) - 0.25 * returns.std()),
        "records": records,
    }


def _make_latest_submission(latest_root: Path, model_specs: dict[str, str], weights: dict[str, float], transform: str, strategy: str, top_k: int, temperature: float) -> pd.DataFrame:
    latest_date_key = "20260529"
    merged = None
    for name, pattern in model_specs.items():
        pred = _load_model_prediction(latest_root, latest_date_key, pattern)
        if pred is None:
            continue
        pred = pred.rename(columns={"score": f"score_{name}"})
        cols = pred[["stock_id", f"score_{name}"]]
        merged = cols if merged is None else merged.merge(cols, on="stock_id", how="inner")
    if merged is None:
        raise RuntimeError("No latest predictions found")
    dummy = merged.copy()
    dummy["label"] = 0.0
    _, sub = _score_daily(dummy, weights, transform, strategy, top_k, temperature)
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description="Search robust multi-model ensemble weights on recent online windows.")
    parser.add_argument("--holdout-root", default="outputs/recent_holdout_matrix_strict_full")
    parser.add_argument("--latest-root", default="outputs/latestA_20260529_strict")
    parser.add_argument("--model-matrix", default="outputs/recent_holdout_matrix_strict_full_summary/model_top1_matrix.csv")
    parser.add_argument("--price-path", default="data/raw/stock_data.csv")
    parser.add_argument("--output-dir", default="outputs/latestA_ensemble_weights_20260529")
    parser.add_argument("--models", default="master_ms,stockmixer_ms,timexer_ms,baseline")
    parser.add_argument("--grid-step", type=float, default=0.25)
    parser.add_argument("--max-models", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    selected = [name.strip() for name in args.models.split(",") if name.strip()]
    model_specs = {name: DEFAULT_MODELS[name] for name in selected}
    matrix = pd.read_csv(args.model_matrix, parse_dates=["date", "buy_date", "sell_date"])
    dates = [str(pd.Timestamp(d).date()) for d in sorted(matrix["date"].dropna().unique())]
    price = _load_price(Path(args.price_path))
    daily = _build_daily_frames(Path(args.holdout_root), dates, model_specs, price, matrix)

    strategies = ["top1_weight", "top2_softmax", "softmax_t0.6", "equal_weight", "proportional_positive_thr0.0"]
    transforms = ["rank", "zscore"]
    rows = []
    for weights in _weight_grid(list(model_specs), args.grid_step, args.max_models):
        for transform in transforms:
            for strategy in strategies:
                metrics = _eval_spec(daily, weights, transform, strategy, args.top_k, temperature=0.6)
                rows.append({"weights": weights, "transform": transform, "strategy": strategy, **metrics})
    rows.sort(key=lambda r: r["robust_score"], reverse=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compact = [{k: v for k, v in row.items() if k != "records"} for row in rows]
    (output_dir / "leaderboard.json").write_text(json.dumps(compact[: args.top_n], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "all_results.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")

    for idx, row in enumerate(rows[: min(args.top_n, 10)], start=1):
        sub = _make_latest_submission(Path(args.latest_root), model_specs, row["weights"], row["transform"], row["strategy"], args.top_k, temperature=0.6)
        sub.to_csv(output_dir / f"candidate_{idx}.csv", index=False)
    summary = {
        "models": selected,
        "n_windows": len(daily),
        "best": compact[0] if compact else None,
        "best_records": rows[0]["records"] if rows else [],
        "warning": "Only five online windows; ensemble weights are auxiliary candidates, not an authority over hard all-in gates.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
