from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml_config


MODEL_SPECS = {
    "master_official": {
        "script": "scripts/train_master_baseline.py",
        "config": "configs/holdout_20260517_master_alpha_official_rank.yaml",
    },
    "stockmixer_lite": {
        "script": "scripts/train_stockmixer_baseline.py",
        "config": "configs/holdout_20260517_stockmixer_alpha_industry_lite.yaml",
    },
    "stockmixer_official": {
        "script": "scripts/train_stockmixer_baseline.py",
        "config": "configs/holdout_20260517_stockmixer_alpha_official_rank.yaml",
    },
    "itransformer": {
        "script": "scripts/train_itransformer_backbone.py",
        "config": "configs/holdout_20260517_itransformer_alpha.yaml",
    },
    "timexer_fast": {
        "script": "scripts/train_timexer_backbone.py",
        "config": "configs/holdout_20260517_timexer_alpha_fast.yaml",
    },
    "lstm": {
        "script": "scripts/train_lstm_baseline.py",
        "config": "configs/holdout_20260517_lstm_alpha.yaml",
    },
}

COL_STOCK = "股票代码"
COL_DATE = "日期"
COL_OPEN = "开盘"


def _normalize_stock_id(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _read_price_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = df[[COL_STOCK, COL_DATE, COL_OPEN]].copy()
    out[COL_STOCK] = out[COL_STOCK].map(_normalize_stock_id)
    out[COL_DATE] = pd.to_datetime(out[COL_DATE])
    out[COL_OPEN] = pd.to_numeric(out[COL_OPEN], errors="coerce")
    return out.dropna(subset=[COL_STOCK, COL_DATE, COL_OPEN])


def _resolve_dates(price_df: pd.DataFrame, trade_date: str, buy_offset: int, sell_offset: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = sorted(price_df[COL_DATE].drop_duplicates().tolist())
    t_date = pd.Timestamp(trade_date)
    idx = dates.index(t_date)
    return dates[idx + buy_offset], dates[idx + sell_offset]


def _realized_return(price_df: pd.DataFrame, stock_id: str, buy_date: pd.Timestamp, sell_date: pd.Timestamp) -> float:
    stock = _normalize_stock_id(stock_id)
    buy = price_df[(price_df[COL_STOCK] == stock) & (price_df[COL_DATE] == buy_date)][COL_OPEN]
    sell = price_df[(price_df[COL_STOCK] == stock) & (price_df[COL_DATE] == sell_date)][COL_OPEN]
    if buy.empty or sell.empty:
        return float("nan")
    return float((sell.iloc[0] - buy.iloc[0]) / buy.iloc[0])


def _write_seed_config(base_config_path: Path, model_name: str, seed: int, output_dir: Path) -> Path:
    cfg = load_yaml_config(base_config_path)
    cfg["seed"] = int(seed)
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["reuse_processed"] = True
    cfg["deep"] = dict(cfg["deep"])
    if "seeds" in cfg["deep"]:
        cfg["deep"]["seeds"] = [int(seed)]
    cfg["output"] = dict(cfg["output"])
    prefix = f"{model_name}_seed_{seed}"
    cfg["output"]["prediction_path"] = str(output_dir / f"{prefix}_predictions.csv")
    cfg["output"]["latest_prediction_path"] = str(output_dir / f"{prefix}_latest_predictions.csv")
    cfg["output"]["metrics_path"] = str(output_dir / f"{prefix}_metrics.json")
    cfg["output"]["submission_path"] = str(output_dir / f"{prefix}_submission.csv")
    path = output_dir / f"{prefix}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _run_seed(python_exe: str, script: str, config_path: Path) -> None:
    subprocess.run([python_exe, script, "--config", str(config_path)], cwd=PROJECT_ROOT, check=True)


def _load_latest(path: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    df["stock_id"] = df["stock_id"].map(_normalize_stock_id)
    df["date"] = pd.to_datetime(df["date"])
    df[f"score_seed_{seed}"] = pd.to_numeric(df["score"], errors="coerce")
    return df[["stock_id", "date", f"score_seed_{seed}"]].copy()


def _top_profile(df: pd.DataFrame, score_col: str) -> dict:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    scores = ranked[score_col].to_numpy(dtype=float)
    scale = float(np.std(scores)) if len(scores) > 1 and float(np.std(scores)) > 1e-8 else 1.0
    margin = float((scores[0] - scores[1]) / scale) if len(scores) > 1 else float("inf")
    strength = float((scores[0] - np.mean(scores)) / scale) if len(scores) > 0 else float("nan")
    return {
        "stock_id": str(ranked.loc[0, "stock_id"]),
        "score": float(scores[0]),
        "margin_z": margin,
        "top_strength": strength,
    }


def _aggregate_latest(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["stock_id", "date"], how="inner")
    score_cols = [col for col in merged.columns if col.startswith("score_seed_")]
    merged["score_mean"] = merged[score_cols].mean(axis=1)
    merged["score_std"] = merged[score_cols].std(axis=1).fillna(0.0)
    return merged


def _evaluate_model(
    name: str,
    spec: dict,
    seeds: list[int],
    args: argparse.Namespace,
    price_df: pd.DataFrame,
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
) -> dict:
    model_dir = Path(args.output_dir) / name
    model_dir.mkdir(parents=True, exist_ok=True)
    latest_frames = []
    seed_rows = []
    for seed in seeds:
        config_path = _write_seed_config(Path(spec["config"]), name, seed, model_dir)
        latest_path = model_dir / f"{name}_seed_{seed}_latest_predictions.csv"
        pred_path = model_dir / f"{name}_seed_{seed}_predictions.csv"
        if not args.aggregate_only and not (args.skip_existing and latest_path.exists() and pred_path.exists()):
            _run_seed(args.python_exe, spec["script"], config_path)
        frame = _load_latest(latest_path, seed)
        latest_frames.append(frame)
        profile = _top_profile(frame, f"score_seed_{seed}")
        profile.update(
            {
                "seed": seed,
                "return": _realized_return(price_df, profile["stock_id"], buy_date, sell_date),
            }
        )
        seed_rows.append(profile)

    aggregated = _aggregate_latest(latest_frames)
    mean_profile = _top_profile(aggregated, "score_mean")
    mean_profile["return"] = _realized_return(price_df, mean_profile["stock_id"], buy_date, sell_date)
    votes = Counter(row["stock_id"] for row in seed_rows)
    vote_stock, vote_count = votes.most_common(1)[0]
    vote_return = _realized_return(price_df, vote_stock, buy_date, sell_date)
    score_cols = [col for col in aggregated.columns if col.startswith("score_seed_")]
    aggregated.to_csv(model_dir / f"{name}_multiseed_latest_predictions.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(model_dir / f"{name}_seed_top1.csv", index=False)
    return {
        "model": name,
        "seeds": seeds,
        "seed_top1": seed_rows,
        "mean_top1": mean_profile,
        "vote_top1": {
            "stock_id": vote_stock,
            "votes": int(vote_count),
            "vote_share": float(vote_count / len(seeds)),
            "return": vote_return,
        },
        "top1_unique_count": int(len(votes)),
        "mean_top1_seed_std": float(aggregated.loc[aggregated["stock_id"] == mean_profile["stock_id"], score_cols].std(axis=1).iloc[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed Top1/all-in holdout evaluation.")
    parser.add_argument("--models", default="master_official,stockmixer_lite,stockmixer_official,itransformer,timexer_fast,lstm")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--output-dir", default="outputs/holdout_20260517/multiseed_top1")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--price-path", default="data/raw/stock_data.csv")
    parser.add_argument("--trade-date", default="2026-05-15")
    parser.add_argument("--buy-offset", type=int, default=1)
    parser.add_argument("--sell-offset", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    price_df = _read_price_frame(Path(args.price_path))
    buy_date, sell_date = _resolve_dates(price_df, args.trade_date, args.buy_offset, args.sell_offset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in selected_models:
        if name not in MODEL_SPECS:
            raise KeyError(f"Unknown model {name}. Available: {sorted(MODEL_SPECS)}")
        results.append(_evaluate_model(name, MODEL_SPECS[name], seeds, args, price_df, buy_date, sell_date))

    leaderboard = []
    for result in results:
        leaderboard.append(
            {
                "model": result["model"],
                "mean_top1_stock": result["mean_top1"]["stock_id"],
                "mean_top1_return": result["mean_top1"]["return"],
                "vote_top1_stock": result["vote_top1"]["stock_id"],
                "vote_top1_return": result["vote_top1"]["return"],
                "vote_share": result["vote_top1"]["vote_share"],
                "top1_unique_count": result["top1_unique_count"],
                "mean_top1_margin_z": result["mean_top1"]["margin_z"],
                "mean_top1_strength": result["mean_top1"]["top_strength"],
                "mean_top1_seed_std": result["mean_top1_seed_std"],
            }
        )
    leaderboard_df = pd.DataFrame(leaderboard).sort_values("mean_top1_return", ascending=False)
    summary = {
        "trade_date": args.trade_date,
        "buy_date": str(pd.Timestamp(buy_date).date()),
        "sell_date": str(pd.Timestamp(sell_date).date()),
        "seeds": seeds,
        "results": results,
        "leaderboard": leaderboard_df.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    leaderboard_df.to_csv(output_dir / "leaderboard.csv", index=False)
    print(leaderboard_df.to_string(index=False))


if __name__ == "__main__":
    main()
