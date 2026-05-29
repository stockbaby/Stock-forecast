from __future__ import annotations

import argparse
import json
import math
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

from src.portfolio.construct import build_top_k_submission


COL_STOCK = "股票代码"
COL_DATE = "日期"
COL_OPEN = "开盘"
LABEL_COL = "y_ret_a_stage_round1_open_open"

MODEL_PATHS = {
    "baseline": "baseline_predictions.csv",
    "master": "master_alpha_official_rank_predictions.csv",
    "stockmixer_lite": "stockmixer_alpha_industry_lite_predictions.csv",
    "stockmixer_fast": "stockmixer_alpha_fast_predictions.csv",
    "stockmixer_official": "stockmixer_alpha_official_rank_predictions.csv",
    "timexer": "timexer_alpha_fast_predictions.csv",
    "itransformer": "itransformer_alpha_predictions.csv",
    "lstm": "lstm_alpha_predictions.csv",
}

TRAIN_MODEL_SPECS = {
    "baseline": {
        "script": "scripts/train_baseline.py",
        "base_config": "configs/a_stage_round1.yaml",
        "output_stem": "baseline",
        "build_dataset": True,
    },
    "master": {
        "script": "scripts/train_master_baseline.py",
        "base_config": "configs/master_alpha_official_rank.yaml",
        "output_stem": "master_alpha_official_rank",
    },
    "stockmixer_lite": {
        "script": "scripts/train_stockmixer_baseline.py",
        "base_config": "configs/stockmixer_alpha_industry_lite.yaml",
        "output_stem": "stockmixer_alpha_industry_lite",
    },
    "stockmixer_fast": {
        "script": "scripts/train_stockmixer_baseline.py",
        "base_config": "configs/stockmixer_alpha_fast.yaml",
        "output_stem": "stockmixer_alpha_fast",
    },
    "stockmixer_official": {
        "script": "scripts/train_stockmixer_baseline.py",
        "base_config": "configs/stockmixer_alpha_official_rank.yaml",
        "output_stem": "stockmixer_alpha_official_rank",
    },
    "timexer": {
        "script": "scripts/train_timexer_backbone.py",
        "base_config": "configs/timexer_alpha_fast.yaml",
        "output_stem": "timexer_alpha_fast",
    },
}

MULTISEED_MODELS = {
    "master_multiseed": "master_official",
    "stockmixer_lite_multiseed": "stockmixer_lite",
    "stockmixer_official_multiseed": "stockmixer_official",
    "timexer_multiseed": "timexer_fast",
    "itransformer_multiseed": "itransformer",
    "lstm_multiseed": "lstm",
}


def _normalize_stock_id(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def _write_csv_auto(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _load_price_frame(path: Path) -> pd.DataFrame:
    df = _read_csv_auto(path)
    out = df[[COL_STOCK, COL_DATE, COL_OPEN]].copy()
    out[COL_STOCK] = out[COL_STOCK].map(_normalize_stock_id)
    out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce")
    out[COL_OPEN] = pd.to_numeric(out[COL_OPEN], errors="coerce")
    return out.dropna(subset=[COL_STOCK, COL_DATE, COL_OPEN])


def _prepare_cutoff_raw(
    source_price_path: Path,
    source_index_path: Path,
    source_stock_list_path: Path,
    output_dir: Path,
    trade_date: str,
) -> tuple[Path, Path | None, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    t_date = pd.Timestamp(trade_date)
    price = _read_csv_auto(source_price_path)
    date_col = COL_DATE if COL_DATE in price.columns else "date"
    price[date_col] = pd.to_datetime(price[date_col], errors="coerce")
    price_cut = price[price[date_col] <= t_date].copy()
    stock_path = output_dir / "stock_data.csv"
    _write_csv_auto(price_cut, stock_path)

    index_path = None
    if source_index_path.exists():
        index = _read_csv_auto(source_index_path)
        index_date_col = COL_DATE if COL_DATE in index.columns else ("date" if "date" in index.columns else "datetime")
        index[index_date_col] = pd.to_datetime(index[index_date_col], errors="coerce")
        index_cut = index[index[index_date_col] <= t_date].copy()
        index_path = output_dir / "hs300_index.csv"
        _write_csv_auto(index_cut, index_path)

    stock_list_path = None
    if source_stock_list_path.exists():
        stock_list_path = output_dir / "hs300_stock_list.csv"
        _write_csv_auto(_read_csv_auto(source_stock_list_path), stock_list_path)
    return output_dir, index_path, stock_list_path


def _write_run_config(
    model_name: str,
    spec: dict,
    trade_date: str,
    seed: int,
    raw_dir: Path,
    index_path: Path | None,
    stock_list_path: Path | None,
    output_dir: Path,
    config_dir: Path,
    reuse_processed: bool,
) -> Path:
    cfg = yaml.safe_load(Path(spec["base_config"]).read_text(encoding="utf-8"))
    cfg["seed"] = int(seed)
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["raw_dir"] = str(raw_dir)
    cfg["data"]["processed_path"] = str(output_dir / f"{model_name}_dataset.csv")
    cfg["data"]["benchmark_end_date"] = trade_date
    cfg["data"]["reuse_processed"] = bool(reuse_processed)
    if index_path:
        cfg["data"]["market_index_path"] = str(index_path)
    if stock_list_path:
        cfg["data"]["industry_map_path"] = str(stock_list_path)
    cfg["output"] = dict(cfg["output"])
    prefix = f"{model_name}_seed_{seed}"
    cfg["output"]["metrics_path"] = str(output_dir / f"{prefix}_metrics.json")
    cfg["output"]["prediction_path"] = str(output_dir / f"{prefix}_predictions.csv")
    cfg["output"]["latest_prediction_path"] = str(output_dir / f"{prefix}_latest_predictions.csv")
    cfg["output"]["submission_path"] = str(output_dir / f"{prefix}_submission.csv")
    cfg["output"]["inference_date"] = trade_date
    cfg["deep"] = dict(cfg.get("deep", {}))
    if "seeds" in cfg["deep"]:
        cfg["deep"]["seeds"] = [int(seed)]
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{prefix}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _run_strict_training(args: argparse.Namespace, windows: list[str], seeds: list[int]) -> None:
    selected_models = [item.strip() for item in args.train_models.split(",") if item.strip()]
    source_price = Path(args.price_path)
    source_index = Path(args.source_index_path)
    source_stock_list = Path(args.source_stock_list_path)
    base_output = Path(args.strict_output_dir)
    for trade_date in windows:
        window_key = trade_date.replace("-", "")
        raw_dir, index_path, stock_list_path = _prepare_cutoff_raw(
            source_price,
            source_index,
            source_stock_list,
            base_output / "raw_cutoffs" / window_key,
            trade_date,
        )
        for model_name in selected_models:
            if model_name not in TRAIN_MODEL_SPECS:
                raise KeyError(f"Unknown train model {model_name}. Available: {sorted(TRAIN_MODEL_SPECS)}")
            spec = TRAIN_MODEL_SPECS[model_name]
            for seed in seeds:
                model_output_dir = base_output / window_key / model_name
                config_path = _write_run_config(
                    model_name,
                    spec,
                    trade_date,
                    seed,
                    raw_dir,
                    index_path,
                    stock_list_path,
                    model_output_dir,
                    base_output / "configs" / window_key,
                    reuse_processed=args.reuse_processed,
                )
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if args.fast_epochs is not None:
                    cfg["deep"]["epochs"] = int(args.fast_epochs)
                if args.fast_batch_size is not None:
                    cfg["deep"]["batch_size"] = int(args.fast_batch_size)
                if args.fast_valid_days is not None:
                    cfg["training"]["valid_days"] = int(args.fast_valid_days)
                if args.fast_recent_train_days is not None:
                    cfg["training"]["recent_train_days"] = int(args.fast_recent_train_days)
                config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
                latest_path = model_output_dir / f"{model_name}_seed_{seed}_latest_predictions.csv"
                pred_path = model_output_dir / f"{model_name}_seed_{seed}_predictions.csv"
                if args.skip_existing and latest_path.exists() and pred_path.exists():
                    continue
                if spec.get("build_dataset"):
                    subprocess.run(
                        [args.python_exe, "scripts/build_dataset.py", "--config", str(config_path)],
                        cwd=PROJECT_ROOT,
                        check=True,
                    )
                subprocess.run(
                    [args.python_exe, spec["script"], "--config", str(config_path)],
                    cwd=PROJECT_ROOT,
                    check=True,
                )


def _resolve_dates(price_df: pd.DataFrame, trade_date: str, buy_offset: int, sell_offset: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = sorted(price_df[COL_DATE].drop_duplicates().tolist())
    t_date = pd.Timestamp(trade_date)
    if t_date not in dates:
        raise ValueError(f"Trade date {trade_date} not found in price data.")
    idx = dates.index(t_date)
    return dates[idx + buy_offset], dates[idx + sell_offset]


def _realized_return(price_df: pd.DataFrame, stock_id: str, buy_date: pd.Timestamp, sell_date: pd.Timestamp) -> float:
    stock = _normalize_stock_id(stock_id)
    buy = price_df[(price_df[COL_STOCK] == stock) & (price_df[COL_DATE] == buy_date)][COL_OPEN]
    sell = price_df[(price_df[COL_STOCK] == stock) & (price_df[COL_DATE] == sell_date)][COL_OPEN]
    if buy.empty or sell.empty:
        return math.nan
    return float((sell.iloc[0] - buy.iloc[0]) / buy.iloc[0])


def _load_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    if "label" in df.columns and LABEL_COL not in df.columns:
        df = df.rename(columns={"label": LABEL_COL})
    if "score" not in df.columns and "score_mean" in df.columns:
        df = df.rename(columns={"score_mean": "score"})
    missing = {"stock_id", "date", "score"}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df.copy()
    out["stock_id"] = out["stock_id"].map(_normalize_stock_id)
    out["date"] = pd.to_datetime(out["date"])
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    return out


def _daily_frame(path: Path, trade_date: str, latest_path: Path | None = None) -> pd.DataFrame | None:
    t_date = pd.Timestamp(trade_date)
    if latest_path and latest_path.exists():
        latest = _load_prediction(latest_path)
        latest_day = latest[latest["date"] == t_date].copy()
        if not latest_day.empty:
            return latest_day
    if not path.exists():
        return None
    pred = _load_prediction(path)
    daily = pred[pred["date"] == t_date].copy()
    return daily if not daily.empty else None


def _top_profile(daily: pd.DataFrame, score_col: str = "score") -> dict:
    ranked = daily.dropna(subset=[score_col]).sort_values(score_col, ascending=False).reset_index(drop=True)
    if ranked.empty:
        return {"stock_id": None, "margin_z": math.nan, "top_strength": math.nan}
    scores = ranked[score_col].to_numpy(dtype=float)
    scale = float(np.std(scores)) if len(scores) > 1 and float(np.std(scores)) > 1e-8 else 1.0
    return {
        "stock_id": str(ranked.loc[0, "stock_id"]),
        "margin_z": float((scores[0] - scores[1]) / scale) if len(scores) > 1 else math.inf,
        "top_strength": float((scores[0] - np.mean(scores)) / scale),
    }


def _submission_return(daily: pd.DataFrame, strategy: str, top_k: int, temperature: float) -> tuple[float, list[str], list[float]]:
    if daily.empty:
        return math.nan, [], []
    submission = build_top_k_submission(
        daily,
        score_col="score",
        stock_col="stock_id",
        top_k=top_k,
        max_weight_sum=1.0,
        strategy=strategy,
        temperature=temperature,
    )
    if LABEL_COL not in daily.columns or submission.empty:
        return math.nan, submission["stock_id"].tolist(), submission["weight"].tolist()
    merged = submission.merge(daily[["stock_id", LABEL_COL]], on="stock_id", how="left")
    return float((merged["weight"] * merged[LABEL_COL]).sum()), submission["stock_id"].tolist(), submission["weight"].tolist()


def _weighted_realized_return(
    price_df: pd.DataFrame,
    stocks: list[str],
    weights: list[float],
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
) -> float:
    if not stocks or not weights:
        return math.nan
    returns = [
        _realized_return(price_df, stock_id, buy_date, sell_date)
        for stock_id in stocks
    ]
    if any(pd.isna(value) for value in returns):
        return math.nan
    return float(np.dot(np.asarray(weights, dtype=float), np.asarray(returns, dtype=float)))


def _multiseed_daily(multiseed_dir: Path, model_dir_name: str, trade_date: str, seeds: list[int]) -> tuple[pd.DataFrame | None, dict]:
    frames = []
    seed_tops = []
    for seed in seeds:
        pred_path = multiseed_dir / model_dir_name / f"{model_dir_name}_seed_{seed}_predictions.csv"
        latest_path = multiseed_dir / model_dir_name / f"{model_dir_name}_seed_{seed}_latest_predictions.csv"
        daily = _daily_frame(pred_path, trade_date, latest_path)
        if daily is None:
            continue
        score_col = f"score_seed_{seed}"
        daily = daily[["stock_id", "date", "score"]].rename(columns={"score": score_col})
        profile = _top_profile(daily, score_col)
        if profile["stock_id"]:
            seed_tops.append(profile["stock_id"])
        frames.append(daily)
    if not frames:
        return None, {"seed_vote_share": math.nan, "seed_unique_count": math.nan, "seed_top1": []}
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["stock_id", "date"], how="inner")
    score_cols = [col for col in merged.columns if col.startswith("score_seed_")]
    merged["score"] = merged[score_cols].mean(axis=1)
    counts = Counter(seed_tops)
    vote_share = counts.most_common(1)[0][1] / len(seed_tops) if seed_tops else math.nan
    return merged, {
        "seed_vote_share": float(vote_share),
        "seed_unique_count": int(len(counts)),
        "seed_top1": seed_tops,
    }


def _strict_seed_daily(strict_dir: Path, trade_date: str, model_name: str, seed: int) -> pd.DataFrame | None:
    window_key = trade_date.replace("-", "")
    model_dir = strict_dir / window_key / model_name
    pred_path = model_dir / f"{model_name}_seed_{seed}_predictions.csv"
    latest_path = model_dir / f"{model_name}_seed_{seed}_latest_predictions.csv"
    return _daily_frame(pred_path, trade_date, latest_path)


def _strict_multiseed_daily(strict_dir: Path, trade_date: str, model_name: str, seeds: list[int]) -> tuple[pd.DataFrame | None, dict]:
    frames = []
    seed_tops = []
    for seed in seeds:
        daily = _strict_seed_daily(strict_dir, trade_date, model_name, seed)
        if daily is None:
            continue
        score_col = f"score_seed_{seed}"
        frames.append(daily[["stock_id", "date", "score"]].rename(columns={"score": score_col}))
        profile = _top_profile(frames[-1], score_col)
        if profile["stock_id"]:
            seed_tops.append(profile["stock_id"])
    if not frames:
        return None, {"seed_vote_share": math.nan, "seed_unique_count": math.nan, "seed_top1": []}
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["stock_id", "date"], how="inner")
    score_cols = [col for col in merged.columns if col.startswith("score_seed_")]
    merged["score"] = merged[score_cols].mean(axis=1)
    counts = Counter(seed_tops)
    vote_share = counts.most_common(1)[0][1] / len(seed_tops) if seed_tops else math.nan
    return merged, {"seed_vote_share": float(vote_share), "seed_unique_count": int(len(counts)), "seed_top1": seed_tops}


def _rank_blend(model_frames: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame | None:
    merged = None
    for model_name, weight in weights.items():
        if model_name not in model_frames or model_frames[model_name] is None:
            return None
        frame = model_frames[model_name][["stock_id", "date", "score"]].copy()
        frame[f"score_{model_name}"] = frame.groupby("date")["score"].rank(pct=True) * float(weight)
        frame = frame.drop(columns=["score"])
        merged = frame if merged is None else merged.merge(frame, on=["stock_id", "date"], how="inner")
    if merged is None:
        return None
    score_cols = [col for col in merged.columns if col.startswith("score_")]
    merged["score"] = merged[score_cols].sum(axis=1)
    return merged[["stock_id", "date", "score"]]


def _candidate_passes(profile: dict, agreement: int, agreement_share: float) -> tuple[bool, float, list[str]]:
    reasons = []
    if profile["margin_z"] < 0.01:
        reasons.append("low_margin")
    if profile["top_strength"] < 2.1:
        reasons.append("low_top_strength")
    if agreement < 2:
        reasons.append("low_candidate_agreement")
    gate_score = float(profile["margin_z"]) + 0.5 * float(profile["top_strength"]) + float(agreement_share)
    return not reasons, gate_score, reasons


def _strict_dynamic_record(
    strict_dir: Path,
    trade_date: str,
    seeds: list[int],
    fallback_return: float,
    price_df: pd.DataFrame,
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
) -> dict:
    model_frames = {}
    for model_name in ["master", "stockmixer_lite", "stockmixer_fast", "stockmixer_official", "timexer"]:
        daily, _ = _strict_multiseed_daily(strict_dir, trade_date, model_name, seeds)
        if daily is not None:
            model_frames[model_name] = daily
    primary = model_frames.get("master")
    candidates = {
        "stockmixer_lite_timexer_rank": {"stockmixer_lite": 0.75, "timexer": 0.25},
        "stockmixer_fast_lite_rank": {"stockmixer_fast": 0.25, "stockmixer_lite": 0.75},
        "stockmixer_official_lite_rank": {"stockmixer_official": 0.25, "stockmixer_lite": 0.75},
        "stockmixer_portfolio_lite_rank": {"stockmixer_fast": 0.5, "stockmixer_lite": 0.5},
    }
    options = []
    blended = {}
    for name, weights in candidates.items():
        frame = _rank_blend(model_frames, weights)
        if frame is None:
            continue
        blended[name] = frame
        profile = _top_profile(frame)
        options.append({"name": name, **profile})
    top_counts = Counter(item["stock_id"] for item in options if item.get("stock_id"))
    for item in options:
        agreement = top_counts.get(item["stock_id"], 0)
        agreement_share = agreement / max(len(options), 1)
        passed, gate_score, reasons = _candidate_passes(item, agreement, agreement_share)
        item.update(
            {
                "passed": passed,
                "gate_score": gate_score,
                "reasons": reasons,
                "candidate_top1_agreement": agreement,
                "candidate_top1_agreement_share": agreement_share,
            }
        )
    passed_options = [item for item in options if item["passed"]]
    if passed_options:
        selected = max(passed_options, key=lambda item: item["gate_score"])
        top_stock = selected["stock_id"]
        selected_name = selected["name"]
        allin_selected = True
        realized = _realized_return(price_df, top_stock, buy_date, sell_date) if top_stock else math.nan
    else:
        selected = _top_profile(primary) if primary is not None else {"stock_id": None}
        top_stock = selected["stock_id"]
        selected_name = "primary"
        allin_selected = False
        realized = fallback_return
    return {
        "date": trade_date,
        "selected": selected_name,
        "top1": top_stock,
        "return": realized,
        "fallback_return": fallback_return,
        "hit_vs_fallback": bool(realized > fallback_return) if pd.notna(realized) and pd.notna(fallback_return) else False,
        "candidate_top1_agreement": max([item.get("candidate_top1_agreement", 0) for item in options] or [0]),
        "candidate_top1_agreement_share": max([item.get("candidate_top1_agreement_share", 0.0) for item in options] or [0.0]),
        "best_margin_z": max([item.get("margin_z", 0.0) for item in options if pd.notna(item.get("margin_z", np.nan))] or [0.0]),
        "best_top_strength": max([item.get("top_strength", 0.0) for item in options if pd.notna(item.get("top_strength", np.nan))] or [0.0]),
        "allin_selected": allin_selected,
    }


def _stats(values: list[float]) -> dict:
    returns = np.asarray([x for x in values if pd.notna(x)], dtype=float)
    if len(returns) == 0:
        return {"n": 0, "hit_rate": math.nan, "mean_return": math.nan, "p05_return": math.nan, "negative_rate": math.nan, "max_drawdown": math.nan}
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    return {
        "n": int(len(returns)),
        "hit_rate": float(np.mean(returns > 0.0)),
        "mean_return": float(np.mean(returns)),
        "p05_return": float(np.quantile(returns, 0.05)),
        "negative_rate": float(np.mean(returns < 0.0)),
        "max_drawdown": float(np.min(equity / np.maximum(peak, 1e-12) - 1.0)),
    }


def _to_markdown(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format(value, floatfmt) if pd.notna(value) else "")
            else:
                values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _load_dynamic_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    summary = json.loads(path.read_text(encoding="utf-8"))
    records = {row["date"]: row for row in summary.get("records", [])}
    latest = summary.get("latest_record")
    if latest:
        records[latest["date"]] = latest
    return records


def _write_doc(path: Path, windows: list[str], summary_df: pd.DataFrame, model_df: pd.DataFrame, dynamic_df: pd.DataFrame) -> None:
    lines = [
        "# Top1 Gate Recent Holdout Matrix",
        "",
        f"Windows: {', '.join(windows)}",
        "",
        "## Core Metrics",
        "",
        _to_markdown(summary_df, floatfmt=".6f"),
        "",
        "## Model Top1 Rows",
        "",
        _to_markdown(model_df, floatfmt=".6f"),
        "",
        "## Dynamic Switch Rows",
        "",
        _to_markdown(dynamic_df, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "- `return` is all-in Top1 realized open-to-open return for the requested T window.",
        "- `hit_vs_fallback` checks whether all-in Top1 beat the MASTER dynamic-risk-budget fallback on the same T.",
        "- `candidate_top1_agreement_share` is the dynamic-switch cross-candidate consistency signal.",
        "- Multi-seed rows are confirmation diagnostics; they should not be used as a standalone all-in trigger unless the matrix keeps showing positive p05 and low negative rate.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recent Top1 holdout windows and summarize all-in gate quality.")
    parser.add_argument("--dates", default="2026-04-17,2026-04-24,2026-04-30,2026-05-08,2026-05-15")
    parser.add_argument("--holdout-dir", default="outputs/holdout_20260517")
    parser.add_argument("--multiseed-dir", default="outputs/holdout_20260517/multiseed_top1")
    parser.add_argument("--dynamic-summary", default="outputs/holdout_20260517/dynamic_candidate_switch/summary.json")
    parser.add_argument("--price-path", default="data/raw/stock_data.csv")
    parser.add_argument("--buy-offset", type=int, default=1)
    parser.add_argument("--sell-offset", type=int, default=5)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fallback-strategy", default="dynamic_risk_budget")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output-dir", default="outputs/recent_holdout_matrix")
    parser.add_argument("--doc-path", default="docs/top1_gate_holdout_matrix.md")
    parser.add_argument("--strict-train", action="store_true", help="Train per-T models from raw data truncated at each T.")
    parser.add_argument("--strict-output-dir", default="outputs/recent_holdout_matrix/strict_walk_forward")
    parser.add_argument("--train-models", default="master,stockmixer_lite,stockmixer_fast,stockmixer_official,timexer")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--source-index-path", default="data/raw/hs300_index.csv")
    parser.add_argument("--source-stock-list-path", default="data/raw/hs300_stock_list.csv")
    parser.add_argument("--reuse-processed", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fast-epochs", type=int)
    parser.add_argument("--fast-batch-size", type=int)
    parser.add_argument("--fast-valid-days", type=int)
    parser.add_argument("--fast-recent-train-days", type=int)
    args = parser.parse_args()

    windows = [item.strip() for item in args.dates.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if args.strict_train:
        _run_strict_training(args, windows, seeds)
        args.holdout_dir = args.strict_output_dir
    holdout_dir = Path(args.holdout_dir)
    price_df = _load_price_frame(Path(args.price_path))
    dynamic_records = _load_dynamic_records(Path(args.dynamic_summary))
    model_rows = []
    dynamic_rows = []

    for trade_date in windows:
        buy_date, sell_date = _resolve_dates(price_df, trade_date, args.buy_offset, args.sell_offset)
        if args.strict_train:
            primary_daily, _ = _strict_multiseed_daily(Path(args.strict_output_dir), trade_date, "master", seeds)
        else:
            primary_path = holdout_dir / MODEL_PATHS["master"]
            primary_latest = holdout_dir / "master_alpha_official_rank_latest_predictions.csv"
            primary_daily = _daily_frame(primary_path, trade_date, primary_latest)
        fallback_return, fallback_stocks, fallback_weights = _submission_return(
            primary_daily if primary_daily is not None else pd.DataFrame(),
            args.fallback_strategy,
            args.top_k,
            args.temperature,
        )
        if pd.isna(fallback_return):
            fallback_return = _weighted_realized_return(price_df, fallback_stocks, fallback_weights, buy_date, sell_date)

        for model_name, filename in MODEL_PATHS.items():
            if args.strict_train:
                if model_name not in TRAIN_MODEL_SPECS:
                    continue
                daily = _strict_seed_daily(Path(args.strict_output_dir), trade_date, model_name, seeds[0])
            else:
                latest = holdout_dir / filename.replace("_predictions.csv", "_latest_predictions.csv")
                daily = _daily_frame(holdout_dir / filename, trade_date, latest)
            if daily is None:
                model_rows.append({"date": trade_date, "model": model_name, "missing": True})
                continue
            profile = _top_profile(daily)
            realized = _realized_return(price_df, profile["stock_id"], buy_date, sell_date)
            label_return = float(daily.loc[daily["stock_id"] == profile["stock_id"], LABEL_COL].iloc[0]) if LABEL_COL in daily.columns and not daily.loc[daily["stock_id"] == profile["stock_id"], LABEL_COL].empty else math.nan
            model_rows.append(
                {
                    "date": trade_date,
                    "buy_date": str(buy_date.date()),
                    "sell_date": str(sell_date.date()),
                    "model": model_name,
                    "top1": profile["stock_id"],
                    "return": realized if pd.notna(realized) else label_return,
                    "fallback_return": fallback_return,
                    "hit_vs_fallback": bool((realized if pd.notna(realized) else label_return) > fallback_return),
                    "margin_z": profile["margin_z"],
                    "top_strength": profile["top_strength"],
                    "missing": False,
                }
            )

        for row_name, model_dir_name in MULTISEED_MODELS.items():
            if args.strict_train:
                strict_model_name = row_name.replace("_multiseed", "")
                if strict_model_name not in TRAIN_MODEL_SPECS:
                    continue
                daily, seed_info = _strict_multiseed_daily(Path(args.strict_output_dir), trade_date, strict_model_name, seeds)
            else:
                daily, seed_info = _multiseed_daily(Path(args.multiseed_dir), model_dir_name, trade_date, seeds)
            if daily is None:
                model_rows.append({"date": trade_date, "model": row_name, "missing": True})
                continue
            profile = _top_profile(daily)
            realized = _realized_return(price_df, profile["stock_id"], buy_date, sell_date)
            model_rows.append(
                {
                    "date": trade_date,
                    "buy_date": str(buy_date.date()),
                    "sell_date": str(sell_date.date()),
                    "model": row_name,
                    "top1": profile["stock_id"],
                    "return": realized,
                    "fallback_return": fallback_return,
                    "hit_vs_fallback": bool(realized > fallback_return),
                    "margin_z": profile["margin_z"],
                    "top_strength": profile["top_strength"],
                    "seed_vote_share": seed_info["seed_vote_share"],
                    "seed_unique_count": seed_info["seed_unique_count"],
                    "seed_top1": ";".join(seed_info["seed_top1"]),
                    "missing": False,
                }
            )

        dyn = (
            _strict_dynamic_record(
                Path(args.strict_output_dir),
                trade_date,
                seeds,
                fallback_return,
                price_df,
                buy_date,
                sell_date,
            )
            if args.strict_train
            else dynamic_records.get(trade_date)
        )
        if dyn:
            if args.strict_train:
                dynamic_rows.append(dyn)
                continue
            stocks = dyn.get("stocks", [])
            top_stock = stocks[0] if stocks else None
            dyn_return = _realized_return(price_df, top_stock, buy_date, sell_date) if top_stock else math.nan
            if pd.isna(dyn_return):
                dyn_return = dyn.get("return", math.nan)
            candidates = dyn.get("candidates", [])
            best_agreement = max([item.get("candidate_top1_agreement", 0) for item in candidates] or [0])
            best_agreement_share = max([item.get("candidate_top1_agreement_share", 0.0) for item in candidates] or [0.0])
            best_margin_z = max([item.get("margin_z", 0.0) for item in candidates if pd.notna(item.get("margin_z", np.nan))] or [0.0])
            best_top_strength = max([item.get("top_strength", 0.0) for item in candidates if pd.notna(item.get("top_strength", np.nan))] or [0.0])
            dynamic_rows.append(
                {
                    "date": trade_date,
                    "selected": dyn.get("selected"),
                    "top1": top_stock,
                    "return": dyn_return,
                    "fallback_return": fallback_return,
                    "hit_vs_fallback": bool(dyn_return > fallback_return),
                    "candidate_top1_agreement": best_agreement,
                    "candidate_top1_agreement_share": best_agreement_share,
                    "best_margin_z": best_margin_z,
                    "best_top_strength": best_top_strength,
                    "allin_selected": bool(dyn.get("selected") != "primary"),
                }
            )
        else:
            dynamic_rows.append({"date": trade_date, "selected": None, "missing": True})

    model_df = pd.DataFrame(model_rows)
    dynamic_df = pd.DataFrame(dynamic_rows)
    summary_rows = []
    for name, group in model_df[model_df.get("missing") == False].groupby("model"):  # noqa: E712
        row = {"source": name, **_stats(group["return"].tolist())}
        row["hit_vs_fallback_rate"] = float(group["hit_vs_fallback"].mean())
        summary_rows.append(row)
    if not dynamic_df.empty:
        row = {"source": "dynamic_switch", **_stats(dynamic_df["return"].tolist())}
        if "hit_vs_fallback" in dynamic_df.columns:
            row["hit_vs_fallback_rate"] = float(dynamic_df["hit_vs_fallback"].mean())
        summary_rows.append(row)
        switched = dynamic_df[dynamic_df.get("allin_selected") == True]  # noqa: E712
        if not switched.empty:
            row = {"source": "dynamic_switch_allin_only", **_stats(switched["return"].tolist())}
            row["hit_vs_fallback_rate"] = float(switched["hit_vs_fallback"].mean())
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_return", "p05_return"], ascending=[False, False])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(output_dir / "model_top1_matrix.csv", index=False)
    dynamic_df.to_csv(output_dir / "dynamic_switch_matrix.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "dates": windows,
                "summary": summary_df.to_dict(orient="records"),
                "model_rows": model_df.to_dict(orient="records"),
                "dynamic_rows": dynamic_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_doc(Path(args.doc_path), windows, summary_df, model_df, dynamic_df)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
