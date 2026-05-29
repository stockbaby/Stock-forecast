from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.construct import build_top_k_submission
from src.training.train_baseline import validate_submission
from src.utils.config import load_yaml_config


def _normalize_stock_id(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _load_prediction(path: str | Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    if "label" in df.columns and label_col not in df.columns:
        df = df.rename(columns={"label": label_col})
    if "score" not in df.columns and "score_mean" in df.columns:
        df = df.rename(columns={"score_mean": "score"})
    missing = {"stock_id", "date", "score"}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df.copy()
    out["stock_id"] = out["stock_id"].map(_normalize_stock_id)
    out["date"] = pd.to_datetime(out["date"])
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    for col in [col for col in out.columns if col.startswith("score_seed_")]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if label_col in out.columns:
        out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    return out


def _normalize_scores(df: pd.DataFrame, score_col: str, transform: str) -> pd.Series:
    grouped = df.groupby("date")[score_col]
    if transform == "rank":
        return grouped.rank(pct=True)
    if transform == "zscore":
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        return ((df[score_col] - mean) / std).fillna(0.0)
    if transform == "raw":
        return df[score_col]
    raise ValueError(f"Unsupported score transform: {transform}")


def _blend_models(
    model_frames: dict[str, pd.DataFrame],
    model_weights: dict[str, float],
    label_col: str,
    transform: str,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    label_source: pd.Series | None = None
    seed_score_cols: list[str] = []
    for model_name, weight in model_weights.items():
        if model_name not in model_frames:
            raise KeyError(f"Candidate references unknown model: {model_name}")
        source = model_frames[model_name]
        model_seed_cols = [col for col in source.columns if col.startswith("score_seed_")]
        frame = source[["stock_id", "date", "score", *model_seed_cols, *([label_col] if label_col in source.columns else [])]].copy()
        frame = frame.rename(columns={"score": f"score_{model_name}"})
        renamed_seed_cols = {}
        for col in model_seed_cols:
            renamed_seed_cols[col] = f"{col}_{model_name}"
        frame = frame.rename(columns=renamed_seed_cols)
        seed_score_cols.extend(renamed_seed_cols.values())
        if label_col in frame.columns:
            label_source = frame[label_col]
            frame = frame.drop(columns=[label_col])
        merged = frame if merged is None else merged.merge(frame, on=["stock_id", "date"], how="inner")
    if merged is None:
        raise ValueError("Candidate must include at least one model weight.")
    merged["score"] = 0.0
    for model_name, weight in model_weights.items():
        score_col = f"score_{model_name}"
        merged[f"{score_col}_norm"] = _normalize_scores(merged, score_col, transform)
        merged["score"] += float(weight) * merged[f"{score_col}_norm"]
    if label_source is not None:
        # Reattach by index only for single-frame candidates; multi-frame labels are merged below if available.
        pass
    for frame in model_frames.values():
        if label_col in frame.columns:
            labels = frame[["stock_id", "date", label_col]].dropna(subset=[label_col]).copy()
            merged = merged.merge(labels, on=["stock_id", "date"], how="left")
            break
    return merged[["stock_id", "date", "score", *seed_score_cols, *([label_col] if label_col in merged.columns else [])]].copy()


def _score_profile(daily: pd.DataFrame, score_col: str) -> dict:
    ranked = daily.dropna(subset=[score_col]).sort_values(score_col, ascending=False).copy()
    if ranked.empty:
        return {
            "top_stock": None,
            "margin_z": -math.inf,
            "top_strength": -math.inf,
            "score_std": 0.0,
        }
    scores = ranked[score_col].to_numpy(dtype=float)
    score_std = float(np.std(scores))
    scale = score_std if score_std > 1e-8 else 1.0
    margin = float((scores[0] - scores[1]) / scale) if len(scores) > 1 else math.inf
    top_strength = float((scores[0] - np.mean(scores)) / scale)
    seed_cols = [col for col in ranked.columns if col.startswith("score_seed_")]
    seed_top_stocks = []
    for col in seed_cols:
        seed_ranked = ranked.dropna(subset=[col]).sort_values(col, ascending=False)
        if not seed_ranked.empty:
            seed_top_stocks.append(str(seed_ranked.iloc[0]["stock_id"]))
    vote_share = 0.0
    seed_vote_count = 0
    mean_top_in_seed_top1 = False
    if seed_top_stocks:
        vote_counts = Counter(seed_top_stocks)
        seed_vote_count = int(vote_counts.get(str(ranked.iloc[0]["stock_id"]), 0))
        vote_share = float(seed_vote_count / len(seed_top_stocks))
        mean_top_in_seed_top1 = seed_vote_count > 0
    return {
        "top_stock": str(ranked.iloc[0]["stock_id"]),
        "margin_z": margin,
        "top_strength": top_strength,
        "score_std": score_std,
        "seed_top1_vote_share": vote_share,
        "seed_top1_vote_count": seed_vote_count,
        "seed_top1_count": len(seed_top_stocks),
        "mean_top_in_seed_top1": mean_top_in_seed_top1,
    }


def _market_gate(daily: pd.DataFrame) -> dict:
    high_vol = float(pd.to_numeric(daily.get("regime_is_high_vol", pd.Series([0.0])), errors="coerce").fillna(0.0).mean())
    drawdown = float(pd.to_numeric(daily.get("regime_drawdown", pd.Series([0.0])), errors="coerce").fillna(0.0).mean())
    trend = float(pd.to_numeric(daily.get("regime_trend", pd.Series([0.0])), errors="coerce").fillna(0.0).mean())
    risk = float(np.clip(0.55 * high_vol + 0.35 * abs(min(drawdown, 0.0)) * 5.0 - 0.25 * max(trend, 0.0), 0.0, 2.0))
    return {"market_risk": risk, "market_trend": trend}


def _candidate_passes(profile: dict, market: dict, gate: dict) -> tuple[bool, float, list[str]]:
    reasons: list[str] = []
    if profile["margin_z"] < float(gate.get("min_margin_z", 0.9)):
        reasons.append("low_margin")
    if profile["top_strength"] < float(gate.get("min_top_strength", 1.0)):
        reasons.append("low_top_strength")
    if market["market_risk"] > float(gate.get("max_market_risk", 0.85)):
        reasons.append("market_risk")
    if profile.get("seed_top1_count", 0) > 0:
        if profile.get("seed_top1_vote_share", 0.0) < float(gate.get("min_seed_vote_share", 0.0)):
            reasons.append("low_seed_vote")
        if gate.get("require_mean_top_in_seed_top1", False) and not profile.get("mean_top_in_seed_top1", False):
            reasons.append("mean_top_not_seed_top1")
    gate_score = (
        float(profile["margin_z"])
        + 0.5 * float(profile["top_strength"])
        + float(gate.get("seed_vote_bonus", 0.5)) * float(profile.get("seed_top1_vote_share", 0.0))
        - float(gate.get("market_risk_penalty", 0.5)) * float(market["market_risk"])
    )
    return not reasons, gate_score, reasons


def _daily_return(daily: pd.DataFrame, submission: pd.DataFrame, label_col: str) -> float:
    if submission.empty or label_col not in daily.columns:
        return 0.0
    merged = submission.merge(daily[["stock_id", label_col]], on="stock_id", how="left")
    return float((merged["weight"] * merged[label_col]).sum())


def _choose_daily(
    date: pd.Timestamp,
    primary: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
    cfg: dict,
    labeled: bool,
) -> tuple[dict, pd.DataFrame]:
    label_col = cfg["label_col"]
    top_k = int(cfg.get("top_k", 5))
    max_weight_sum = float(cfg.get("max_weight_sum", 1.0))
    temperature = float(cfg.get("temperature", 0.8))
    primary_strategy = cfg.get("primary_strategy", "dynamic_risk_budget")
    switch_strategy = cfg.get("switch_strategy", "top1_weight")

    primary_daily = primary[primary["date"] == date].copy()
    primary_submission = build_top_k_submission(
        primary_daily,
        score_col="score",
        stock_col="stock_id",
        top_k=top_k,
        max_weight_sum=max_weight_sum,
        strategy=primary_strategy,
        temperature=temperature,
    )
    market = _market_gate(primary_daily)
    options = []
    for name, frame in candidates.items():
        daily = frame[frame["date"] == date].copy()
        profile = _score_profile(daily, "score")
        passed, gate_score, reasons = _candidate_passes(profile, market, cfg.get("gate", {}))
        options.append(
            {
                "name": name,
                "passed": passed,
                "gate_score": gate_score,
                "reasons": reasons,
                **profile,
            }
        )
    top_counts = Counter(item["top_stock"] for item in options if item["top_stock"])
    for item in options:
        agreement = top_counts.get(item["top_stock"], 0)
        item["candidate_top1_agreement"] = int(agreement)
        item["candidate_top1_agreement_share"] = float(agreement / max(len(options), 1))
        if agreement < int(cfg.get("gate", {}).get("min_candidate_top1_agreement", 1)):
            item["passed"] = False
            item["reasons"] = [*item["reasons"], "low_candidate_agreement"]
        if item["candidate_top1_agreement_share"] < float(cfg.get("gate", {}).get("min_candidate_top1_agreement_share", 0.0)):
            item["passed"] = False
            item["reasons"] = [*item["reasons"], "low_candidate_agreement_share"]
    passed_options = [item for item in options if item["passed"]]
    if passed_options:
        selected = max(passed_options, key=lambda item: item["gate_score"])
        selected_frame = candidates[selected["name"]]
        selected_daily = selected_frame[selected_frame["date"] == date].copy()
        submission = build_top_k_submission(
            selected_daily,
            score_col="score",
            stock_col="stock_id",
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            strategy=switch_strategy,
            temperature=temperature,
        )
        selected_name = selected["name"]
        selected_reason = "candidate_gate_pass"
        active_daily = selected_daily
    else:
        submission = primary_submission
        selected_name = "primary"
        selected_reason = "primary_default"
        active_daily = primary_daily
    validate_submission(submission, top_k, max_weight_sum)
    record = {
        "date": str(pd.Timestamp(date).date()),
        "selected": selected_name,
        "reason": selected_reason,
        "market": market,
        "candidates": options,
        "stocks": submission["stock_id"].tolist(),
        "weights": submission["weight"].round(10).tolist(),
    }
    if labeled:
        record["return"] = _daily_return(active_daily if selected_name != "primary" else primary_daily, submission, label_col)
    return record, submission


def _summary_stats(records: list[dict]) -> dict:
    returns = np.asarray([row["return"] for row in records if "return" in row], dtype=float)
    if len(returns) == 0:
        return {}
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    return {
        "num_days": int(len(returns)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "p05_return": float(np.quantile(returns, 0.05)),
        "negative_rate": float(np.mean(returns < 0.0)),
        "max_drawdown": float(np.min(equity / np.maximum(peak, 1e-12) - 1.0)),
        "switch_rate": float(np.mean([row["selected"] != "primary" for row in records])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic switch from MASTER primary to high-confidence challenger candidates.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    label_col = cfg["label_col"]
    output_dir = Path(cfg.get("output_dir", "outputs/dynamic_candidate_switch"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_frames = {
        name: _load_prediction(item["prediction_path"], label_col)
        for name, item in cfg["models"].items()
        if Path(item["prediction_path"]).exists()
    }
    primary = model_frames[cfg["primary_model"]]
    candidates = {
        item["name"]: _blend_models(model_frames, item["weights"], label_col, item.get("transform", "rank"))
        for item in cfg.get("candidates", [])
    }

    dates = sorted(pd.to_datetime(primary.dropna(subset=[label_col])["date"]).drop_duplicates().tolist())
    records = [
        _choose_daily(date, primary, candidates, cfg, labeled=True)[0]
        for date in dates
    ]

    latest_submission_path = None
    latest_record = None
    latest_model_frames = {}
    for name, item in cfg["models"].items():
        latest_path = item.get("latest_prediction_path")
        if latest_path and Path(latest_path).exists():
            latest_model_frames[name] = _load_prediction(latest_path, label_col)
    if latest_model_frames and cfg["primary_model"] in latest_model_frames:
        latest_primary = latest_model_frames[cfg["primary_model"]]
        latest_candidates = {
            item["name"]: _blend_models(latest_model_frames, item["weights"], label_col, item.get("transform", "rank"))
            for item in cfg.get("candidates", [])
            if all(model_name in latest_model_frames for model_name in item["weights"])
        }
        latest_date = pd.Timestamp(latest_primary["date"].max())
        latest_record, latest_submission = _choose_daily(latest_date, latest_primary, latest_candidates, cfg, labeled=False)
        latest_submission_path = output_dir / "latest_submission.csv"
        latest_submission.to_csv(latest_submission_path, index=False)

    summary = {
        "config": args.config,
        "primary_model": cfg["primary_model"],
        "history": _summary_stats(records),
        "records": records,
        "latest_record": latest_record,
        "latest_submission_path": str(latest_submission_path) if latest_submission_path else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {key: value for key, value in summary.items() if key != "records"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
