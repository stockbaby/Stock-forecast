from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.search_latest_ensemble_weights import (
    DEFAULT_MODELS,
    _date_returns,
    _eval_spec,
    _load_price,
    _score_daily,
    _build_daily_frames,
)
from src.portfolio.construct import build_top_k_submission
from src.training.train_baseline import validate_submission


def _norm_stock(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _stats(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean_return": math.nan, "p05_return": math.nan, "std_return": math.nan, "negative_rate": math.nan}
    return {
        "n": int(len(arr)),
        "mean_return": float(arr.mean()),
        "p05_return": float(np.quantile(arr, 0.05)),
        "std_return": float(arr.std()),
        "negative_rate": float((arr < 0.0).mean()),
        "robust_score": float(arr.mean() + 0.5 * np.quantile(arr, 0.05) - 0.25 * arr.std()),
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_diag(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"top1": str})
    if "top1" in df.columns:
        df["top1"] = df["top1"].map(_norm_stock)
    return df


def _gate_selector(dynamic_path: Path, latest_diag: pd.DataFrame, decision_summary: dict) -> tuple[pd.DataFrame, dict]:
    dynamic = pd.read_csv(dynamic_path) if dynamic_path.exists() else pd.DataFrame()
    rows = []
    if not dynamic.empty:
        for _, row in dynamic.iterrows():
            rows.append(
                {
                    "date": row.get("date"),
                    "target_allin_beats_fallback": bool(row.get("return", 0.0) > row.get("fallback_return", 0.0)),
                    "candidate_top1_agreement": float(row.get("candidate_top1_agreement", 0.0) or 0.0),
                    "candidate_top1_agreement_share": float(row.get("candidate_top1_agreement_share", 0.0) or 0.0),
                    "best_margin_z": float(row.get("best_margin_z", 0.0) or 0.0),
                    "best_top_strength": float(row.get("best_top_strength", 0.0) or 0.0),
                    "allin_selected_by_old_gate": bool(row.get("allin_selected", False)),
                    "allin_return": float(row.get("return", 0.0) or 0.0),
                    "fallback_return": float(row.get("fallback_return", 0.0) or 0.0),
                }
            )
    train = pd.DataFrame(rows)

    max_seed_vote = 0.0
    max_margin = 0.0
    max_strength = 0.0
    stock_agreement = float(decision_summary.get("agreement", {}).get("best_stock_family_count", 0) or 0)
    industry_agreement = float(decision_summary.get("agreement", {}).get("best_industry_family_count", 0) or 0)
    if not latest_diag.empty:
        max_margin = float(pd.to_numeric(latest_diag.get("margin_z"), errors="coerce").max(skipna=True) or 0.0)
        strengths = pd.to_numeric(latest_diag.get("top_strength"), errors="coerce")
        max_strength = float(strengths.max(skipna=True) if not strengths.dropna().empty else 0.0)
        seed_text = latest_diag.get("seed_consistency", pd.Series(dtype=str)).astype(str)
        votes = pd.to_numeric(seed_text.str.extract(r"^([0-9.]+)")[0], errors="coerce")
        max_seed_vote = float(votes.max(skipna=True) if not votes.dropna().empty else 0.0)

    official_conflict = bool(decision_summary.get("official_completely_different", True))
    fallback_ok = bool(decision_summary.get("fallback_ok", False))
    rule_score = 0.0
    rule_score += 0.25 * min(stock_agreement / 2.0, 1.0)
    rule_score += 0.15 * min(industry_agreement / 2.0, 1.0)
    rule_score += 0.20 * min(max_seed_vote, 1.0)
    rule_score += 0.15 * min(max_margin / 1.0, 1.0)
    rule_score += 0.15 * min(max_strength / 6.0, 1.0)
    rule_score += 0.10 if fallback_ok else -0.15
    rule_score -= 0.25 if official_conflict else 0.0
    allow = bool(rule_score >= 0.65 and stock_agreement >= 2 and fallback_ok and not official_conflict)

    latest = {
        "selector": "gate_selector",
        "support_score": float(max(0.0, min(1.0, rule_score))),
        "allow_allin": allow,
        "stock_agreement_families": stock_agreement,
        "industry_agreement_families": industry_agreement,
        "max_seed_vote": max_seed_vote,
        "max_margin_z": max_margin,
        "max_top_strength": max_strength,
        "official_conflict": official_conflict,
        "fallback_ok": fallback_ok,
        "reason": "allow only when same-stock cross-family agreement, fallback support, and no official conflict all hold",
    }
    return train, latest


def _submission_return(price: pd.DataFrame, sub: pd.DataFrame, buy_date: str, sell_date: str) -> float:
    labels = _date_returns(price, buy_date, sell_date)
    merged = sub.copy()
    merged["stock_id"] = merged["stock_id"].map(_norm_stock)
    merged = merged.merge(labels, on="stock_id", how="left")
    return float((merged["weight"] * merged["label"]).sum())


def _candidate_from_prediction(frame: pd.DataFrame, top_k: int, strategy: str = "softmax", temperature: float = 0.8) -> pd.DataFrame:
    sub = build_top_k_submission(
        frame,
        score_col="score",
        stock_col="stock_id",
        top_k=top_k,
        max_weight_sum=1.0,
        strategy=strategy,
        temperature=temperature,
    )
    validate_submission(sub, top_k=min(top_k, len(sub)), max_weight_sum=1.0)
    return sub


def _portfolio_selector(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    matrix = pd.read_csv(args.model_matrix, parse_dates=["date", "buy_date", "sell_date"])
    dates = [str(pd.Timestamp(d).date()) for d in sorted(matrix["date"].dropna().unique())]
    price = _load_price(Path(args.price_path))
    model_specs = {name: DEFAULT_MODELS[name] for name in ["master", "master_ms", "stockmixer_ms", "baseline"]}
    daily = _build_daily_frames(Path(args.holdout_root), dates, model_specs, price, matrix)

    official = pd.read_csv(args.official_matrix) if Path(args.official_matrix).exists() else pd.DataFrame()
    fusion_summary = _load_json(Path(args.fusion_summary))
    fusion_records = {rec["date"]: float(rec["return"]) for rec in fusion_summary.get("best_records", [])}
    fusion_top1 = {rec["date"]: rec.get("top1", "") for rec in fusion_summary.get("best_records", [])}

    rows = []
    for date, frame in daily.items():
        mrow = matrix[matrix["date"] == pd.to_datetime(date)].iloc[0]
        buy_date = str(pd.Timestamp(mrow["buy_date"]).date())
        sell_date = str(pd.Timestamp(mrow["sell_date"]).date())
        candidates = {}
        master_frame = frame[["stock_id", "label", "score_master"]].rename(columns={"score_master": "score"})
        candidates["master_top3"] = (_candidate_from_prediction(master_frame, 3), None)
        master_ms_frame = frame[["stock_id", "label", "score_master_ms"]].rename(columns={"score_master_ms": "score"})
        candidates["master_ms_top3"] = (_candidate_from_prediction(master_ms_frame, 3), None)
        if date in fusion_records:
            candidates["fusion_top2"] = (pd.DataFrame(), fusion_records[date])
        if not official.empty:
            orow = official[official["date"] == date]
            if not orow.empty:
                stocks = [_norm_stock(s) for s in str(orow.iloc[0]["official_top5"]).split(";")]
                candidates["official_top5"] = (pd.DataFrame({"stock_id": stocks, "weight": [1.0 / len(stocks)] * len(stocks)}), None)
        for name, (sub, known_return) in candidates.items():
            ret = float(known_return) if known_return is not None else _submission_return(price, sub, buy_date, sell_date)
            top1 = fusion_top1.get(date, "") if name == "fusion_top2" else (sub.iloc[0]["stock_id"] if not sub.empty else "")
            rows.append({"date": date, "candidate": name, "return": ret, "top1": _norm_stock(top1)})
    ret_df = pd.DataFrame(rows)
    leaderboard = []
    for name, group in ret_df.groupby("candidate"):
        leaderboard.append({"candidate": name, **_stats(group["return"].tolist())})
    leaderboard = sorted(leaderboard, key=lambda r: r.get("robust_score", -math.inf), reverse=True)

    latest_paths = {
        "master_top3": args.latest_master_top3,
        "master_ms_top3": args.latest_master_ms_top3,
        "fusion_top2": args.latest_fusion_top2,
        "official_top5": args.latest_official_top5,
    }
    selected = leaderboard[0]["candidate"] if leaderboard else "master_top3"
    latest_submission = latest_paths.get(selected, "")
    summary = {
        "selector": "portfolio_selector",
        "selected_candidate": selected,
        "latest_submission": latest_submission,
        "leaderboard": leaderboard,
        "note": "small candidate set; ranked by mean + 0.5*p05 - 0.25*std over five windows",
    }
    return ret_df, summary


def _weight_meta_learner(latest_diag: pd.DataFrame, gate: dict, portfolio: dict, fusion_summary: dict) -> dict:
    weights = {"master": 0.50, "stockmixer": 0.20, "timexer": 0.15, "baseline": 0.10, "official": 0.05}
    notes: list[str] = []
    if fusion_summary.get("best", {}).get("weights"):
        fw = fusion_summary["best"]["weights"]
        weights = {
            "master": float(fw.get("master", 0.0) + fw.get("master_ms", 0.0)),
            "stockmixer": float(fw.get("stockmixer", 0.0) + fw.get("stockmixer_ms", 0.0)),
            "timexer": float(fw.get("timexer", 0.0) + fw.get("timexer_ms", 0.0)),
            "baseline": float(fw.get("baseline", 0.0)),
            "official": 0.0,
        }
        notes.append("initialized from robust fusion search")

    if gate.get("official_conflict"):
        weights["official"] += 0.10
        weights["stockmixer"] *= 0.85
        weights["timexer"] *= 0.90
        notes.append("official baseline conflicts with alpha candidates: reduce concentration and add defense")
    if gate.get("max_seed_vote", 0.0) < 0.67:
        weights["stockmixer"] *= 0.85
        weights["timexer"] *= 0.90
        weights["master"] += 0.05
        notes.append("weak seed agreement outside MASTER: tilt back to MASTER")
    if gate.get("industry_agreement_families", 0.0) >= 2:
        weights["stockmixer"] += 0.05
        weights["timexer"] += 0.03
        notes.append("StockMixer/TimeXer industry agreement: keep some industry-alpha exposure")

    if not latest_diag.empty:
        baseline = latest_diag[latest_diag["source"].astype(str).str.contains("baseline", case=False, na=False)]
        if not baseline.empty and str(baseline.iloc[0].get("volume_price_confirmation", "")).startswith("strong"):
            weights["baseline"] += 0.05
            notes.append("baseline candidate has strong volume/price confirmation")

    total = sum(max(v, 0.0) for v in weights.values())
    weights = {k: float(max(v, 0.0) / total) for k, v in weights.items()}
    max_single_stock_weight = 0.60 if not gate.get("allow_allin") else 1.0
    if portfolio.get("selected_candidate") == "fusion_top2":
        max_single_stock_weight = min(max_single_stock_weight, 0.60)
    return {
        "selector": "weight_meta_learner",
        "recommended_model_family_weights": weights,
        "max_single_stock_weight": max_single_stock_weight,
        "concentration_mode": "capped_top2_or_top3" if max_single_stock_weight < 1.0 else "allin_allowed",
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run three auxiliary selectors: gate, portfolio, and weight meta learner.")
    parser.add_argument("--output-dir", default="outputs/three_selectors_latestA_20260529")
    parser.add_argument("--dynamic-matrix", default="outputs/recent_holdout_matrix_strict_full_summary/dynamic_switch_matrix.csv")
    parser.add_argument("--latest-diagnostic", default="outputs/final_submission_decider_latestA_20260529/candidate_diagnostic_table.csv")
    parser.add_argument("--decision-summary", default="outputs/final_submission_decider_latestA_20260529/decision_summary.json")
    parser.add_argument("--holdout-root", default="outputs/recent_holdout_matrix_strict_full")
    parser.add_argument("--model-matrix", default="outputs/recent_holdout_matrix_strict_full_summary/model_top1_matrix.csv")
    parser.add_argument("--price-path", default="data/raw/stock_data.csv")
    parser.add_argument("--official-matrix", default="outputs/official_baseline_holdout/official_baseline_holdout_matrix.csv")
    parser.add_argument("--fusion-summary", default="outputs/latestA_ensemble_weights_20260529_wide/summary.json")
    parser.add_argument("--latest-master-top3", default="outputs/final_submission_decider_latestA_20260529/master_top3.csv")
    parser.add_argument("--latest-master-ms-top3", default="outputs/final_submission_decider_latestA_20260529/master_multiseed_top3.csv")
    parser.add_argument("--latest-fusion-top2", default="outputs/latestA_ensemble_weights_20260529_wide/candidate_1.csv")
    parser.add_argument("--latest-official-top5", default="outputs/final_submission_decider_latestA_20260529/official_baseline_top5_equal.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_diag = _read_diag(Path(args.latest_diagnostic))
    decision_summary = _load_json(Path(args.decision_summary))
    fusion_summary = _load_json(Path(args.fusion_summary))

    gate_train, gate_latest = _gate_selector(Path(args.dynamic_matrix), latest_diag, decision_summary)
    portfolio_returns, portfolio_summary = _portfolio_selector(args)
    weight_summary = _weight_meta_learner(latest_diag, gate_latest, portfolio_summary, fusion_summary)

    gate_train.to_csv(output_dir / "gate_training_windows.csv", index=False)
    pd.DataFrame([gate_latest]).to_csv(output_dir / "gate_latest_score.csv", index=False)
    portfolio_returns.to_csv(output_dir / "portfolio_candidate_returns.csv", index=False)
    pd.DataFrame(portfolio_summary["leaderboard"]).to_csv(output_dir / "portfolio_selector_leaderboard.csv", index=False)
    (output_dir / "portfolio_selector_summary.json").write_text(json.dumps(portfolio_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "weight_meta_learner.json").write_text(json.dumps(weight_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "gate_selector": gate_latest,
        "portfolio_selector": portfolio_summary,
        "weight_meta_learner": weight_summary,
        "warning": "All three selectors are auxiliary because only five recent windows are available.",
    }
    (output_dir / "three_selector_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
