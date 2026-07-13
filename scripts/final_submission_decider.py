from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.industry_context import load_industry_map
from src.portfolio.construct import build_top_k_submission
from src.training.train_baseline import validate_submission


MODEL_SPECS = [
    {
        "name": "master",
        "family": "master",
        "path": "outputs/holdout_20260517/master_alpha_official_rank_latest_predictions.csv",
        "submission": "outputs/holdout_20260517/master_alpha_official_rank_submission.csv",
    },
    {
        "name": "stockmixer_official",
        "family": "stockmixer",
        "path": "outputs/holdout_20260517/stockmixer_alpha_official_rank_latest_predictions.csv",
        "submission": "outputs/holdout_20260517/stockmixer_alpha_official_rank_submission.csv",
    },
    {
        "name": "stockmixer_official_multiseed",
        "family": "stockmixer",
        "path": "outputs/holdout_20260517/multiseed_top1/stockmixer_official/stockmixer_official_multiseed_latest_predictions.csv",
    },
    {
        "name": "timexer",
        "family": "timexer",
        "path": "outputs/holdout_20260517/timexer_alpha_fast_latest_predictions.csv",
        "submission": "outputs/holdout_20260517/timexer_alpha_fast_submission.csv",
    },
    {
        "name": "lightgbm_local_baseline",
        "family": "baseline",
        "path": "outputs/holdout_20260517/baseline_latest_predictions.csv",
        "submission": "outputs/holdout_20260517/baseline_submission.csv",
    },
]

CORE_MODELS = ["master", "stockmixer_official", "timexer"]


def _norm_stock(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _load_prediction(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    if "score" not in df.columns and "score_mean" in df.columns:
        df = df.rename(columns={"score_mean": "score"})
    if not {"stock_id", "date", "score"}.issubset(df.columns):
        raise ValueError(f"{path} must contain stock_id,date,score or score_mean")
    df = df.copy()
    df["stock_id"] = df["stock_id"].map(_norm_stock)
    df["date"] = pd.to_datetime(df["date"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    for col in [col for col in df.columns if col.startswith("score_seed_")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    latest_date = df["date"].max()
    return df[df["date"] == latest_date].dropna(subset=["score"]).copy()


def _score_profile(df: pd.DataFrame) -> dict:
    ranked = df.sort_values("score", ascending=False).reset_index(drop=True)
    if ranked.empty:
        return {
            "top1": "",
            "top5": [],
            "margin_z": math.nan,
            "top_strength": math.nan,
            "seed_vote_share": math.nan,
            "seed_unique_count": 0,
            "seed_top1": [],
        }
    scores = ranked["score"].to_numpy(dtype=float)
    std = float(np.std(scores))
    scale = std if std > 1e-8 else 1.0
    seed_cols = [col for col in ranked.columns if col.startswith("score_seed_")]
    seed_top1: list[str] = []
    for col in seed_cols:
        seed_ranked = ranked.dropna(subset=[col]).sort_values(col, ascending=False)
        if not seed_ranked.empty:
            seed_top1.append(str(seed_ranked.iloc[0]["stock_id"]))
    top1 = str(ranked.iloc[0]["stock_id"])
    vote_share = math.nan
    if seed_top1:
        vote_share = Counter(seed_top1).get(top1, 0) / len(seed_top1)
    return {
        "top1": top1,
        "top5": ranked.head(5)["stock_id"].astype(str).tolist(),
        "margin_z": float((scores[0] - scores[1]) / scale) if len(scores) > 1 else math.inf,
        "top_strength": float((scores[0] - np.mean(scores)) / scale),
        "seed_vote_share": vote_share,
        "seed_unique_count": len(set(seed_top1)),
        "seed_top1": seed_top1,
    }


def _load_official_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    if path.name.endswith(".csv"):
        maybe_submission = pd.read_csv(path, dtype={"stock_id": str})
        if {"stock_id", "weight"}.issubset(maybe_submission.columns):
            top5 = maybe_submission["stock_id"].map(_norm_stock).tolist()
            return {
                "date": "",
                "top1": top5[0] if top5 else "",
                "top5": top5,
                "returns": "",
            }
    df = pd.read_csv(path, dtype={"official_top1": str})
    if df.empty:
        return None
    row = df.iloc[-1]
    top5 = [_norm_stock(item) for item in str(row["official_top5"]).split(";") if item]
    return {
        "date": str(row["date"]),
        "top1": _norm_stock(row["official_top1"]),
        "top5": top5,
        "returns": str(row.get("official_top5_returns", "")),
    }


def _load_submission_top(path: str | Path) -> tuple[list[str], list[float]]:
    p = Path(path)
    if not p.exists():
        return [], []
    df = pd.read_csv(p, dtype={"stock_id": str})
    if not {"stock_id", "weight"}.issubset(df.columns):
        return [], []
    df = df.copy()
    df["stock_id"] = df["stock_id"].map(_norm_stock)
    return df["stock_id"].tolist(), pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).tolist()


def _load_dynamic(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rec = data.get("latest_record")
    if not rec:
        rows = data.get("dynamic_rows") or []
        rec = rows[-1] if rows else None
    if not rec:
        return None
    stocks = rec.get("stocks") or rec.get("top5") or []
    top1 = rec.get("top1")
    if top1 and not stocks:
        stocks = [top1]
    return {
        "date": rec.get("date", ""),
        "selected": rec.get("selected", ""),
        "top1": stocks[0] if stocks else "",
        "top5": stocks,
        "weights": rec.get("weights", []),
        "market_risk": rec.get("market", {}).get("market_risk", math.nan),
        "candidate_details": rec.get("candidates", []),
    }


def _load_best_relation(path: Path) -> dict | None:
    if not path.exists():
        return None
    stocks, weights = _load_submission_top(path)
    if stocks:
        return {
            "method": path.stem,
            "submission_path": str(path),
            "top1": stocks[0],
            "top5": stocks,
            "weights": weights,
            "last_holdout_return": math.nan,
        }
    scores = pd.read_csv(path)
    if scores.empty:
        return None
    row = scores.sort_values("portfolio_return", ascending=False).iloc[0]
    stocks, weights = _load_submission_top(row["submission_path"])
    return {
        "method": row["method"],
        "submission_path": row["submission_path"],
        "top1": stocks[0] if stocks else "",
        "top5": stocks,
        "weights": weights,
        "last_holdout_return": float(row["portfolio_return"]),
    }


def _load_price_frame(raw_path: Path, tail_path: Path) -> pd.DataFrame:
    frames = []
    if raw_path.exists():
        raw = pd.read_csv(raw_path, usecols=["股票代码", "日期", "收盘", "成交额"], dtype={"股票代码": str})
        raw = raw.rename(columns={"股票代码": "stock_id", "日期": "date", "收盘": "close", "成交额": "amount"})
        frames.append(raw)
    if tail_path.exists():
        tail = pd.read_csv(tail_path, usecols=["stock_id", "date", "close", "amount"], dtype={"stock_id": str})
        frames.append(tail)
    if not frames:
        return pd.DataFrame(columns=["stock_id", "date", "close", "amount"])
    out = pd.concat(frames, ignore_index=True)
    out["stock_id"] = out["stock_id"].map(_norm_stock)
    out["date"] = pd.to_datetime(out["date"])
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.dropna(subset=["date", "close", "amount"])
    return out.drop_duplicates(subset=["stock_id", "date"], keep="last")


def _volume_confirmation(price: pd.DataFrame, stock: str, signal_date: pd.Timestamp) -> str:
    if price.empty or not stock:
        return "unknown"
    hist = price[(price["stock_id"] == stock) & (price["date"] <= signal_date)].sort_values("date")
    if len(hist) < 6:
        return "unknown"
    amount5 = float(hist.tail(5)["amount"].mean())
    amount20 = float(hist.tail(20)["amount"].mean()) if len(hist) >= 20 else float(hist["amount"].mean())
    close_now = float(hist.iloc[-1]["close"])
    close_5 = float(hist.iloc[-6]["close"])
    ret5 = close_now / close_5 - 1.0 if close_5 else 0.0
    ratio = amount5 / amount20 if amount20 else 0.0
    tag = "strong" if ratio >= 1.15 and ret5 > 0 else "mixed" if ratio >= 0.8 else "weak"
    return f"{tag}; amount5/20={ratio:.2f}; ret5={ret5:.2%}"


def _build_diagnostics(
    model_frames: dict[str, pd.DataFrame],
    model_specs: list[dict],
    official: dict | None,
    dynamic: dict | None,
    relation: dict | None,
    industry: pd.DataFrame,
    price: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    industry_map = industry.set_index("stock_id")["industry_name"].to_dict() if not industry.empty else {}
    profiles: dict[str, dict] = {}
    rows = []
    model_top5 = {}
    for spec in model_specs:
        name = spec["name"]
        frame = model_frames.get(name)
        if frame is None:
            continue
        profile = _score_profile(frame)
        profiles[name] = {**profile, "family": spec["family"]}
        model_top5[name] = set(profile["top5"])
    top1_counts = Counter(p["top1"] for p in profiles.values() if p["top1"])
    all_top5 = defaultdict(int)
    for top5 in model_top5.values():
        for stock in top5:
            all_top5[stock] += 1
    latest_date = next(iter(model_frames.values()))["date"].max() if model_frames else pd.Timestamp.today()
    for spec in model_specs:
        name = spec["name"]
        if name not in profiles:
            continue
        p = profiles[name]
        top1 = p["top1"]
        rows.append(
            {
                "source": name,
                "family": spec["family"],
                "top1": top1,
                "top1_industry": industry_map.get(top1, "other"),
                "top5": ";".join(p["top5"]),
                "margin_z": round(p["margin_z"], 4),
                "top_strength": round(p["top_strength"], 4),
                "seed_consistency": "na"
                if math.isnan(p["seed_vote_share"])
                else f"{p['seed_vote_share']:.2f} ({p['seed_unique_count']} unique)",
                "same_top1_models": top1_counts.get(top1, 0),
                "top5_overlap_models": sum(all_top5[s] for s in p["top5"]),
                "volume_price_confirmation": _volume_confirmation(price, top1, latest_date),
            }
        )
    if official:
        top1 = official["top1"]
        rows.append(
            {
                "source": "official_baseline_top5",
                "family": "official_baseline",
                "top1": top1,
                "top1_industry": industry_map.get(top1, "other"),
                "top5": ";".join(official["top5"]),
                "margin_z": "na",
                "top_strength": "defensive",
                "seed_consistency": "na",
                "same_top1_models": top1_counts.get(top1, 0),
                "top5_overlap_models": sum(all_top5[s] for s in official["top5"]),
                "volume_price_confirmation": _volume_confirmation(price, top1, latest_date),
            }
        )
    if relation:
        top1 = relation["top1"]
        rows.append(
            {
                "source": "relation_postprocess_best",
                "family": "relation",
                "top1": top1,
                "top1_industry": industry_map.get(top1, "other"),
                "top5": ";".join(relation["top5"]),
                "margin_z": "na",
                "top_strength": f"last_holdout={relation['last_holdout_return']:.2%}",
                "seed_consistency": "na",
                "same_top1_models": top1_counts.get(top1, 0),
                "top5_overlap_models": sum(all_top5[s] for s in relation["top5"]),
                "volume_price_confirmation": _volume_confirmation(price, top1, latest_date),
            }
        )
    if dynamic:
        top1 = _norm_stock(dynamic["top1"])
        rows.append(
            {
                "source": "dynamic_switch",
                "family": "switch",
                "top1": top1,
                "top1_industry": industry_map.get(top1, "other"),
                "top5": ";".join(_norm_stock(s) for s in dynamic["top5"]),
                "margin_z": "na",
                "top_strength": f"selected={dynamic['selected']}",
                "seed_consistency": "na",
                "same_top1_models": top1_counts.get(top1, 0),
                "top5_overlap_models": sum(all_top5[_norm_stock(s)] for s in dynamic["top5"]),
                "volume_price_confirmation": _volume_confirmation(price, top1, latest_date),
            }
        )
    return pd.DataFrame(rows), profiles


def _independent_agreement(profiles: dict[str, dict], industry_map: dict[str, str]) -> dict:
    stock_families: defaultdict[str, set[str]] = defaultdict(set)
    industry_families: defaultdict[str, set[str]] = defaultdict(set)
    for name in CORE_MODELS:
        p = profiles.get(name)
        if not p:
            continue
        top1 = p["top1"]
        family = p["family"]
        stock_families[top1].add(family)
        industry_families[industry_map.get(top1, "other")].add(family)
    best_stock, best_stock_fams = max(stock_families.items(), key=lambda item: len(item[1]), default=("", set()))
    best_industry, best_industry_fams = max(industry_families.items(), key=lambda item: len(item[1]), default=("", set()))
    return {
        "best_stock": best_stock,
        "best_stock_family_count": len(best_stock_fams),
        "best_industry": best_industry,
        "best_industry_family_count": len(best_industry_fams),
    }


def _submission_from_stocks(stocks: list[str], weights: list[float]) -> pd.DataFrame:
    if len(weights) != len(stocks):
        weights = [1.0 / max(len(stocks), 1)] * len(stocks)
    return pd.DataFrame({"stock_id": [_norm_stock(s) for s in stocks], "weight": weights})


def _make_softmax_submission(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    return build_top_k_submission(
        frame,
        score_col="score",
        stock_col="stock_id",
        top_k=top_k,
        max_weight_sum=1.0,
        strategy="softmax",
        temperature=0.8,
    )


def _decide(
    profiles: dict[str, dict],
    diagnostics: pd.DataFrame,
    official: dict | None,
    dynamic: dict | None,
    model_frames: dict[str, pd.DataFrame],
    industry: pd.DataFrame,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    industry_map = industry.set_index("stock_id")["industry_name"].to_dict() if not industry.empty else {}
    agreement = _independent_agreement(profiles, industry_map)
    official_top5 = set(official["top5"]) if official else set()
    master_top5 = set(profiles.get("master", {}).get("top5", []))
    stockmixer = profiles.get("stockmixer_official", {})
    timexer = profiles.get("timexer", {})
    master = profiles.get("master", {})
    top1s = [profiles[name]["top1"] for name in CORE_MODELS if name in profiles]
    top1_counter = Counter(top1s)
    consensus_stock = top1_counter.most_common(1)[0][0] if top1_counter else ""
    stock_consensus = agreement["best_stock_family_count"] >= 2
    industry_consensus = agreement["best_industry_family_count"] >= 2 and agreement["best_industry"] != "other"
    candidate = consensus_stock if stock_consensus else stockmixer.get("top1", "")
    candidate_in_official = candidate in official_top5
    official_completely_different = bool(official_top5) and candidate not in official_top5 and len(official_top5 & master_top5) == 0
    fallback_ok = bool(master_top5 & official_top5) or candidate_in_official or not official_completely_different
    top_strength = max(
        float(stockmixer.get("top_strength", 0.0) or 0.0),
        float(timexer.get("top_strength", 0.0) or 0.0),
        float(master.get("top_strength", 0.0) or 0.0),
    )
    margin = max(
        float(stockmixer.get("margin_z", 0.0) or 0.0),
        float(timexer.get("margin_z", 0.0) or 0.0),
        float(master.get("margin_z", 0.0) or 0.0),
    )
    allin_allowed = (stock_consensus or industry_consensus) and fallback_ok and top_strength >= 3.0 and margin >= 0.10
    if official_completely_different:
        allin_allowed = False

    if allin_allowed and stock_consensus:
        decision = "all_in"
        selected_stock = consensus_stock
        weight_cap = 1.0
        reason = "跨模型同股共识成立，且未被官方 baseline/fallback 否决。"
    elif allin_allowed and industry_consensus:
        decision = "top1_capped_60"
        selected_stock = candidate
        weight_cap = 0.6
        reason = "跨模型同行业共识成立但不是同股，只给 60% 上限。"
    elif stock_consensus or industry_consensus:
        decision = "top3_defensive"
        selected_stock = candidate
        weight_cap = 0.6
        reason = "存在共识，但 fallback/官方防守信号不够配合，降为 Top3 分散。"
    else:
        decision = "top5_defensive"
        selected_stock = master.get("top1", "")
        weight_cap = 0.4
        reason = "Top1 分歧大，无真正跨模型共识，禁止单股 all-in。"

    submissions: dict[str, pd.DataFrame] = {}
    if selected_stock:
        submissions["allin_candidate"] = _submission_from_stocks([selected_stock], [1.0])
        submissions["capped_top1_candidate"] = _submission_from_stocks(
            [selected_stock, *[s for s in master.get("top5", []) if s != selected_stock][:4]],
            [weight_cap, *([round((1.0 - weight_cap) / 4.0, 10)] * 4)],
        )
    if "master" in model_frames:
        submissions["master_top3"] = _make_softmax_submission(model_frames["master"], 3)
        submissions["master_top5"] = _make_softmax_submission(model_frames["master"], 5)
    for name in ["master_multiseed", "stockmixer_official_multiseed", "timexer_multiseed"]:
        if name in model_frames:
            submissions[f"{name}_top3"] = _make_softmax_submission(model_frames[name], 3)
            submissions[f"{name}_top5"] = _make_softmax_submission(model_frames[name], 5)
    if official:
        submissions["official_baseline_top5_equal"] = _submission_from_stocks(official["top5"], [0.2] * len(official["top5"]))
    if dynamic and dynamic.get("top5"):
        submissions["dynamic_switch"] = _submission_from_stocks(dynamic["top5"], dynamic.get("weights", []))
    for name, sub in list(submissions.items()):
        if sub.empty:
            submissions.pop(name)
            continue
        validate_submission(sub, top_k=min(5, len(sub)), max_weight_sum=1.0)

    selected_submission = "allin_candidate" if decision == "all_in" else "capped_top1_candidate" if decision == "top1_capped_60" else "master_top3" if decision == "top3_defensive" else "master_top5"
    if selected_submission not in submissions and submissions:
        selected_submission = next(iter(submissions))
    selected_submission_top1 = ""
    if selected_submission in submissions and not submissions[selected_submission].empty:
        selected_submission_top1 = _norm_stock(submissions[selected_submission].iloc[0]["stock_id"])
    summary = {
        "decision": decision,
        "selected_submission": selected_submission,
        "selected_stock": selected_submission_top1 or selected_stock,
        "gated_candidate_stock": selected_stock,
        "reason": reason,
        "allin_allowed": allin_allowed,
        "agreement": agreement,
        "official_completely_different": official_completely_different,
        "candidate_in_official_top5": candidate_in_official,
        "fallback_ok": fallback_ok,
        "max_core_top_strength": top_strength,
        "max_core_margin_z": margin,
    }
    return summary, submissions


def _write_report(path: Path, summary: dict, diagnostics: pd.DataFrame, holdout_summary: pd.DataFrame | None) -> None:
    lines = [
        "# Final submission decider",
        "",
        "## Decision",
        "",
        f"- decision: `{summary['decision']}`",
        f"- selected_submission: `{summary['selected_submission']}`",
        f"- selected_stock: `{summary.get('selected_stock', '')}`",
        f"- gated_candidate_stock: `{summary.get('gated_candidate_stock', '')}`",
        f"- allin_allowed: `{summary['allin_allowed']}`",
        f"- reason: {summary['reason']}",
        "",
        "## Gate facts",
        "",
        f"- independent stock agreement families: `{summary['agreement']['best_stock_family_count']}` on `{summary['agreement']['best_stock']}`",
        f"- independent industry agreement families: `{summary['agreement']['best_industry_family_count']}` on `{summary['agreement']['best_industry']}`",
        f"- official completely different: `{summary['official_completely_different']}`",
        f"- candidate in official Top5: `{summary['candidate_in_official_top5']}`",
        f"- fallback ok: `{summary['fallback_ok']}`",
        f"- max core top_strength / margin_z: `{summary['max_core_top_strength']:.4f}` / `{summary['max_core_margin_z']:.4f}`",
        "",
        "## Candidate diagnostic table",
        "",
        diagnostics.to_markdown(index=False),
    ]
    if holdout_summary is not None and not holdout_summary.empty:
        lines.extend(["", "## Five-window reference", "", holdout_summary.to_markdown(index=False)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final submission candidate diagnostics and gated decision.")
    parser.add_argument("--output-dir", default="outputs/final_submission_decider")
    parser.add_argument("--official-baseline", default="outputs/official_baseline_holdout/official_baseline_holdout_matrix.csv")
    parser.add_argument("--dynamic-summary", default="outputs/holdout_20260517/dynamic_candidate_switch/summary.json")
    parser.add_argument("--relation-scores", default="outputs/holdout_20260517/relation/test_scores.csv")
    parser.add_argument("--industry-map", default="data/raw/hs300_stock_list.csv")
    parser.add_argument("--raw-price", default="data/raw/stock_data.csv")
    parser.add_argument("--tail-price", default="data/interim/akshare_tail_partial_20260522.csv")
    parser.add_argument("--holdout-summary", default="outputs/official_baseline_holdout/combined_summary_with_official.csv")
    parser.add_argument("--latest-strict-dir", default="")
    parser.add_argument("--latest-date-key", default="20260529")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = MODEL_SPECS
    if args.latest_strict_dir:
        root = Path(args.latest_strict_dir) / args.latest_date_key
        model_specs = [
            {
                "name": "master",
                "family": "master",
                "path": str(root / "master/master_seed_42_latest_predictions.csv"),
                "submission": str(root / "master/master_seed_42_submission.csv"),
            },
            {
                "name": "master_multiseed",
                "family": "master",
                "path": str(root / "master/master_multiseed_latest_predictions.csv"),
            },
            {
                "name": "stockmixer_official",
                "family": "stockmixer",
                "path": str(root / "stockmixer_official/stockmixer_official_seed_42_latest_predictions.csv"),
                "submission": str(root / "stockmixer_official/stockmixer_official_seed_42_submission.csv"),
            },
            {
                "name": "stockmixer_official_multiseed",
                "family": "stockmixer",
                "path": str(root / "stockmixer_official/stockmixer_official_multiseed_latest_predictions.csv"),
            },
            {
                "name": "timexer",
                "family": "timexer",
                "path": str(root / "timexer/timexer_seed_42_latest_predictions.csv"),
                "submission": str(root / "timexer/timexer_seed_42_submission.csv"),
            },
            {
                "name": "timexer_multiseed",
                "family": "timexer",
                "path": str(root / "timexer/timexer_multiseed_latest_predictions.csv"),
            },
            {
                "name": "lightgbm_local_baseline",
                "family": "baseline",
                "path": str(root / "baseline/baseline_seed_42_latest_predictions.csv"),
                "submission": str(root / "baseline/baseline_seed_42_submission.csv"),
            },
        ]

    model_frames = {}
    available_specs = []
    for spec in model_specs:
        path = Path(spec["path"])
        if path.exists():
            model_frames[spec["name"]] = _load_prediction(path)
            available_specs.append(spec)

    official = _load_official_baseline(Path(args.official_baseline))
    dynamic = _load_dynamic(Path(args.dynamic_summary))
    relation = _load_best_relation(Path(args.relation_scores))
    industry = load_industry_map(args.industry_map)
    price = _load_price_frame(Path(args.raw_price), Path(args.tail_price))
    diagnostics, profiles = _build_diagnostics(
        model_frames,
        available_specs,
        official,
        dynamic,
        relation,
        industry,
        price,
    )
    summary, submissions = _decide(profiles, diagnostics, official, dynamic, model_frames, industry)

    diagnostics.to_csv(output_dir / "candidate_diagnostic_table.csv", index=False, encoding="utf-8-sig")
    for name, sub in submissions.items():
        sub.to_csv(output_dir / f"{name}.csv", index=False)
    summary["submission_paths"] = {name: str(output_dir / f"{name}.csv") for name in submissions}
    (output_dir / "decision_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    holdout_summary = pd.read_csv(args.holdout_summary) if Path(args.holdout_summary).exists() else None
    _write_report(output_dir / "decision_report.md", summary, diagnostics, holdout_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
