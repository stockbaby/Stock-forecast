from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import normalize_price_frame
from src.features.industry_context import load_industry_map


NUMERIC_FEATURES = [
    "same_top1_count",
    "independent_family_count",
    "top1_agreement_share",
    "same_industry_family_count",
    "seed_vote_share",
    "seed_unique_count",
    "margin_z",
    "top_strength",
    "candidate_ret_5",
    "candidate_ret_20",
    "candidate_amount_5_20",
    "candidate_volatility_20",
    "candidate_liquidity_rank",
    "industry_ret_5",
    "industry_ret_20",
    "market_ret_5",
    "market_ret_20",
    "market_volatility_20",
]
CAT_FEATURES = ["family", "industry_name"]

FAMILY_MAP = {
    "master": "master",
    "master_multiseed": "master",
    "stockmixer_lite": "stockmixer",
    "stockmixer_fast": "stockmixer",
    "stockmixer_official": "stockmixer",
    "stockmixer_lite_multiseed": "stockmixer",
    "stockmixer_official_multiseed": "stockmixer",
    "timexer": "timexer",
    "timexer_multiseed": "timexer",
    "baseline": "baseline",
    "lightgbm_local_baseline": "baseline",
}


def _norm_stock(value: object) -> str:
    text = str(value).split(".")[0].strip()
    return text.zfill(6) if text and text.lower() != "nan" else ""


def _stats(values: pd.Series) -> dict:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(arr) == 0:
        return {"n": 0, "mean_return": math.nan, "p05_return": math.nan, "negative_rate": math.nan}
    return {
        "n": int(len(arr)),
        "mean_return": float(arr.mean()),
        "p05_return": float(np.quantile(arr, 0.05)),
        "negative_rate": float((arr < 0.0).mean()),
    }


def _load_price(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["stock_id", "date", "close", "amount", "industry_name"])
    raw = pd.read_csv(path)
    price = normalize_price_frame(raw)
    price["stock_id"] = price["stock_id"].map(_norm_stock)
    price["date"] = pd.to_datetime(price["date"])
    return price


def _load_industry(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["stock_id", "industry_name"])
    try:
        out = load_industry_map(path)[["stock_id", "industry_name"]].copy()
        out["stock_id"] = out["stock_id"].map(_norm_stock)
        out["industry_name"] = out["industry_name"].fillna("other").astype(str)
        return out.drop_duplicates("stock_id")
    except Exception:
        df = pd.read_csv(path, dtype={"stock_id": str})
        if "stock_id" not in df.columns:
            return pd.DataFrame(columns=["stock_id", "industry_name"])
        industry_col = "industry_name" if "industry_name" in df.columns else "industry" if "industry" in df.columns else None
        out = df[["stock_id"]].copy()
        out["stock_id"] = out["stock_id"].map(_norm_stock)
        out["industry_name"] = df[industry_col].astype(str) if industry_col else "other"
        return out.drop_duplicates("stock_id")


def _add_price_features(price: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    if price.empty:
        return pd.DataFrame(columns=["stock_id", "date"])
    out = price.merge(industry, on="stock_id", how="left")
    out["industry_name"] = out["industry_name"].fillna("other")
    out = out.sort_values(["stock_id", "date"]).copy()
    g = out.groupby("stock_id", group_keys=False)
    out["candidate_ret_5"] = g["close"].pct_change(5)
    out["candidate_ret_20"] = g["close"].pct_change(20)
    out["ret_1"] = g["close"].pct_change()
    out["candidate_volatility_20"] = g["ret_1"].rolling(20).std().reset_index(level=0, drop=True)
    if "amount" in out.columns:
        amount_5 = g["amount"].rolling(5).mean().reset_index(level=0, drop=True)
        amount_20 = g["amount"].rolling(20).mean().reset_index(level=0, drop=True)
        out["candidate_amount_5_20"] = amount_5 / amount_20.replace(0, np.nan)
        out["amount_20"] = amount_20
    else:
        out["candidate_amount_5_20"] = np.nan
        out["amount_20"] = np.nan

    daily = out.groupby("date")
    out["candidate_liquidity_rank"] = daily["amount_20"].rank(pct=True).fillna(0.5)
    out["market_ret_5"] = daily["candidate_ret_5"].transform("mean")
    out["market_ret_20"] = daily["candidate_ret_20"].transform("mean")
    out["market_volatility_20"] = daily["ret_1"].transform("std")
    out["industry_ret_5"] = out.groupby(["date", "industry_name"])["candidate_ret_5"].transform("mean")
    out["industry_ret_20"] = out.groupby(["date", "industry_name"])["candidate_ret_20"].transform("mean")
    keep = [
        "stock_id",
        "date",
        "industry_name",
        "candidate_ret_5",
        "candidate_ret_20",
        "candidate_amount_5_20",
        "candidate_volatility_20",
        "candidate_liquidity_rank",
        "market_ret_5",
        "market_ret_20",
        "market_volatility_20",
        "industry_ret_5",
        "industry_ret_20",
    ]
    return out[keep]


def _build_rows(model_matrix: Path, price_features: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(model_matrix, dtype={"top1": str})
    df = df[df.get("missing", False).astype(str).str.lower() != "true"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["top1"] = df["top1"].map(_norm_stock)
    df["family"] = df["model"].map(FAMILY_MAP).fillna(df["model"].astype(str))
    df["seed_vote_share"] = pd.to_numeric(df.get("seed_vote_share", np.nan), errors="coerce").fillna(0.0)
    df["seed_unique_count"] = pd.to_numeric(df.get("seed_unique_count", np.nan), errors="coerce").fillna(0.0)
    for col in ["margin_z", "top_strength", "return", "fallback_return"]:
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce")

    by_date_top1 = df.groupby(["date", "top1"])
    same_top1 = by_date_top1["model"].transform("count")
    family_count = by_date_top1["family"].transform("nunique")
    df["same_top1_count"] = same_top1
    df["independent_family_count"] = family_count
    df["top1_agreement_share"] = same_top1 / df.groupby("date")["model"].transform("count").replace(0, np.nan)

    out = df.merge(price_features, left_on=["top1", "date"], right_on=["stock_id", "date"], how="left")
    out["industry_name"] = out["industry_name"].fillna("other")
    ind_family = out.groupby(["date", "industry_name"])["family"].transform("nunique")
    out["same_industry_family_count"] = ind_family
    out["target_allin_beats_fallback"] = (out["return"] > out["fallback_return"]).astype(int)
    out["candidate_stock"] = out["top1"]
    return out


def _feature_matrix(train: pd.DataFrame, score: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    frames = [train]
    if score is not None:
        frames.append(score)
    both = pd.concat(frames, ignore_index=True)
    for col in NUMERIC_FEATURES:
        if col not in both.columns:
            both[col] = 0.0
        both[col] = pd.to_numeric(both[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in CAT_FEATURES:
        if col not in both.columns:
            both[col] = "unknown"
        both[col] = both[col].fillna("unknown").astype(str)
    x_all = pd.get_dummies(both[NUMERIC_FEATURES + CAT_FEATURES], columns=CAT_FEATURES)
    x_train = x_all.iloc[: len(train)].copy()
    x_score = x_all.iloc[len(train) :].copy() if score is not None else None
    return x_train, x_score


def _fit_predict_models(train: pd.DataFrame, score: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_train, x_score = _feature_matrix(train, score)
    y = train["target_allin_beats_fallback"].to_numpy(int)
    if len(np.unique(y)) < 2:
        base = float(y.mean()) if len(y) else 0.0
        return np.repeat(base, len(score)), np.repeat(base, len(score))
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.7, class_weight="balanced", max_iter=1000, random_state=42),
    )
    logistic.fit(x_train, y)
    log_prob = logistic.predict_proba(x_score)[:, 1]
    gbdt = GradientBoostingClassifier(n_estimators=30, max_depth=2, learning_rate=0.05, random_state=42)
    gbdt.fit(x_train, y)
    gbdt_prob = gbdt.predict_proba(x_score)[:, 1]
    return log_prob, gbdt_prob


def _walk_forward(df: pd.DataFrame, min_train_dates: int, threshold: float) -> pd.DataFrame:
    rows = []
    dates = sorted(df["date"].dropna().unique())
    for idx, date in enumerate(dates):
        score = df[df["date"] == date].copy()
        if idx < min_train_dates:
            prob = (
                0.35 * score["top1_agreement_share"].fillna(0.0)
                + 0.25 * (score["independent_family_count"].fillna(0.0) / 3.0).clip(0.0, 1.0)
                + 0.20 * (score["seed_vote_share"].fillna(0.0)).clip(0.0, 1.0)
                + 0.20 * (score["top_strength"].fillna(0.0) / 8.0).clip(0.0, 1.0)
            ).to_numpy(float)
            gbdt_prob = prob.copy()
            note = "warmup_rule"
        else:
            train = df[df["date"] < date].copy()
            prob, gbdt_prob = _fit_predict_models(train, score)
            note = "walk_forward"
        for i, (_, rec) in enumerate(score.iterrows()):
            aux_prob = float((prob[i] + gbdt_prob[i]) / 2.0)
            rows.append(
                {
                    "date": str(pd.Timestamp(rec["date"]).date()),
                    "model": rec["model"],
                    "family": rec["family"],
                    "candidate_stock": rec["candidate_stock"],
                    "target_allin_beats_fallback": int(rec["target_allin_beats_fallback"]),
                    "actual_return": float(rec["return"]),
                    "fallback_return": float(rec["fallback_return"]),
                    "logistic_prob": float(prob[i]),
                    "gbdt_prob": float(gbdt_prob[i]),
                    "aux_prob_allin_beats_fallback": aux_prob,
                    "aux_supports_allin": bool(aux_prob >= threshold),
                    "note": note,
                }
            )
    return pd.DataFrame(rows)


def _score_latest(train: pd.DataFrame, latest_matrix: Path, latest_price_features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if not latest_matrix.exists():
        return pd.DataFrame()
    latest = _build_rows(latest_matrix, latest_price_features)
    log_prob, gbdt_prob = _fit_predict_models(train, latest)
    out = latest[
        [
            "date",
            "model",
            "family",
            "candidate_stock",
            "industry_name",
            "same_top1_count",
            "independent_family_count",
            "same_industry_family_count",
            "seed_vote_share",
            "margin_z",
            "top_strength",
            "candidate_ret_5",
            "candidate_amount_5_20",
        ]
    ].copy()
    out["logistic_prob"] = log_prob
    out["gbdt_prob"] = gbdt_prob
    out["aux_prob_allin_beats_fallback"] = (out["logistic_prob"] + out["gbdt_prob"]) / 2.0
    out["aux_supports_allin"] = out["aux_prob_allin_beats_fallback"] >= threshold
    out["date"] = out["date"].dt.date.astype(str)
    return out.sort_values("aux_prob_allin_beats_fallback", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auxiliary walk-forward selector for all-in support scoring.")
    parser.add_argument("--model-matrix", default="outputs/recent_holdout_matrix_strict_full_summary/model_top1_matrix.csv")
    parser.add_argument("--price-path", default="data/raw/stock_data.csv")
    parser.add_argument("--industry-map", default="data/raw/hs300_stock_list.csv")
    parser.add_argument("--latest-model-matrix", default="outputs/latestA_20260529_summary/model_top1_matrix.csv")
    parser.add_argument("--latest-price-path", default="data/raw_latestA_20260529/stock_data.csv")
    parser.add_argument("--output-dir", default="outputs/auxiliary_allin_selector")
    parser.add_argument("--min-train-dates", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    industry = _load_industry(Path(args.industry_map))
    price_features = _add_price_features(_load_price(Path(args.price_path)), industry)
    train = _build_rows(Path(args.model_matrix), price_features)
    walk = _walk_forward(train, min_train_dates=args.min_train_dates, threshold=args.threshold)

    latest_price_features = _add_price_features(_load_price(Path(args.latest_price_path)), industry)
    latest = _score_latest(train, Path(args.latest_model_matrix), latest_price_features, threshold=args.threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "training_frame.csv", index=False)
    walk.to_csv(output_dir / "walk_forward_predictions.csv", index=False)
    latest.to_csv(output_dir / "latest_auxiliary_scores.csv", index=False)

    summary = {
        "purpose": "auxiliary_only_do_not_drive_submission",
        "n_training_rows": int(len(train)),
        "n_dates": int(train["date"].nunique()),
        "features": NUMERIC_FEATURES + CAT_FEATURES,
        "threshold": args.threshold,
        "walk_forward_accuracy": float((walk["aux_supports_allin"] == walk["target_allin_beats_fallback"].astype(bool)).mean()),
        "walk_forward_selected_rate": float(walk["aux_supports_allin"].mean()),
        "selected_allin_stats": _stats(walk.loc[walk["aux_supports_allin"], "actual_return"]),
        "rejected_fallback_stats": _stats(walk.loc[~walk["aux_supports_allin"], "fallback_return"]),
        "latest_top_scores": latest.head(10).to_dict(orient="records") if not latest.empty else [],
        "warning": "Only five recent windows are available; use as support/opposition signal, not as final decision authority.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
