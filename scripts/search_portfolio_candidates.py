from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.construct import build_top_k_submission, evaluate_portfolio_strategy
from src.training.metrics import precision_at_k, rank_ic, top_k_portfolio_return
from src.training.train_baseline import validate_submission


DEFAULT_MODELS = [
    ("master_official", "outputs/predictions/master_alpha_official_rank_predictions.csv"),
    ("stockmixer_official", "outputs/predictions/stockmixer_alpha_official_rank_predictions.csv"),
    ("itransformer", "outputs/predictions/itransformer_alpha_predictions.csv"),
    ("ensemble_alpha", "outputs/predictions/ensemble_alpha_predictions.csv"),
    ("stockmixer_fast", "outputs/predictions/stockmixer_alpha_fast_predictions.csv"),
    ("stockmixer_industry_fast", "outputs/predictions/stockmixer_alpha_industry_fast_predictions.csv"),
]


def _normalize_stock_id(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def _load_prediction(name: str, path: str, label_col: str) -> pd.DataFrame | None:
    pred_path = Path(path)
    if not pred_path.exists():
        return None
    df = pd.read_csv(pred_path, dtype={"stock_id": str})
    if "label" in df.columns and label_col not in df.columns:
        df = df.rename(columns={"label": label_col})
    required = {"stock_id", "date", "score", label_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df[["stock_id", "date", label_col, "score"]].copy()
    out["stock_id"] = out["stock_id"].map(_normalize_stock_id)
    out["date"] = pd.to_datetime(out["date"])
    out = out.rename(columns={"score": f"score_{name}", label_col: f"label_{name}"})
    return out


def _score_transform(df: pd.DataFrame, score_col: str, transform: str) -> pd.Series:
    grouped = df.groupby("date")[score_col]
    if transform == "raw":
        return df[score_col]
    if transform == "rank":
        return grouped.rank(pct=True)
    if transform == "zscore":
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        return ((df[score_col] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raise ValueError(f"Unsupported transform: {transform}")


def _weight_grid(names: list[str], step: float, max_models: int) -> list[dict[str, float]]:
    combos: list[dict[str, float]] = []
    units = int(round(1.0 / step))
    for subset_size in range(1, min(max_models, len(names)) + 1):
        for subset in itertools.combinations(names, subset_size):
            for values in itertools.product(range(units + 1), repeat=subset_size):
                if sum(values) != units or max(values) == 0:
                    continue
                spec = {name: 0.0 for name in names}
                for name, value in zip(subset, values):
                    spec[name] = value / units
                combos.append(spec)
    unique = {tuple(sorted(item.items())): item for item in combos}
    return list(unique.values())


def _make_submission(
    df: pd.DataFrame,
    score_col: str,
    strategy: str,
    top_k: int,
    max_weight_sum: float,
    temperature: float,
    industry_map_path: str | None,
    max_per_industry: int | None,
) -> pd.DataFrame:
    latest = df[df["date"] == df["date"].max()].copy()
    submission = build_top_k_submission(
        latest,
        score_col=score_col,
        stock_col="stock_id",
        top_k=top_k,
        max_weight_sum=max_weight_sum,
        strategy=strategy,
        temperature=temperature,
        industry_map_path=industry_map_path,
        max_per_industry=max_per_industry,
    )
    validate_submission(submission, top_k, max_weight_sum)
    return submission


def _evaluate_candidate(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    strategy: str,
    top_k: int,
    max_weight_sum: float,
    temperature: float,
    industry_map_path: str | None,
    max_per_industry: int | None,
) -> dict:
    portfolio_eval = evaluate_portfolio_strategy(
        df,
        label_col=label_col,
        score_col=score_col,
        strategy=strategy,
        top_k=top_k,
        max_weight_sum=max_weight_sum,
        temperature=temperature,
        industry_map_path=industry_map_path,
        max_per_industry=max_per_industry,
    )
    return {
        "mean_return": float(portfolio_eval["mean_return"]),
        "std_return": float(portfolio_eval.get("std_return", np.nan)),
        "num_days": int(portfolio_eval["num_days"]),
        "rank_ic": rank_ic(df, label_col, score_col),
        "precision_at_k": precision_at_k(df, label_col, score_col, top_k),
        "top_k_portfolio_return": top_k_portfolio_return(df, label_col, score_col, top_k),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search post-processing and ensemble portfolio candidates.")
    parser.add_argument("--label-col", default="y_ret_a_stage_round1_open_open")
    parser.add_argument("--output-dir", default="outputs/portfolio_search")
    parser.add_argument("--industry-map-path", default="data/raw/hs300_stock_list.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-weight-sum", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--max-ensemble-models", type=int, default=4)
    parser.add_argument("--models", nargs="*", help="Optional subset of model names to search.")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name, path in DEFAULT_MODELS:
        if args.models and name not in args.models:
            continue
        frame = _load_prediction(name, path, args.label_col)
        if frame is not None:
            frames.append((name, frame))
    if not frames:
        raise RuntimeError("No prediction files found.")

    merged = frames[0][1]
    for _, frame in frames[1:]:
        merged = merged.merge(frame, on=["stock_id", "date"], how="inner")
    label_cols = [col for col in merged.columns if col.startswith("label_")]
    merged[args.label_col] = merged[label_cols].bfill(axis=1).iloc[:, 0]
    merged = merged.drop(columns=label_cols)

    model_names = [name for name, _ in frames]
    transforms = ["rank", "zscore"]
    strategies = [
        "proportional_positive_thr0.0",
        "equal_weight",
        "softmax_t0.6",
        "positive_only_thr0.0",
    ]
    industry_caps: list[int | None] = [None, 2, 3]
    rows: list[dict] = []

    for transform in transforms:
        working = merged[["stock_id", "date", args.label_col]].copy()
        for name in model_names:
            working[f"score_{name}_{transform}"] = _score_transform(merged, f"score_{name}", transform)

        single_specs = [
            {"kind": "single", "weights": {name: 1.0}, "score_name": name}
            for name in model_names
        ]
        ensemble_specs = [
            {"kind": "ensemble", "weights": weights, "score_name": "blend"}
            for weights in _weight_grid(model_names, args.grid_step, args.max_ensemble_models)
            if sum(weight > 0 for weight in weights.values()) >= 2
        ]
        for spec in single_specs + ensemble_specs:
            candidate = working[["stock_id", "date", args.label_col]].copy()
            candidate["score"] = 0.0
            for name, weight in spec["weights"].items():
                if weight:
                    candidate["score"] += float(weight) * working[f"score_{name}_{transform}"]

            for strategy in strategies:
                for cap in industry_caps:
                    try:
                        metrics = _evaluate_candidate(
                            candidate,
                            label_col=args.label_col,
                            score_col="score",
                            strategy=strategy,
                            top_k=args.top_k,
                            max_weight_sum=args.max_weight_sum,
                            temperature=args.temperature,
                            industry_map_path=args.industry_map_path,
                            max_per_industry=cap,
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "status": "error",
                                "error": str(exc),
                                "transform": transform,
                                "strategy": strategy,
                                "max_per_industry": cap,
                                "kind": spec["kind"],
                                "weights": spec["weights"],
                            }
                        )
                        continue
                    rows.append(
                        {
                            "status": "ok",
                            "transform": transform,
                            "strategy": strategy,
                            "max_per_industry": cap,
                            "kind": spec["kind"],
                            "weights": spec["weights"],
                            **metrics,
                        }
                    )

    ok_rows = [row for row in rows if row["status"] == "ok"]
    ok_rows.sort(key=lambda item: item["mean_return"], reverse=True)
    (output_dir / "search_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "leaderboard.json").write_text(
        json.dumps(ok_rows[: args.top_n], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for rank, row in enumerate(ok_rows[: min(args.top_n, 10)], start=1):
        transform = row["transform"]
        candidate = merged[["stock_id", "date", args.label_col]].copy()
        candidate["score"] = 0.0
        for name, weight in row["weights"].items():
            if weight:
                candidate["score"] += float(weight) * _score_transform(merged, f"score_{name}", transform)
        submission = _make_submission(
            candidate,
            score_col="score",
            strategy=row["strategy"],
            top_k=args.top_k,
            max_weight_sum=args.max_weight_sum,
            temperature=args.temperature,
            industry_map_path=args.industry_map_path,
            max_per_industry=row["max_per_industry"],
        )
        submission.to_csv(output_dir / f"candidate_{rank}.csv", index=False)

    print(json.dumps(ok_rows[: args.top_n], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
