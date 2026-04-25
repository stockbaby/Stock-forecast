from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import save_dataframe
from src.portfolio.construct import build_top_k_submission, select_best_portfolio_strategy
from src.training.metrics import precision_at_k, rank_ic, top_k_portfolio_return
from src.training.train_baseline import validate_submission
from src.training.validation import evaluate_recent_windows
from src.utils.config import load_yaml_config


def _load_prediction_frame(item: dict, label_name: str) -> pd.DataFrame:
    path = item["path"]
    score_name = item["name"]
    label_col = item.get("label_col", label_name)
    df = pd.read_csv(path, dtype={"stock_id": str})
    required_cols = {"stock_id", "date", "score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file {path} missing columns: {sorted(missing)}")
    if label_col not in df.columns:
        raise ValueError(f"Prediction file {path} missing label column: {label_col}")
    out = df[["stock_id", "date", label_col, "score"]].copy()
    out = out.rename(columns={label_col: f"{label_name}_{score_name}", "score": f"score_{score_name}"})
    out["date"] = pd.to_datetime(out["date"])
    return out


def _normalize_scores(df: pd.DataFrame, score_cols: list[str], method: str) -> pd.DataFrame:
    out = df.copy()
    for col in score_cols:
        grouped = out.groupby("date")[col]
        if method == "rank":
            out[f"{col}_norm"] = grouped.rank(pct=True)
        elif method == "zscore":
            mean = grouped.transform("mean")
            std = grouped.transform("std").replace(0, pd.NA)
            out[f"{col}_norm"] = (out[col] - mean) / std
            out[f"{col}_norm"] = out[f"{col}_norm"].fillna(0.0)
        else:
            raise ValueError(f"Unsupported normalize method: {method}")
    return out


def _weight_candidates(model_names: list[str], step: float = 0.1) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    combos: list[dict[str, float]] = []
    for values in itertools.product(range(units + 1), repeat=len(model_names)):
        if sum(values) != units:
            continue
        if max(values) == 0:
            continue
        combos.append({name: value / units for name, value in zip(model_names, values)})
    combos.sort(key=lambda item: tuple(item[name] for name in model_names), reverse=True)
    return combos


def _score_from_window_metrics(window_metrics: list[dict]) -> float:
    if not window_metrics:
        return float("-inf")
    score = 0.0
    total_weight = 0.0
    for item in window_metrics:
        weight = float(item["window_days"])
        total_weight += weight
        score += weight * (
            1.8 * float(item["strategy_mean_return"])
            + 0.3 * float(item["top_k_portfolio_return"])
            + 0.1 * float(item["rank_ic"])
            + 0.1 * float(item["precision_at_k"])
        )
    return score / max(total_weight, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend model predictions and export ensemble submissions.")
    parser.add_argument("--config", default="configs/ensemble_alpha.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    label_name = cfg["label"]["name"]
    normalize_method = cfg["ensemble"].get("normalize_method", "rank")
    top_k = int(cfg["training"]["top_k"])
    max_weight_sum = float(cfg["training"]["max_weight_sum"])
    temperature = float(cfg.get("portfolio", {}).get("temperature", 1.0))
    strategies = cfg.get("portfolio", {}).get(
        "strategies",
        ["proportional_positive_thr0.0", "equal_weight", "softmax_t0.6"],
    )
    recent_windows = cfg.get("validation", {}).get("recent_windows", [20, 40, 60, 90])
    num_candidate_submissions = int(cfg.get("output", {}).get("num_candidate_submissions", 3))

    frames = [_load_prediction_frame(item, label_name) for item in cfg["models"]]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["stock_id", "date"], how="inner")

    label_cols = [col for col in merged.columns if col.startswith(f"{label_name}_")]
    merged[label_name] = merged[label_cols].bfill(axis=1).iloc[:, 0]
    merged = merged.drop(columns=label_cols)

    model_names = [item["name"] for item in cfg["models"]]
    score_cols = [f"score_{name}" for name in model_names]
    merged = _normalize_scores(merged, score_cols, normalize_method)

    best: dict | None = None
    search_weights = cfg["ensemble"].get("weight_grid")
    if search_weights:
        weight_candidates = search_weights
    else:
        weight_candidates = _weight_candidates(model_names, step=float(cfg["ensemble"].get("grid_step", 0.1)))

    for weight_spec in weight_candidates:
        working = merged[["stock_id", "date", label_name, *[f"{col}_norm" for col in score_cols]]].copy()
        working["score"] = 0.0
        for name in model_names:
            working["score"] += float(weight_spec[name]) * working[f"score_{name}_norm"]

        best_strategy, strategy_results = select_best_portfolio_strategy(
            working,
            label_col=label_name,
            score_col="score",
            strategies=strategies,
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            temperature=temperature,
        )
        recent_metrics = evaluate_recent_windows(
            working,
            label_col=label_name,
            score_col="score",
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            strategy=best_strategy,
            temperature=temperature,
            windows=recent_windows,
        )
        score = _score_from_window_metrics(recent_metrics)
        candidate = {
            "weights": weight_spec,
            "strategy": best_strategy,
            "strategy_results": strategy_results,
            "recent_window_metrics": recent_metrics,
            "selection_score": score,
            "pred_df": working[["stock_id", "date", label_name, "score"]].copy(),
        }
        if best is None or score > best["selection_score"]:
            best = candidate

    if best is None:
        raise RuntimeError("No ensemble candidate was produced.")

    valid_pred_df = best["pred_df"].copy()
    latest_date = valid_pred_df["date"].max()
    latest_df = valid_pred_df[valid_pred_df["date"] == latest_date].copy()
    submission = build_top_k_submission(
        latest_df,
        score_col="score",
        stock_col="stock_id",
        top_k=top_k,
        max_weight_sum=max_weight_sum,
        strategy=best["strategy"],
        temperature=temperature,
    )
    validate_submission(submission, top_k, max_weight_sum)

    save_dataframe(valid_pred_df, cfg["output"]["prediction_path"])
    save_dataframe(submission, cfg["output"]["submission_path"])

    candidate_dir = Path(cfg["output"]["candidate_dir"])
    candidate_dir.mkdir(parents=True, exist_ok=True)
    all_candidates: list[dict] = []
    for weight_spec in weight_candidates:
        working = merged[["stock_id", "date", label_name, *[f"{col}_norm" for col in score_cols]]].copy()
        working["score"] = 0.0
        for name in model_names:
            working["score"] += float(weight_spec[name]) * working[f"score_{name}_norm"]
        best_strategy, _ = select_best_portfolio_strategy(
            working,
            label_col=label_name,
            score_col="score",
            strategies=strategies,
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            temperature=temperature,
        )
        recent_metrics = evaluate_recent_windows(
            working,
            label_col=label_name,
            score_col="score",
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            strategy=best_strategy,
            temperature=temperature,
            windows=recent_windows,
        )
        all_candidates.append(
            {
                "weights": weight_spec,
                "strategy": best_strategy,
                "selection_score": _score_from_window_metrics(recent_metrics),
                "pred_df": working,
            }
        )
    all_candidates.sort(key=lambda item: item["selection_score"], reverse=True)
    for rank, item in enumerate(all_candidates[:num_candidate_submissions], start=1):
        latest_df = item["pred_df"][item["pred_df"]["date"] == item["pred_df"]["date"].max()].copy()
        candidate_submission = build_top_k_submission(
            latest_df,
            score_col="score",
            stock_col="stock_id",
            top_k=top_k,
            max_weight_sum=max_weight_sum,
            strategy=item["strategy"],
            temperature=temperature,
        )
        save_dataframe(candidate_submission, candidate_dir / f"candidate_{rank}.csv")

    metrics = {
        "model_name": "ensemble",
        "normalize_method": normalize_method,
        "weights": best["weights"],
        "selected_portfolio_strategy": best["strategy"],
        "selection_score": best["selection_score"],
        "n_valid_rows": int(len(valid_pred_df)),
        "rank_ic": rank_ic(valid_pred_df, label_name, "score"),
        "precision_at_k": precision_at_k(valid_pred_df, label_name, "score", top_k),
        "top_k_portfolio_return": top_k_portfolio_return(valid_pred_df, label_name, "score", top_k),
        "recent_window_metrics": best["recent_window_metrics"],
        "candidate_submissions": [
            f"candidate_{idx}.csv" for idx in range(1, min(num_candidate_submissions, len(all_candidates)) + 1)
        ],
    }

    Path(cfg["output"]["metrics_path"]).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
