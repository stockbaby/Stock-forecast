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


FEATURES = [
    "candidate_top1_agreement",
    "candidate_top1_agreement_share",
    "best_margin_z",
    "best_top_strength",
    "fallback_return_trailing_mean",
    "switch_return_trailing_mean",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, lr: float, epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    z = (x - mean) / std
    weights = np.zeros(z.shape[1], dtype=float)
    bias = 0.0
    for _ in range(epochs):
        pred = _sigmoid(z @ weights + bias)
        err = pred - y
        weights -= lr * ((z.T @ err) / len(y) + l2 * weights)
        bias -= lr * float(err.mean())
    return weights, mean, std


def _predict_logistic(x: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return _sigmoid(((x - mean) / std) @ weights)


def _load_training_frame(dynamic_path: Path) -> pd.DataFrame:
    dynamic = pd.read_csv(dynamic_path)
    dynamic["date"] = pd.to_datetime(dynamic["date"])
    dynamic = dynamic.sort_values("date").reset_index(drop=True)
    dynamic["target_allin_beats_fallback"] = (dynamic["return"] > dynamic["fallback_return"]).astype(int)
    dynamic["fallback_return_trailing_mean"] = dynamic["fallback_return"].shift(1).rolling(5, min_periods=1).mean().fillna(0.0)
    dynamic["switch_return_trailing_mean"] = dynamic["return"].shift(1).rolling(5, min_periods=1).mean().fillna(0.0)
    if "best_margin_z" not in dynamic.columns:
        dynamic["best_margin_z"] = 0.0
    if "best_top_strength" not in dynamic.columns:
        dynamic["best_top_strength"] = 0.0
    for col in FEATURES:
        if col not in dynamic.columns:
            dynamic[col] = 0.0
        dynamic[col] = pd.to_numeric(dynamic[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return dynamic


def _stats(returns: list[float]) -> dict:
    arr = np.asarray([value for value in returns if pd.notna(value)], dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean_return": math.nan, "p05_return": math.nan, "negative_rate": math.nan, "max_drawdown": math.nan}
    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    return {
        "n": int(len(arr)),
        "mean_return": float(arr.mean()),
        "p05_return": float(np.quantile(arr, 0.05)),
        "negative_rate": float((arr < 0.0).mean()),
        "max_drawdown": float(np.min(equity / np.maximum(peak, 1e-12) - 1.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward logistic selector for Top1 all-in versus fallback.")
    parser.add_argument("--dynamic-matrix", default="outputs/recent_holdout_matrix/dynamic_switch_matrix.csv")
    parser.add_argument("--output-dir", default="outputs/recent_holdout_matrix/gate_selector")
    parser.add_argument("--min-train-days", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--l2", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=600)
    args = parser.parse_args()

    df = _load_training_frame(Path(args.dynamic_matrix))
    predictions = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        if idx < args.min_train_days:
            prob = float(row.get("candidate_top1_agreement_share", 0.0))
            selected_allin = bool(prob >= args.threshold)
            model_note = "warmup_rule"
        else:
            train = df.iloc[:idx]
            x_train = train[FEATURES].to_numpy(dtype=float)
            y_train = train["target_allin_beats_fallback"].to_numpy(dtype=float)
            weights, mean, std = _fit_logistic(x_train, y_train, l2=args.l2, lr=args.lr, epochs=args.epochs)
            prob = float(_predict_logistic(row[FEATURES].to_numpy(dtype=float).reshape(1, -1), weights, mean, std)[0])
            selected_allin = bool(prob >= args.threshold)
            model_note = "walk_forward_logistic"
        chosen_return = float(row["return"]) if selected_allin else float(row["fallback_return"])
        predictions.append(
            {
                "date": str(pd.Timestamp(row["date"]).date()),
                "prob_allin_beats_fallback": prob,
                "selected_allin": selected_allin,
                "actual_allin_beats_fallback": bool(row["target_allin_beats_fallback"]),
                "chosen_return": chosen_return,
                "allin_return": float(row["return"]),
                "fallback_return": float(row["fallback_return"]),
                "model_note": model_note,
            }
        )

    pred_df = pd.DataFrame(predictions)
    summary = {
        "features": FEATURES,
        "threshold": args.threshold,
        "selector": _stats(pred_df["chosen_return"].tolist()),
        "always_allin": _stats(pred_df["allin_return"].tolist()),
        "always_fallback": _stats(pred_df["fallback_return"].tolist()),
        "selection_rate": float(pred_df["selected_allin"].mean()) if not pred_df.empty else math.nan,
        "classification_accuracy": float(
            (pred_df["selected_allin"] == pred_df["actual_allin_beats_fallback"]).mean()
        )
        if not pred_df.empty
        else math.nan,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_dir / "walk_forward_predictions.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
