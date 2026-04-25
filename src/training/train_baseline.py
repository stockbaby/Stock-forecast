from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.io import save_dataframe
from src.models.baseline import fit_baseline_model, predict_scores
from src.portfolio.construct import build_top_k_submission
from src.training.metrics import precision_at_k, rank_ic, top_k_portfolio_return


@dataclass
class TrainConfig:
    processed_path: str
    label_name: str
    metrics_path: str
    prediction_path: str
    submission_path: str
    model_type: str
    top_k: int
    max_weight_sum: float


def _default_time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df["date"].dropna().unique())
    if len(unique_dates) < 20:
        raise ValueError("Not enough dates to split train/validation. Need at least 20 trading days.")

    cutoff = unique_dates[int(len(unique_dates) * 0.8)]
    train_df = df[df["date"] < cutoff].copy()
    valid_df = df[df["date"] >= cutoff].copy()
    return train_df, valid_df


def run_training(config: TrainConfig) -> dict:
    df = pd.read_csv(config.processed_path)
    df["date"] = pd.to_datetime(df["date"])

    feature_columns = [
        col
        for col in df.columns
        if col not in {"date", "stock_id", config.label_name}
        and not col.startswith("Unnamed:")
    ]
    model_df = df.dropna(subset=[config.label_name]).copy()
    train_df, valid_df = _default_time_split(model_df)

    fitted = fit_baseline_model(train_df, feature_columns, config.label_name, model_type=config.model_type)
    valid_df = valid_df.copy()
    valid_df["score"] = predict_scores(fitted, valid_df)

    metrics = {
        "model_name": fitted.name,
        "n_train_rows": int(len(train_df)),
        "n_valid_rows": int(len(valid_df)),
        "n_features": int(len(feature_columns)),
        "rank_ic": rank_ic(valid_df, config.label_name, "score"),
        "precision_at_k": precision_at_k(valid_df, config.label_name, "score", config.top_k),
        "top_k_portfolio_return": top_k_portfolio_return(valid_df, config.label_name, "score", config.top_k),
    }

    save_dataframe(valid_df, config.prediction_path)

    latest_date = valid_df["date"].max()
    latest_df = valid_df[valid_df["date"] == latest_date].copy()
    submission = build_top_k_submission(
        latest_df,
        score_col="score",
        stock_col="stock_id",
        top_k=config.top_k,
        max_weight_sum=config.max_weight_sum,
    )
    save_dataframe(submission, config.submission_path)

    metrics_path = Path(config.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
