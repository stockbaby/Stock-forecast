from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.io import save_dataframe
from src.models.baseline import fit_baseline_model, predict_scores
from src.portfolio.construct import build_top_k_submission, select_best_portfolio_strategy
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
    train_end: str | None = None
    valid_start: str | None = None
    valid_end: str | None = None
    valid_days: int | None = None
    portfolio_strategies: list[str] | None = None
    portfolio_temperature: float = 1.0
    inference_date: str | None = None
    latest_prediction_path: str | None = None


def _default_time_split(
    df: pd.DataFrame,
    train_end: str | None = None,
    valid_start: str | None = None,
    valid_end: str | None = None,
    valid_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df["date"].dropna().unique())
    if len(unique_dates) < 20:
        raise ValueError("Not enough dates to split train/validation. Need at least 20 trading days.")

    if train_end or valid_start or valid_end:
        train_mask = pd.Series(True, index=df.index)
        valid_mask = pd.Series(True, index=df.index)
        if train_end:
            train_mask &= df["date"] <= pd.to_datetime(train_end)
        if valid_start:
            valid_mask &= df["date"] >= pd.to_datetime(valid_start)
        if valid_end:
            valid_mask &= df["date"] <= pd.to_datetime(valid_end)
        train_df = df[train_mask]
        valid_df = df[valid_mask]
        if len(train_df) > 0 and len(valid_df) > 0:
            return train_df, valid_df

    if valid_days is not None and valid_days > 0:
        split_idx = max(len(unique_dates) - valid_days, 1)
        cutoff = unique_dates[split_idx]
        train_df = df[df["date"] < cutoff]
        valid_df = df[df["date"] >= cutoff]
        if len(train_df) > 0 and len(valid_df) > 0:
            return train_df, valid_df

    cutoff = unique_dates[int(len(unique_dates) * 0.8)]
    train_df = df[df["date"] < cutoff]
    valid_df = df[df["date"] >= cutoff]
    return train_df, valid_df


def run_training(config: TrainConfig) -> dict:
    df = pd.read_csv(config.processed_path, dtype={"stock_id": str})
    df["date"] = pd.to_datetime(df["date"])
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    feature_columns = [
        col
        for col in df.columns
        if col not in {"date", "stock_id", config.label_name}
        and not col.startswith("Unnamed:")
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    model_df = df[df[config.label_name].notna()]
    train_df, valid_df = _default_time_split(
        model_df,
        train_end=config.train_end,
        valid_start=config.valid_start,
        valid_end=config.valid_end,
        valid_days=config.valid_days,
    )

    fitted = fit_baseline_model(train_df, feature_columns, config.label_name, model_type=config.model_type)
    valid_df = valid_df.copy()
    valid_df["score"] = predict_scores(fitted, valid_df)

    infer_df: pd.DataFrame | None = None
    if config.inference_date:
        inference_date = pd.Timestamp(config.inference_date)
        infer_df = df[df["date"] == inference_date].copy()
        if not infer_df.empty:
            infer_df["score"] = predict_scores(fitted, infer_df)

    metrics = {
        "model_name": fitted.name,
        "n_train_rows": int(len(train_df)),
        "n_valid_rows": int(len(valid_df)),
        "n_features": int(len(feature_columns)),
        "rank_ic": rank_ic(valid_df, config.label_name, "score"),
        "precision_at_k": precision_at_k(valid_df, config.label_name, "score", config.top_k),
        "top_k_portfolio_return": top_k_portfolio_return(valid_df, config.label_name, "score", config.top_k),
    }

    strategies = config.portfolio_strategies or [
        "proportional_positive",
        "equal_weight",
        "softmax",
        "positive_only",
        "positive_softmax",
        "top3_equal",
    ]
    best_strategy, strategy_results = select_best_portfolio_strategy(
        valid_df,
        label_col=config.label_name,
        score_col="score",
        strategies=strategies,
        top_k=config.top_k,
        max_weight_sum=config.max_weight_sum,
        temperature=config.portfolio_temperature,
    )
    metrics["selected_portfolio_strategy"] = best_strategy
    metrics["portfolio_strategy_results"] = strategy_results

    save_dataframe(valid_df, config.prediction_path)

    if infer_df is not None and not infer_df.empty:
        if config.latest_prediction_path:
            save_dataframe(infer_df[["stock_id", "date", "score"]], config.latest_prediction_path)
        latest_df = infer_df.copy()
        metrics["submission_source"] = "latest_inference"
        metrics["submission_date"] = str(pd.Timestamp(latest_df["date"].max()).date())
    else:
        if config.inference_date:
            raise ValueError(f"No latest inference rows found for configured T={config.inference_date}.")
        latest_date = valid_df["date"].max()
        latest_df = valid_df[valid_df["date"] == latest_date].copy()
        metrics["submission_source"] = "validation_fallback"
        metrics["submission_date"] = str(pd.Timestamp(latest_date).date())
    if config.inference_date:
        assert_prediction_date(latest_df, config.inference_date, "submission source prediction")
    submission = build_top_k_submission(
        latest_df,
        score_col="score",
        stock_col="stock_id",
        top_k=config.top_k,
        max_weight_sum=config.max_weight_sum,
        strategy=best_strategy,
        temperature=config.portfolio_temperature,
    )
    save_dataframe(submission, config.submission_path)
    validate_submission(submission, config.top_k, config.max_weight_sum)

    metrics_path = Path(config.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def validate_submission(submission: pd.DataFrame, top_k: int, max_weight_sum: float) -> None:
    required = ["stock_id", "weight"]
    missing = [col for col in required if col not in submission.columns]
    if missing:
        raise ValueError(f"Submission missing required columns: {missing}")
    if submission.empty:
        raise ValueError("Submission is empty.")
    if len(submission) > top_k:
        raise ValueError(f"Submission contains {len(submission)} rows, exceeds top_k={top_k}.")
    if submission["stock_id"].astype(str).nunique() != len(submission):
        raise ValueError("Submission contains duplicated stock_id.")
    normalized_codes = submission["stock_id"].astype(str).str.strip()
    if not normalized_codes.str.fullmatch(r"\d{6}").all():
        raise ValueError("Submission stock_id must be 6-digit numeric codes.")
    weight_sum = float(submission["weight"].sum())
    if weight_sum > max_weight_sum + 1e-8:
        raise ValueError(f"Submission weight sum {weight_sum:.6f} exceeds limit {max_weight_sum}.")
    if (submission["weight"] < 0).any():
        raise ValueError("Submission contains negative weights.")


def assert_prediction_date(frame: pd.DataFrame, expected_date: str | pd.Timestamp, context: str = "prediction") -> None:
    if "date" not in frame.columns:
        raise ValueError(f"{context} missing date column; cannot verify T-date alignment.")
    if frame.empty:
        raise ValueError(f"{context} is empty; cannot verify T-date alignment.")
    expected = pd.Timestamp(expected_date).normalize()
    dates = pd.to_datetime(frame["date"]).dropna().dt.normalize().drop_duplicates()
    if len(dates) != 1:
        values = [str(pd.Timestamp(value).date()) for value in sorted(dates.tolist())]
        raise ValueError(f"{context} must contain exactly one prediction date, got {values}.")
    actual = pd.Timestamp(dates.iloc[0]).normalize()
    if actual != expected:
        raise ValueError(
            f"{context} date mismatch: latest_date={actual.date()} expected_unlabeled_T={expected.date()}."
        )
