from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.construct import build_top_k_submission, select_best_portfolio_strategy
from src.training.dataset_builder import DatasetBuildConfig, build_model_dataset
from src.training.metrics import precision_at_k, rank_ic, top_k_portfolio_return
from src.training.train_baseline import _default_time_split, save_dataframe, validate_submission
from src.models.deep_sequence import build_lstm_sequences, build_prediction_sequences, train_lstm_regressor
from src.utils.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LSTM baseline on alpha-style features.")
    parser.add_argument("--config", default="configs/lstm_alpha.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    build_cfg = DatasetBuildConfig(
        raw_dir=cfg["data"]["raw_dir"],
        processed_path=cfg["data"]["processed_path"],
        market_index_path=cfg["data"].get("market_index_path"),
        industry_map_path=cfg["data"].get("industry_map_path"),
        windows=cfg["features"]["lookback_windows"],
        label_name=cfg["label"]["name"],
        buy_offset=cfg["label"]["horizon_buy_offset"],
        sell_offset=cfg["label"]["horizon_sell_offset"],
        sell_fallback_offset=cfg["label"].get("horizon_sell_fallback_offset"),
    )
    df = build_model_dataset(build_cfg)
    df["date"] = pd.to_datetime(df["date"])
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    feature_columns = [
        col
        for col in df.columns
        if col not in {"date", "stock_id", cfg["label"]["name"]}
        and not col.startswith("Unnamed:")
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    model_df = df[df[cfg["label"]["name"]].notna()]
    train_df, valid_df = _default_time_split(model_df, valid_days=cfg["training"].get("valid_days"))
    recent_train_days = cfg["training"].get("recent_train_days")
    if recent_train_days:
        train_dates = sorted(train_df["date"].dropna().unique())
        if len(train_dates) > int(recent_train_days):
            train_df = train_df[train_df["date"] >= train_dates[-int(recent_train_days)]]

    dataset = build_lstm_sequences(
        train_df=train_df,
        valid_df=valid_df,
        feature_columns=feature_columns,
        label_column=cfg["label"]["name"],
        lookback=cfg["deep"]["lookback"],
    )
    model, valid_pred_df = train_lstm_regressor(
        dataset,
        hidden_size=cfg["deep"]["hidden_size"],
        num_layers=cfg["deep"]["num_layers"],
        dropout=cfg["deep"]["dropout"],
        batch_size=cfg["deep"]["batch_size"],
        epochs=cfg["deep"]["epochs"],
        learning_rate=cfg["deep"]["learning_rate"],
    )
    inference_date_value = (
        cfg.get("output", {}).get("inference_date")
        or cfg.get("data", {}).get("benchmark_end_date")
        or df["date"].max()
    )
    inference_date = pd.Timestamp(inference_date_value)
    infer_source_df = df[df["date"] <= inference_date].groupby("stock_id", group_keys=False).tail(cfg["deep"]["lookback"])
    x_infer, infer_meta = build_prediction_sequences(
        df=infer_source_df,
        feature_columns=feature_columns,
        lookback=cfg["deep"]["lookback"],
        target_dates=[inference_date],
    )
    infer_pred_df = None
    if len(x_infer) > 0:
        import torch

        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(x_infer).to(device)).detach().cpu().numpy()
        infer_pred_df = infer_meta.copy()
        infer_pred_df["score"] = preds

    metrics = {
        "model_name": "lstm",
        "n_train_sequences": int(len(dataset.x_train)),
        "n_valid_sequences": int(len(dataset.x_valid)),
        "n_inference_sequences": int(len(x_infer)),
        "inference_date": str(inference_date.date()),
        "n_features": int(len(feature_columns)),
        "rank_ic": rank_ic(
            valid_pred_df.rename(columns={"label": cfg["label"]["name"]}),
            cfg["label"]["name"],
            "score",
        ),
        "precision_at_k": precision_at_k(
            valid_pred_df.rename(columns={"label": cfg["label"]["name"]}),
            cfg["label"]["name"],
            "score",
            cfg["training"]["top_k"],
        ),
        "top_k_portfolio_return": top_k_portfolio_return(
            valid_pred_df.rename(columns={"label": cfg["label"]["name"]}),
            cfg["label"]["name"],
            "score",
            cfg["training"]["top_k"],
        ),
    }

    strategies = cfg.get("portfolio", {}).get(
        "strategies",
        ["proportional_positive", "equal_weight", "softmax", "positive_only"],
    )
    best_strategy, strategy_results = select_best_portfolio_strategy(
        valid_pred_df.rename(columns={"label": cfg["label"]["name"]}),
        label_col=cfg["label"]["name"],
        score_col="score",
        strategies=strategies,
        top_k=cfg["training"]["top_k"],
        max_weight_sum=cfg["training"]["max_weight_sum"],
        temperature=cfg.get("portfolio", {}).get("temperature", 1.0),
    )
    metrics["selected_portfolio_strategy"] = best_strategy
    metrics["portfolio_strategy_results"] = strategy_results

    save_dataframe(valid_pred_df, cfg["output"]["prediction_path"])

    latest_prediction_path = cfg["output"].get("latest_prediction_path")
    if infer_pred_df is not None and not infer_pred_df.empty:
        if latest_prediction_path:
            save_dataframe(infer_pred_df, latest_prediction_path)
        latest_df = infer_pred_df.copy()
        metrics["submission_source"] = "latest_inference"
        metrics["submission_date"] = str(pd.Timestamp(latest_df["date"].max()).date())
    else:
        latest_date = valid_pred_df["date"].max()
        latest_df = valid_pred_df[valid_pred_df["date"] == latest_date].copy()
        metrics["submission_source"] = "validation_fallback"
        metrics["submission_date"] = str(pd.Timestamp(latest_date).date())
    submission = build_top_k_submission(
        latest_df,
        score_col="score",
        stock_col="stock_id",
        top_k=cfg["training"]["top_k"],
        max_weight_sum=cfg["training"]["max_weight_sum"],
        strategy=best_strategy,
        temperature=cfg.get("portfolio", {}).get("temperature", 1.0),
    )
    save_dataframe(submission, cfg["output"]["submission_path"])
    validate_submission(submission, cfg["training"]["top_k"], cfg["training"]["max_weight_sum"])

    Path(cfg["output"]["metrics_path"]).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
