from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deep_sequence import build_lstm_sequences, build_prediction_sequences
from src.models.master import MasterTrainConfig, train_master_regressor
from src.portfolio.construct import build_top_k_submission, select_best_portfolio_strategy
from src.training.dataset_builder import DatasetBuildConfig, build_model_dataset
from src.training.metrics import precision_at_k, rank_ic, top_hit_rate, top_k_portfolio_return
from src.training.train_baseline import _default_time_split, assert_prediction_date, save_dataframe, validate_submission
from src.utils.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MASTER-style market-guided transformer baseline.")
    parser.add_argument("--config", default="configs/master_alpha.yaml")
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
    processed_path = Path(cfg["data"]["processed_path"])
    if cfg["data"].get("reuse_processed") and processed_path.exists():
        df = pd.read_csv(processed_path, dtype={"stock_id": str})
    else:
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
            train_start = train_dates[-int(recent_train_days)]
            train_df = train_df[train_df["date"] >= train_start].copy()

    dataset = build_lstm_sequences(
        train_df=train_df,
        valid_df=valid_df,
        feature_columns=feature_columns,
        label_column=cfg["label"]["name"],
        lookback=cfg["deep"]["lookback"],
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
    dataset.x_infer = x_infer
    dataset.infer_meta = infer_meta
    master_cfg = MasterTrainConfig(
        hidden_dim=cfg["deep"]["hidden_dim"],
        num_heads=cfg["deep"]["num_heads"],
        num_layers=cfg["deep"]["num_layers"],
        ff_dim=cfg["deep"]["ff_dim"],
        dropout=cfg["deep"]["dropout"],
        batch_size=cfg["deep"]["batch_size"],
        epochs=cfg["deep"]["epochs"],
        learning_rate=cfg["deep"]["learning_rate"],
        weight_decay=cfg["deep"].get("weight_decay", 1e-4),
        early_stopping_patience=cfg["deep"].get("early_stopping_patience", 4),
        lr_decay_factor=cfg["deep"].get("lr_decay_factor", 0.5),
        min_lr=cfg["deep"].get("min_lr", 1e-5),
        market_gate_strength=cfg["deep"].get("market_gate_strength", 1.0),
        regression_weight=cfg["deep"].get("regression_weight", 0.7),
        rank_weight=cfg["deep"].get("rank_weight", 0.2),
        corr_weight=cfg["deep"].get("corr_weight", 0.1),
        official_rank_weight=cfg["deep"].get("official_rank_weight", 0.0),
        official_top_k=cfg["deep"].get("official_top_k", 5),
        official_top_k_weight=cfg["deep"].get("official_top_k_weight", 2.0),
        official_base_weight=cfg["deep"].get("official_base_weight", 1.0),
        official_temperature=cfg["deep"].get("official_temperature", 1.0),
        date_batching=cfg["deep"].get("date_batching", False),
        validation_strategy=cfg["deep"].get("validation_strategy", "proportional_positive_thr0.0"),
        validation_rank_weight=cfg["deep"].get("validation_rank_weight", 0.1),
        portfolio_return_weight=cfg["deep"].get("portfolio_return_weight", 0.0),
        portfolio_temperature=cfg["deep"].get("portfolio_temperature", 0.25),
        portfolio_top_k=cfg["deep"].get("portfolio_top_k", cfg["training"]["top_k"]),
        top_hit_weight=cfg["deep"].get("top_hit_weight", 0.0),
        top_hit_k=cfg["deep"].get("top_hit_k", 2),
        top_hit_temperature=cfg["deep"].get("top_hit_temperature", 0.25),
        label_clip=cfg["deep"].get("label_clip", 0.18),
        seed=cfg["seed"],
    )
    model, valid_pred_df = train_master_regressor(dataset, master_cfg)
    infer_pred_df = getattr(model, "infer_pred_df", None)

    eval_df = valid_pred_df.rename(columns={"label": cfg["label"]["name"]})
    metrics = {
        "model_name": "master",
        "n_train_sequences": int(len(dataset.x_train)),
        "n_valid_sequences": int(len(dataset.x_valid)),
        "n_inference_sequences": int(len(x_infer)),
        "validation_latest_date": str(pd.Timestamp(valid_pred_df["date"].max()).date()),
        "inference_date": str(inference_date.date()),
        "n_features": int(len(feature_columns)),
        "rank_ic": rank_ic(eval_df, cfg["label"]["name"], "score"),
        "precision_at_k": precision_at_k(eval_df, cfg["label"]["name"], "score", cfg["training"]["top_k"]),
        "top1_hit_rate": top_hit_rate(eval_df, cfg["label"]["name"], "score", true_top_k=1, pred_top_k=1),
        "top2_hit_rate": top_hit_rate(eval_df, cfg["label"]["name"], "score", true_top_k=2, pred_top_k=2),
        "top_k_portfolio_return": top_k_portfolio_return(
            eval_df, cfg["label"]["name"], "score", cfg["training"]["top_k"]
        ),
    }

    strategies = cfg.get("portfolio", {}).get(
        "strategies",
        ["proportional_positive_thr0.0", "equal_weight", "softmax_t0.6", "positive_only_thr0.0"],
    )
    best_strategy, strategy_results = select_best_portfolio_strategy(
        eval_df,
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
        if cfg.get("output", {}).get("inference_date") or cfg.get("data", {}).get("benchmark_end_date"):
            raise ValueError(f"No latest inference rows found for configured T={inference_date.date()}.")
        latest_date = valid_pred_df["date"].max()
        latest_df = valid_pred_df[valid_pred_df["date"] == latest_date].copy()
        metrics["submission_source"] = "validation_fallback"
        metrics["submission_date"] = str(pd.Timestamp(latest_date).date())
    assert_prediction_date(latest_df, inference_date, "MASTER submission source prediction")

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
