from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deep_sequence import build_lstm_sequences
from src.models.stockmixer import StockMixerTrainConfig, train_stockmixer_regressor
from src.portfolio.construct import build_top_k_submission, select_best_portfolio_strategy
from src.training.dataset_builder import DatasetBuildConfig, build_model_dataset
from src.training.metrics import precision_at_k, rank_ic, top_k_portfolio_return
from src.training.train_baseline import _default_time_split, save_dataframe, validate_submission
from src.utils.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a StockMixer-style baseline on alpha-style features.")
    parser.add_argument("--config", default="configs/stockmixer_alpha.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    build_cfg = DatasetBuildConfig(
        raw_dir=cfg["data"]["raw_dir"],
        processed_path=cfg["data"]["processed_path"],
        market_index_path=cfg["data"].get("market_index_path"),
        windows=cfg["features"]["lookback_windows"],
        label_name=cfg["label"]["name"],
        buy_offset=cfg["label"]["horizon_buy_offset"],
        sell_offset=cfg["label"]["horizon_sell_offset"],
        sell_fallback_offset=cfg["label"].get("horizon_sell_fallback_offset"),
    )
    df = build_model_dataset(build_cfg)
    df["date"] = pd.to_datetime(df["date"])

    feature_columns = [
        col
        for col in df.columns
        if col not in {"date", "stock_id", cfg["label"]["name"]}
        and not col.startswith("Unnamed:")
    ]
    model_df = df.dropna(subset=[cfg["label"]["name"]]).copy()
    train_df, valid_df = _default_time_split(model_df, valid_days=cfg["training"].get("valid_days"))

    dataset = build_lstm_sequences(
        train_df=train_df,
        valid_df=valid_df,
        feature_columns=feature_columns,
        label_column=cfg["label"]["name"],
        lookback=cfg["deep"]["lookback"],
    )
    mixer_cfg = StockMixerTrainConfig(
        hidden_dim=cfg["deep"]["hidden_dim"],
        mixer_dim=cfg["deep"]["mixer_dim"],
        temporal_dim=cfg["deep"]["temporal_dim"],
        dropout=cfg["deep"]["dropout"],
        batch_size=cfg["deep"]["batch_size"],
        epochs=cfg["deep"]["epochs"],
        learning_rate=cfg["deep"]["learning_rate"],
        weight_decay=cfg["deep"].get("weight_decay", 1e-4),
        early_stopping_patience=cfg["deep"].get("early_stopping_patience", 4),
        lr_decay_factor=cfg["deep"].get("lr_decay_factor", 0.5),
        min_lr=cfg["deep"].get("min_lr", 1e-5),
        recent_weight_power=cfg["deep"].get("recent_weight_power", 1.8),
        regression_weight=cfg["deep"].get("regression_weight", 0.7),
        rank_weight=cfg["deep"].get("rank_weight", 0.2),
        corr_weight=cfg["deep"].get("corr_weight", 0.1),
        label_clip=cfg["deep"].get("label_clip", 0.2),
        patch_sizes=tuple(cfg["deep"].get("patch_sizes", [5, 10, 20, 30])),
    )

    seed_predictions: list[pd.DataFrame] = []
    for seed in cfg["deep"].get("seeds", [cfg.get("seed", 42)]):
        mixer_cfg.seed = int(seed)
        _, seed_pred_df = train_stockmixer_regressor(dataset, mixer_cfg)
        seed_pred_df = seed_pred_df.rename(columns={"score": f"score_seed_{seed}"})
        seed_predictions.append(seed_pred_df)

    valid_pred_df = seed_predictions[0][["stock_id", "date", "label"]].copy()
    score_cols = []
    for seed_pred_df in seed_predictions:
        seed_score_col = [col for col in seed_pred_df.columns if col.startswith("score_seed_")][0]
        valid_pred_df = valid_pred_df.merge(
            seed_pred_df[["stock_id", "date", seed_score_col]],
            on=["stock_id", "date"],
            how="left",
        )
        score_cols.append(seed_score_col)
    valid_pred_df["score"] = valid_pred_df[score_cols].mean(axis=1)

    eval_df = valid_pred_df.rename(columns={"label": cfg["label"]["name"]})
    metrics = {
        "model_name": "stockmixer",
        "n_train_sequences": int(len(dataset.x_train)),
        "n_valid_sequences": int(len(dataset.x_valid)),
        "n_features": int(len(feature_columns)),
        "seeds": cfg["deep"].get("seeds", [cfg.get("seed", 42)]),
        "rank_ic": rank_ic(eval_df, cfg["label"]["name"], "score"),
        "precision_at_k": precision_at_k(eval_df, cfg["label"]["name"], "score", cfg["training"]["top_k"]),
        "top_k_portfolio_return": top_k_portfolio_return(
            eval_df, cfg["label"]["name"], "score", cfg["training"]["top_k"]
        ),
    }

    strategies = cfg.get("portfolio", {}).get(
        "strategies",
        ["proportional_positive", "equal_weight", "softmax", "positive_only"],
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

    latest_date = valid_pred_df["date"].max()
    latest_df = valid_pred_df[valid_pred_df["date"] == latest_date].copy()
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
