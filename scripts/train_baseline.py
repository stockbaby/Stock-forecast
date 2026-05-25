from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_baseline import TrainConfig, run_training
from src.utils.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline model and export predictions/submission.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    train_cfg = TrainConfig(
        processed_path=cfg["data"]["processed_path"],
        label_name=cfg["label"]["name"],
        metrics_path=cfg["output"]["metrics_path"],
        prediction_path=cfg["output"]["prediction_path"],
        submission_path=cfg["output"]["submission_path"],
        model_type=cfg["training"]["model_type"],
        top_k=cfg["training"]["top_k"],
        max_weight_sum=cfg["training"]["max_weight_sum"],
        train_end=cfg["training"].get("train_end"),
        valid_start=cfg["training"].get("valid_start"),
        valid_end=cfg["training"].get("valid_end"),
        valid_days=cfg["training"].get("valid_days"),
        portfolio_strategies=cfg.get("portfolio", {}).get("strategies"),
        portfolio_temperature=cfg.get("portfolio", {}).get("temperature", 1.0),
        inference_date=cfg.get("output", {}).get("inference_date") or cfg.get("data", {}).get("benchmark_end_date"),
        latest_prediction_path=cfg.get("output", {}).get("latest_prediction_path"),
    )
    metrics = run_training(train_cfg)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
