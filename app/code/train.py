from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.code.src.path_utils import add_project_root_to_path

PROJECT_ROOT = add_project_root_to_path()

from src.training.train_baseline import TrainConfig, run_training
from src.utils.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Official app/ training entrypoint.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "a_stage_round1.yaml"),
    )
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
    )
    metrics = run_training(train_cfg)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
