from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_CODE_DIR = Path(__file__).resolve().parent
APP_HELPER_DIR = APP_CODE_DIR / "src"
if str(APP_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(APP_HELPER_DIR))

from path_utils import add_project_root_to_path

PROJECT_ROOT = add_project_root_to_path()

from src.training.dataset_builder import DatasetBuildConfig, build_model_dataset
from src.utils.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset for official app/ entrypoint.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "a_stage_round1.yaml"),
    )
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
    print(f"featurework completed: {len(df)} rows")


if __name__ == "__main__":
    main()
