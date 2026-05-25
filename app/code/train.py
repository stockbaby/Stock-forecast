from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

APP_CODE_DIR = Path(__file__).resolve().parent
APP_HELPER_DIR = APP_CODE_DIR / "src"
if str(APP_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(APP_HELPER_DIR))

from path_utils import add_project_root_to_path

PROJECT_ROOT = add_project_root_to_path()
APP_ROOT = APP_CODE_DIR.parent
from src.utils.config import load_yaml_config


def stage_mounted_data() -> None:
    data_root = PROJECT_ROOT / "data"
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ["stock_data.csv", "train.csv", "test.csv", "hs300_stock_list.csv", "hs300_index.csv"]:
        source = data_root / name
        target = raw_dir / name
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official app/ training entrypoint.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "master_alpha_official_rank.yaml"),
        help="Training config. Default reproduces the selected MASTER official-rank run.",
    )
    parser.add_argument(
        "--model-result",
        default=str(APP_ROOT / "model" / "result.csv"),
        help="Path used by test.py for fast offline prediction.",
    )
    parser.add_argument(
        "--final-output",
        default=str(APP_ROOT / "output" / "result.csv"),
        help="Also write the latest trained submission here for local checks.",
    )
    args = parser.parse_args()

    stage_mounted_data()
    cfg = load_yaml_config(args.config)
    script_path = PROJECT_ROOT / "scripts" / "train_master_baseline.py"
    subprocess.run(
        [sys.executable, str(script_path), "--config", str(Path(args.config))],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    submission_path = PROJECT_ROOT / cfg["output"]["submission_path"]
    model_result = Path(args.model_result)
    final_output = Path(args.final_output)
    model_result.parent.mkdir(parents=True, exist_ok=True)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(submission_path, model_result)
    shutil.copyfile(submission_path, final_output)

    metrics_path = PROJECT_ROOT / cfg["output"]["metrics_path"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    expected_date = cfg.get("output", {}).get("inference_date") or cfg.get("data", {}).get("benchmark_end_date")
    metadata = {
        "submission_source": metrics.get("submission_source"),
        "submission_date": metrics.get("submission_date"),
        "expected_inference_date": str(expected_date) if expected_date else None,
        "source_submission_path": str(submission_path),
        "source_metrics_path": str(metrics_path),
    }
    if metadata["submission_source"] != "latest_inference":
        raise ValueError(f"Submission must come from latest_inference, got {metadata['submission_source']!r}.")
    if expected_date and metadata["submission_date"] != str(expected_date):
        raise ValueError(
            f"Submission date mismatch: submission_date={metadata['submission_date']} expected_unlabeled_T={expected_date}."
        )
    model_result.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics["model_result_path"] = str(model_result)
    metrics["final_output_path"] = str(final_output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
