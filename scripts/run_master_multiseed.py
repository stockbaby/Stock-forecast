from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.portfolio.construct import build_top_k_submission, evaluate_portfolio_strategy
from src.training.metrics import precision_at_k, rank_ic, top_k_portfolio_return
from src.training.train_baseline import assert_prediction_date, save_dataframe, validate_submission
from src.utils.config import load_yaml_config


def _prediction_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"master_seed_{seed}_predictions.csv"


def _latest_prediction_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"master_seed_{seed}_latest_predictions.csv"


def _metrics_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"master_seed_{seed}_metrics.json"


def _submission_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"master_seed_{seed}_submission.csv"


def _write_seed_config(base_cfg: dict, output_dir: Path, seed: int) -> Path:
    cfg = dict(base_cfg)
    cfg["seed"] = int(seed)
    cfg["data"] = dict(base_cfg["data"])
    cfg["data"]["reuse_processed"] = True
    cfg["output"] = dict(base_cfg["output"])
    cfg["output"]["prediction_path"] = str(_prediction_path(output_dir, seed))
    cfg["output"]["latest_prediction_path"] = str(_latest_prediction_path(output_dir, seed))
    cfg["output"]["metrics_path"] = str(_metrics_path(output_dir, seed))
    cfg["output"]["submission_path"] = str(_submission_path(output_dir, seed))
    config_path = output_dir / f"master_seed_{seed}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def _run_seed(config_path: Path, python_exe: str) -> None:
    command = [python_exe, "scripts/train_master_baseline.py", "--config", str(config_path)]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _load_seed_prediction(path: Path, seed: int, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"stock_id": str})
    if "label" in df.columns and label_col not in df.columns:
        df = df.rename(columns={"label": label_col})
    if label_col not in df.columns:
        df[label_col] = pd.NA
    df["stock_id"] = df["stock_id"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    return df[["stock_id", "date", label_col, "score"]].rename(columns={"score": f"score_seed_{seed}"})


def aggregate_predictions(prediction_paths: list[Path], seeds: list[int], label_col: str) -> pd.DataFrame:
    frames = [_load_seed_prediction(path, seed, label_col) for path, seed in zip(prediction_paths, seeds)]
    merged = frames[0]
    for frame in frames[1:]:
        score_cols = [col for col in frame.columns if col.startswith("score_seed_")]
        merged = merged.merge(frame[["stock_id", "date", *score_cols]], on=["stock_id", "date"], how="inner")
    score_cols = [col for col in merged.columns if col.startswith("score_seed_")]
    merged["score"] = merged[score_cols].mean(axis=1)
    merged["score_std"] = merged[score_cols].std(axis=1).fillna(0.0)
    return merged[["stock_id", "date", "score", "score_std", label_col]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MASTER over multiple seeds and average predictions.")
    parser.add_argument("--config", default="configs/master_alpha_official_rank.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62, 72, 82])
    parser.add_argument("--output-dir", default="outputs/predictions/master_multiseed")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml_config(args.config)
    label_col = base_cfg["label"]["name"]
    expected_inference_date = (
        base_cfg.get("output", {}).get("inference_date")
        or base_cfg.get("data", {}).get("benchmark_end_date")
    )

    prediction_paths: list[Path] = []
    latest_prediction_paths: list[Path] = []
    for seed in args.seeds:
        config_path = _write_seed_config(base_cfg, output_dir, seed)
        pred_path = _prediction_path(output_dir, seed)
        latest_path = _latest_prediction_path(output_dir, seed)
        prediction_paths.append(pred_path)
        latest_prediction_paths.append(latest_path)
        has_required_outputs = pred_path.exists() and (not expected_inference_date or latest_path.exists())
        if args.aggregate_only or (args.skip_existing and has_required_outputs):
            continue
        _run_seed(config_path, args.python_exe)

    missing = [str(path) for path in prediction_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing seed prediction files: {missing}")
    if expected_inference_date:
        missing_latest = [str(path) for path in latest_prediction_paths if not path.exists()]
        if missing_latest:
            raise FileNotFoundError(f"Missing seed latest prediction files for T={expected_inference_date}: {missing_latest}")

    averaged = aggregate_predictions(prediction_paths, args.seeds, label_col=label_col)
    aggregate_prediction_path = output_dir / "master_multiseed_predictions.csv"
    save_dataframe(averaged, aggregate_prediction_path)
    latest_averaged = None
    aggregate_latest_prediction_path = None
    if expected_inference_date:
        latest_averaged = aggregate_predictions(latest_prediction_paths, args.seeds, label_col=label_col)
        assert_prediction_date(latest_averaged, expected_inference_date, "MASTER multiseed latest prediction")
        aggregate_latest_prediction_path = output_dir / "master_multiseed_latest_predictions.csv"
        save_dataframe(latest_averaged, aggregate_latest_prediction_path)

    eval_df = averaged.rename(columns={label_col: "label"})
    metrics = {
        "seeds": args.seeds,
        "prediction_path": str(aggregate_prediction_path),
        "latest_prediction_path": str(aggregate_latest_prediction_path) if aggregate_latest_prediction_path else None,
        "expected_inference_date": str(pd.Timestamp(expected_inference_date).date()) if expected_inference_date else None,
        "rank_ic": rank_ic(eval_df, "label", "score"),
        "precision_at_k": precision_at_k(eval_df, "label", "score", base_cfg["training"]["top_k"]),
        "top_k_portfolio_return": top_k_portfolio_return(eval_df, "label", "score", base_cfg["training"]["top_k"]),
    }
    strategy_results = []
    for strategy in base_cfg.get("portfolio", {}).get("strategies", ["proportional_positive_thr0.0", "softmax_t0.6"]):
        strategy_results.append(
            evaluate_portfolio_strategy(
                eval_df,
                label_col="label",
                score_col="score",
                strategy=strategy,
                top_k=base_cfg["training"]["top_k"],
                max_weight_sum=base_cfg["training"]["max_weight_sum"],
                temperature=base_cfg.get("portfolio", {}).get("temperature", 1.0),
            )
        )
    metrics["portfolio_strategy_results"] = strategy_results
    best_strategy = max(strategy_results, key=lambda item: item["mean_return"])["strategy"]
    metrics["selected_portfolio_strategy"] = best_strategy

    if latest_averaged is not None:
        latest = latest_averaged.copy()
        metrics["submission_source"] = "latest_inference"
        metrics["submission_date"] = str(pd.Timestamp(latest["date"].max()).date())
    else:
        latest = averaged[averaged["date"] == averaged["date"].max()].copy()
        metrics["submission_source"] = "validation_fallback"
        metrics["submission_date"] = str(pd.Timestamp(latest["date"].max()).date())
    submission = build_top_k_submission(
        latest,
        score_col="score",
        stock_col="stock_id",
        top_k=base_cfg["training"]["top_k"],
        max_weight_sum=base_cfg["training"]["max_weight_sum"],
        strategy=best_strategy,
        temperature=base_cfg.get("portfolio", {}).get("temperature", 1.0),
    )
    validate_submission(submission, base_cfg["training"]["top_k"], base_cfg["training"]["max_weight_sum"])
    save_dataframe(submission, output_dir / "master_multiseed_submission.csv")
    (output_dir / "master_multiseed_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(submission.to_string(index=False))


if __name__ == "__main__":
    main()
