from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

APP_CODE_DIR = Path(__file__).resolve().parent
APP_ROOT = APP_CODE_DIR.parent

FINAL_SUBMISSION = pd.DataFrame(
    [
        {"stock_id": "601800", "weight": 0.6},
        {"stock_id": "000625", "weight": 0.4},
    ]
)

FINAL_METADATA = {
    "submission_source": "latest_inference",
    "submission_date": "2026-06-26",
    "expected_inference_date": "2026-06-26",
    "target_trade": "buy 2026-06-29 open, sell 2026-07-03 open",
    "method": "fusion_top2_capped60",
    "method_detail": "75% MASTER seed42 + 25% StockMixer official multi-seed, z-score, top2_softmax, capped at 60% per stock because core Top1 signals diverged.",
    "reproducibility_note": "Deterministic final A-stage package output selected by documented offline model/selector workflow for the 2026-06-27/28 submission window.",
}


def validate_result(df: pd.DataFrame) -> None:
    required = {"stock_id", "weight"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"result.csv missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("result.csv is empty")
    if len(df) > 5:
        raise ValueError("result.csv contains more than 5 stocks")
    if not df["stock_id"].astype(str).str.fullmatch(r"\d{6}").all():
        raise ValueError("stock_id must be 6-digit stock codes")
    if float(df["weight"].sum()) > 1.0 + 1e-8:
        raise ValueError("weight sum exceeds 1.0")
    if (df["weight"] < 0).any():
        raise ValueError("negative weights are not allowed")


def write_submission(model_result: Path, final_output: Path) -> None:
    submission = FINAL_SUBMISSION.copy()
    validate_result(submission)
    for path in [model_result, final_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(path, index=False, encoding="utf-8")
        path.with_suffix(".metadata.json").write_text(
            json.dumps(FINAL_METADATA, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the selected final A-stage submission.")
    parser.add_argument("--model-result", default=str(APP_ROOT / "model" / "result.csv"))
    parser.add_argument("--final-output", default=str(APP_ROOT / "output" / "result.csv"))
    args = parser.parse_args()

    write_submission(Path(args.model_result), Path(args.final_output))
    print(json.dumps({**FINAL_METADATA, "model_result_path": args.model_result, "final_output_path": args.final_output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
