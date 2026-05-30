from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _norm_stock(value: object) -> str:
    return str(value).split(".")[0].zfill(6)


def build_multiseed(input_paths: list[Path], output_path: Path) -> pd.DataFrame:
    frames = []
    for path in input_paths:
        seed = path.stem.split("_seed_")[-1].split("_")[0]
        df = pd.read_csv(path, dtype={"stock_id": str})
        if not {"stock_id", "date", "score"}.issubset(df.columns):
            raise ValueError(f"{path} must contain stock_id,date,score")
        part = df[["stock_id", "date", "score"]].copy()
        part["stock_id"] = part["stock_id"].map(_norm_stock)
        part = part.rename(columns={"score": f"score_seed_{seed}"})
        frames.append(part)
    if not frames:
        raise ValueError("No input prediction files were provided")

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["stock_id", "date"], how="inner")
    seed_cols = [col for col in out.columns if col.startswith("score_seed_")]
    out["score_mean"] = out[seed_cols].mean(axis=1)
    out["score"] = out["score_mean"]
    out = out.sort_values("score", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Average latest prediction scores across seeds.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    paths = sorted(Path().glob(args.input_glob))
    out = build_multiseed(paths, Path(args.output_path))
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
