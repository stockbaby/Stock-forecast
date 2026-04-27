from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.relation_postprocess import add_relation_score
from src.portfolio.construct import build_top_k_submission, evaluate_portfolio_strategy
from src.training.metrics import precision_at_k, rank_ic, top_k_portfolio_return
from src.training.train_baseline import save_dataframe, validate_submission


def _float_grid(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def _int_or_none_grid(text: str) -> list[int | None]:
    values: list[int | None] = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        values.append(None if token in {"none", "null", "na"} else int(token))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Search relation 2.0 peer/style post-processing parameters.")
    parser.add_argument("--prediction-path", required=True)
    parser.add_argument("--prepared-scored-path")
    parser.add_argument("--output-dir", default="outputs/relation_v2_search")
    parser.add_argument("--industry-map-path", default="data/raw/hs300_stock_list.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--label-col", default="y_ret_a_stage_round1_open_open")
    parser.add_argument("--source-score-col", default="score")
    parser.add_argument("--alphas", default="-0.7,-0.5,-0.3")
    parser.add_argument("--beta-alphas", default="-0.2,-0.1,0,0.1")
    parser.add_argument("--vol-alphas", default="-0.2,-0.1,0,0.1")
    parser.add_argument("--liquidity-alphas", default="-0.2,-0.1,0,0.1")
    parser.add_argument("--corr-alphas", default="-0.2,-0.1,0,0.1,0.2")
    parser.add_argument("--corr-window", type=int, default=60)
    parser.add_argument("--corr-top-n", type=int, default=8)
    parser.add_argument("--strategies", nargs="*", default=["softmax_t0.6", "proportional_positive_thr0.0", "equal_weight"])
    parser.add_argument("--caps", default="none,2,3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-weight-sum", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepared_scored_path:
        base_scored = pd.read_csv(args.prepared_scored_path, dtype={"stock_id": str})
        base_scored["date"] = pd.to_datetime(base_scored["date"])
    else:
        df = pd.read_csv(args.prediction_path, dtype={"stock_id": str})
        if "label" in df.columns and args.label_col not in df.columns:
            df = df.rename(columns={"label": args.label_col})

        base_scored = add_relation_score(
            df,
            industry_map_path=args.industry_map_path,
            mode="multi_peer",
            alpha=0.0,
            source_score_col=args.source_score_col,
            raw_dir=args.raw_dir,
            corr_window=args.corr_window,
            corr_top_n=args.corr_top_n,
        )

    rows: list[dict] = []
    caps = _int_or_none_grid(args.caps)
    for alpha, beta_alpha, vol_alpha, liquidity_alpha, corr_alpha in itertools.product(
        _float_grid(args.alphas),
        _float_grid(args.beta_alphas),
        _float_grid(args.vol_alphas),
        _float_grid(args.liquidity_alphas),
        _float_grid(args.corr_alphas),
    ):
        candidate = base_scored.copy()
        candidate["score_relation"] = (1.0 - alpha) * candidate["score_z"] + alpha * candidate["industry_rank_score"]
        candidate["score_relation"] += beta_alpha * candidate.get("beta_20_rank", 0.5)
        candidate["score_relation"] += vol_alpha * candidate.get("volatility_20_rank", 0.5)
        candidate["score_relation"] += liquidity_alpha * candidate.get("liquidity_20_rank", 0.5)
        candidate["score_relation"] += corr_alpha * candidate.get("corr_peer_score", 0.0)
        for strategy in args.strategies:
            for cap in caps:
                portfolio_eval = evaluate_portfolio_strategy(
                    candidate,
                    label_col=args.label_col,
                    score_col="score_relation",
                    strategy=strategy,
                    top_k=args.top_k,
                    max_weight_sum=args.max_weight_sum,
                    temperature=args.temperature,
                    industry_map_path=args.industry_map_path,
                    max_per_industry=cap,
                )
                rows.append(
                    {
                        "alpha": alpha,
                        "beta_alpha": beta_alpha,
                        "vol_alpha": vol_alpha,
                        "liquidity_alpha": liquidity_alpha,
                        "corr_alpha": corr_alpha,
                        "strategy": strategy,
                        "max_per_industry": cap,
                        "mean_return": float(portfolio_eval["mean_return"]),
                        "std_return": float(portfolio_eval.get("std_return", 0.0)),
                        "num_days": int(portfolio_eval["num_days"]),
                        "rank_ic": rank_ic(candidate, args.label_col, "score_relation"),
                        "precision_at_k": precision_at_k(candidate, args.label_col, "score_relation", args.top_k),
                        "top_k_portfolio_return": top_k_portfolio_return(
                            candidate, args.label_col, "score_relation", args.top_k
                        ),
                    }
                )

    rows.sort(key=lambda item: (item["mean_return"], -item["std_return"]), reverse=True)
    (output_dir / "search_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "leaderboard.json").write_text(
        json.dumps(rows[: args.top_n], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best = rows[0]
    best_scored = base_scored.copy()
    best_scored["score_relation"] = (1.0 - best["alpha"]) * best_scored["score_z"] + best["alpha"] * best_scored[
        "industry_rank_score"
    ]
    best_scored["score_relation"] += best["beta_alpha"] * best_scored.get("beta_20_rank", 0.5)
    best_scored["score_relation"] += best["vol_alpha"] * best_scored.get("volatility_20_rank", 0.5)
    best_scored["score_relation"] += best["liquidity_alpha"] * best_scored.get("liquidity_20_rank", 0.5)
    best_scored["score_relation"] += best["corr_alpha"] * best_scored.get("corr_peer_score", 0.0)
    latest = best_scored[best_scored["date"] == best_scored["date"].max()].copy()
    submission = build_top_k_submission(
        latest,
        score_col="score_relation",
        stock_col="stock_id",
        top_k=args.top_k,
        max_weight_sum=args.max_weight_sum,
        strategy=best["strategy"],
        temperature=args.temperature,
        industry_map_path=args.industry_map_path,
        max_per_industry=best["max_per_industry"],
    )
    validate_submission(submission, args.top_k, args.max_weight_sum)
    save_dataframe(submission, output_dir / "best_submission.csv")
    save_dataframe(best_scored, output_dir / "best_scored_predictions.csv")
    print(json.dumps(rows[: args.top_n], ensure_ascii=False, indent=2))
    print(submission.to_string(index=False))


if __name__ == "__main__":
    main()
