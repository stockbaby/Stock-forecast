# Latest Progress 2026-05-29

## Current Decision

This round shifts the project from "which single model do we believe" to "which Top1 signal deserves all-in today".

The current main candidate is:

```text
dynamic candidate switch
+ StockMixer/ensemble challenger gate
+ candidate Top1 agreement
+ multi-seed Top1 diagnostics
+ all-in when the gate passes
```

For the 2026-05-17 holdout replay:

- T: 2026-05-15
- Buy: 2026-05-18 open
- Sell: 2026-05-22 open
- selected submission: `688981,1.0`
- realized return: `0.101618`

The key interpretation is not "always all-in". It is:

- all-in is allowed when several model/blend candidates converge on the same Top1;
- seed agreement is a confirmation and risk signal, not a standalone permission;
- if model agreement is weak, fall back to `top2_softmax`, `dynamic_risk_budget`, or the MASTER/relation primary.

## What Changed

Implemented in commit `2297f48`:

- `top1_weight` is now a first-class main-candidate strategy in MASTER and StockMixer configs.
- MASTER and StockMixer support `top1_margin_weight` and `top1_margin_target`.
- Training metrics now include `top1_margin_z` and `top1_portfolio_return`.
- `scripts/dynamic_candidate_switch.py` now reads multi-seed score columns, computes seed Top1 votes, and gates all-in by candidate Top1 agreement.
- `scripts/run_multiseed_top1_holdout.py` now emits all-in gate artifacts:
  - `outputs/holdout_20260517/multiseed_top1/allin_candidates.csv`
  - `outputs/holdout_20260517/multiseed_top1/allin_stock_gate.csv`
  - `outputs/holdout_20260517/multiseed_top1/allin_recommendation.json`

## Holdout Evidence

### Dynamic Candidate Switch

Latest output:

```csv
stock_id,weight
688981,1.0
```

The latest gate selected `stockmixer_portfolio_lite_rank`.

All four challenger candidates had the same Top1:

```text
688981
```

So the candidate Top1 agreement was:

```text
4 / 4 = 1.0
```

This is the strongest support for all-in in this window.

### Multi-Seed Top1 Replay

Aggregate-only multi-seed leaderboard:

| Model | Mean Top1 | Mean return | Vote Top1 | Vote return | Vote share | Unique Top1 |
|---|---:|---:|---:|---:|---:|---:|
| stockmixer_lite | 688256 | 0.083750 | 688256 | 0.083750 | 0.667 | 2 |
| lstm | 600958 | 0.010471 | 688981 | 0.101618 | 0.333 | 3 |
| itransformer | 002415 | -0.018675 | 002415 | -0.018675 | 0.333 | 3 |
| timexer_fast | 002460 | -0.034589 | 300750 | -0.013981 | 0.333 | 3 |
| master_official | 600276 | -0.041465 | 002460 | -0.034589 | 0.333 | 3 |
| stockmixer_official | 300442 | -0.105785 | 300442 | -0.105785 | 0.667 | 2 |

Important lesson:

- `stockmixer_lite -> 688256` has strong seed agreement and good return.
- `stockmixer_official -> 300442` also has strong seed agreement but bad return.
- Therefore seed-only agreement cannot be enough for all-in.

The stricter aggregate gate reports `688981` as a high-return observation but keeps `allin_allowed=false`, because it is not independently supported by enough base-model votes in that aggregate table. This is intentional: the final all-in permission should come from cross-model/blend agreement, not one model's seed pattern alone.

## Why MASTER Is Not Always Best Here

MASTER remains the stable base model, but it is not necessarily the sharpest short-window Top1 model.

Likely reasons:

- MASTER was designed to learn broad cross-stock and market-guided structure, which helps average ranking quality but may smooth away short-term theme bursts.
- The current competition window is only about five trading days, so the payoff is dominated by whether the model catches the single strongest short-term target.
- StockMixer/ensemble candidates can be more reactive because they mix stock, time, and indicator signals with less architectural inertia.
- Relation postprocess is rule/statistics based and can improve robustness, but it may also dilute a very sharp Top1 unless the gate explicitly allows concentration.

So the current best posture is:

```text
MASTER/relation = stable default
StockMixer/ensemble = short-window challenger
gate = decides whether to all-in or fall back
```

## Follow-Up Research

### P0: Expand Holdout Windows

Do not tune the gate on only the 2026-05-15 window.

Next experiments:

- replay multiple recent T dates, for example 2026-04-17, 2026-04-24, 2026-04-30, 2026-05-08, 2026-05-15;
- run the same multi-seed and dynamic-switch pipeline on each;
- record whether candidate agreement predicts profitable all-in.

Deliverable:

```text
scripts/run_recent_holdout_matrix.py
docs/top1_gate_holdout_matrix.md
```

Decision metric:

```text
all-in hit rate, mean return, p05 return, negative rate, max drawdown
```

### P1: Make The Gate Learnable

The current gate is rule-based. That is good for transparency, but it cannot learn interactions such as:

- high margin but bad liquidity;
- model agreement in the same architecture family;
- strong sector momentum but weak index regime;
- seed agreement that is useful for one model but dangerous for another.

Next version:

- train a small logistic/GBDT selector on historical daily records;
- target: whether all-in Top1 beats fallback portfolio on that date;
- features:
  - candidate Top1 agreement count/share;
  - seed vote share;
  - top1 margin z;
  - top strength;
  - model family;
  - market risk/trend;
  - industry collective momentum;
  - volume confirmation;
  - liquidity and volatility.

Keep the learned selector small and walk-forward only. It should decide the portfolio mode, not replace the stock model.

### P2: Top1-Oriented Training

The new `top1_margin_weight` is only the first step.

Next losses worth implementing:

- `top_hit_loss`: directly rewards the true top label being near the top predictions.
- pairwise top-vs-bottom loss: force the true top bucket above the bottom bucket.
- listwise Top1 temperature loss: a sharper version of ListNet focused on the head.
- Top1 margin calibration: require score gap only when labels are also separated enough, to avoid forcing noisy days.

Training rule:

```text
batch by same date
optimize cross-sectional ranking
validate by Top1/all-in return, not only RankIC
```

### P3: Short-Window Theme And Confirmation Features

The all-in strategy needs "why this stock now" features.

Add or strengthen:

- 3/5/10 day relative strength vs HS300;
- turnover and amount acceleration;
- gap/open strength where available;
- industry collective momentum;
- same-industry breadth and volume confirmation;
- limit-up proximity / recent breakout style features if data supports them;
- liquidity penalty for fragile spikes.

These should feed both the models and the gate.

### P4: Relation Layer Upgrade

Current relation postprocess is effective but manual.

Next relation directions:

- industry relation;
- beta-neighbor relation;
- volatility-neighbor relation;
- liquidity-neighbor relation;
- return-correlation neighbors;
- hidden concept neighbors inspired by HIST-style relation modeling.

Near-term implementation should stay light:

```text
do relation scoring as postprocess first
promote only robust signals into model features later
```

### P5: Drift And Online Adaptation

Recent windows matter more than older windows, but naive recent-only training has been unstable.

Research direction:

- keep full-history model as anchor;
- add a recent-window adapter or residual head;
- walk-forward tune adapter weight;
- compare against DoubleAdapt-style incremental adaptation.

This is useful because the task window is short and market regimes move.

### P6: Final Submission Policy

The final submitter should not be a fixed model name.

Recommended policy:

```text
1. Generate MASTER/relation stable submission.
2. Generate StockMixer and ensemble challenger submissions.
3. Run multi-seed diagnostics.
4. Run dynamic candidate switch.
5. If cross-model/blend Top1 agreement is strong, submit all-in.
6. If disagreement is high, submit top2/top3 or MASTER/relation.
7. Keep equal-weight Top5 as conservative backup.
```

## External References Checked

- MASTER official code: https://github.com/SJTU-DMTai/MASTER
- MASTER paper page: https://ojs.aaai.org/index.php/AAAI/article/view/27767
- StockMixer official code: https://github.com/SJTU-DMTai/StockMixer
- StockMixer paper page: https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/
- HIST official code: https://github.com/Wentao-Xu/HIST
- HIST arXiv: https://arxiv.org/abs/2110.13716
- DoubleAdapt official code: https://github.com/SJTU-DMTai/DoubleAdapt
- DoubleAdapt arXiv: https://arxiv.org/abs/2306.09862

## Immediate Next Actions

1. Build a multi-window holdout matrix runner.
2. Use it to tune only gate thresholds, not model weights.
3. Train the Top1-margin MASTER/StockMixer configs for real, then rerun the same matrix.
4. Add short-window theme/volume confirmation features.
5. Promote the rule gate to a tiny walk-forward selector if rule thresholds remain brittle.
