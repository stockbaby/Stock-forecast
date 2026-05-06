# Latest Progress - 2026-05-06

## Current Submission

The current production result files are:

```text
app/model/result.csv
app/output/result.csv
```

Current aggressive submission:

```csv
stock_id,weight
002493,1.0
```

Using the recovered A-stage realized open-to-open window:

```text
T = 2026-04-24
Buy open = 2026-04-27
Sell open = 2026-04-30
```

The replayed score is:

```text
0.1066892464013548
```

This is much higher than the original public score `0.012833583539611287`, because the earlier pipeline used the last labeled validation day instead of the true unlabeled online inference date.

## Fixed Critical Issue

The main bug was prediction-date misalignment:

- Training correctly uses rows with labels.
- Inference must score the unlabeled `T=2026-04-24` cross-section.
- The old training scripts built submissions from `valid_pred_df["date"].max()`, which was `2026-04-20`.
- The fixed MASTER and StockMixer flows now build separate latest inference sequences for `2026-04-24`.

Relevant commits:

```text
6398050 Add true latest MASTER inference
bcd617a Add regime candidate switching with StockMixer latest inference
```

## Portfolio Objective

The competition target is not global ranking quality. The actual target is:

```text
At time T, using only data available up to T,
choose up to 5 stocks and weights to maximize open(T+1) -> open(T+5) portfolio return.
```

Therefore RankIC, MSE, and global correlation are only auxiliary diagnostics. The model should be pushed toward head-selection and portfolio-return maximization.

Implemented:

- `portfolio_return_loss` for MASTER.
- `portfolio_return_loss` for StockMixer.
- New experiment configs:

```text
configs/master_alpha_portfolio_return.yaml
configs/stockmixer_alpha_fast_portfolio_return.yaml
```

Important note: early StockMixer fast tests showed that making this objective too dominant can overfit noisy heads. The production configs remain stable; portfolio-return configs are for controlled walk-forward experiments.

Relevant commit:

```text
c586fb0 Add portfolio return training objective
```

## Risk And Fallback

`top1_weight` is the best replayed A-stage result, but it has high variance.

Historical 91-day validation for `master_official + top1_weight`:

```text
mean     = 0.0212975786
std      = 0.0652199321
var      = 0.0042536395
neg_rate = 0.3736263736
min      = -0.15968676
p05      = -0.069150975
```

Fallback candidates:

```text
master_official + top2_softmax
mean     = 0.0214453710
std      = 0.0491337492
var      = 0.0024141253
neg_rate = 0.3186813187

master_official + confidence_topk
mean     = 0.0198166875
std      = 0.0522138149
var      = 0.0027262825
neg_rate = 0.3626373626
```

Current interpretation:

- `top1_weight` is the aggressive score-maximizing version.
- `top2_softmax` is the best historical risk-adjusted fallback.
- `confidence_topk` is useful when Top1 conviction is moderate but not enough for full concentration.

Relevant commit:

```text
a9c6614 Add confidence-aware dynamic allocation
```

## Multi-Method Validation

Added a unified validation script:

```text
scripts/validate_multiple_methods.py
```

Default methods:

```text
master_official
stockmixer_fast
stockmixer_official
master_multiseed
```

Default portfolio strategies:

```text
top1_weight
confidence_topk
top2_softmax
top3_softmax
proportional_positive_thr0.0
```

Run:

```powershell
D:\anaconda\envs\stock-forecast\python.exe scripts\validate_multiple_methods.py --output-dir outputs\method_validation
```

Outputs:

```text
outputs/method_validation/multi_method_validation.csv
outputs/method_validation/daily_records.json
```

Latest comparison highlights:

| Method | Strategy | Mean | Std | Var | Neg Rate | Latest A Score |
|---|---|---:|---:|---:|---:|---:|
| master_multiseed | top2_softmax | 0.024498 | 0.049118 | 0.002413 | 0.340659 | n/a |
| master_official | top2_softmax | 0.021445 | 0.049134 | 0.002414 | 0.318681 | 0.049636 |
| master_official | top1_weight | 0.021298 | 0.065220 | 0.004254 | 0.373626 | 0.106689 |
| master_official | confidence_topk | 0.019817 | 0.052214 | 0.002726 | 0.362637 | 0.062790 |

Relevant commit:

```text
7fd9085 Add multi-method validation script
```

## New Feature Direction

Added theme and event-style features:

- short-term momentum
- momentum acceleration
- volume breakout
- close-to-high / breakout strength
- industry collective momentum
- industry volume confirmation
- theme event pressure

These are intended to capture short-window theme-driven markets, which appeared important in the A-stage replay.

Relevant commit:

```text
b23ffc0 Add theme momentum breakout features
```

## Recommended Next Steps

1. Run `configs/master_alpha_portfolio_return.yaml` as a long experiment.
2. Validate it with `scripts/validate_multiple_methods.py`.
3. Use `top2_softmax` as the default fallback when Top1 variance or recent drawdown is too high.
4. Extend latest inference support to multi-seed MASTER so `master_multiseed + top2_softmax` can generate true future-window submissions.
5. For each new competition phase, always verify that the prediction file is generated for the true unlabeled `T` date before portfolio construction.
