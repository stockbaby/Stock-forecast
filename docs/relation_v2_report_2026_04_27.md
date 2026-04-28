# Relation 2.0 Experiment Report

## Summary

This run extends the previous `MASTER official-rank + industry relation` candidate with beta, volatility, liquidity, and historical-correlation peer signals.

The selected stable candidate is now synced to:

```text
app/model/result.csv
app/output/result.csv
```

## Selected Candidate

```text
score_relation =
  1.35 * score_z
  -0.35 * industry_rank_score
  +0.05 * beta_20_rank
  -0.15 * volatility_20_rank
  +0.10 * liquidity_20_rank
  +0.10 * corr_peer_score
```

`corr_peer_score` is the score-weighted signal from the most positively correlated neighbors over a 60-day return window.

Submission:

```csv
stock_id,weight
002422,0.4435093966463539
688981,0.24835891346065442
300433,0.1419101612088723
688008,0.0883365840656386
002049,0.07788494461848086
```

## Results

| Candidate | Strategy | 91d mean | 91d std | Note |
|---|---|---:|---:|---|
| MASTER submitted | `proportional_positive_thr0.0` | `0.01904` | `0.03941` | original submitted MASTER |
| MASTER relation best | `softmax_t0.6` | `0.02177` | `0.05253` | previous industry relation best |
| MASTER relation 2.0 max-mean | `softmax_t0.6` | `0.02210` | `0.05082` | highest 91d mean |
| MASTER relation 2.0 stable | `softmax_t0.6` | `0.02210` | `0.05158` | selected stable candidate |

Window stability:

| Candidate | 20d | 40d | 60d | 90d |
|---|---:|---:|---:|---:|
| MASTER relation best | `0.03275` | `0.00841` | `0.01821` | `0.02214` |
| MASTER relation 2.0 stable | `0.03279` | `0.00853` | `0.01821` | `0.02244` |

The selected stable candidate is slightly below the max-mean variant on 91-day mean, but it is no worse than the previous relation best on all 20/40/60/90-day windows.

## Multi-Seed MASTER

MASTER was trained with:

```text
seed=[42,52,62,72,82]
```

Naive mean prediction did not improve the result:

```text
MASTER multi-seed average mean_return = 0.01584
```

The extra seeds had weaker Top-K portfolio returns, so the averaged prediction diluted the strong seed 42 signal. The generated artifacts are retained for future uncertainty filtering or selective seed weighting, but they are not used as the current submission candidate.

## Artifacts

```text
scripts/relation_postprocess.py
scripts/search_relation_v2.py
scripts/run_master_multiseed.py
outputs/submissions/result_master_relation_v2_stable.csv
outputs/predictions/master_relation_v2_stable_metrics.json
outputs/predictions/master_multiseed/master_multiseed_metrics.json
outputs/predictions/relation_v2_stability_ranked.json
```

## Relation 2.1 Update

A follow-up stability search added seed-disagreement features and a simple regime-risk penalty. The useful increment came from the regime-risk penalty and a slightly stronger correlation-neighbor term; direct uncertainty penalties were tested but did not improve the stable candidate.

Selected relation 2.1 parameters:

```text
alpha=-0.35
beta_alpha=0.05
vol_alpha=-0.175
liquidity_alpha=0.075
corr_alpha=0.125
uncertainty_alpha=0.0
uncertainty_rank_alpha=0.0
regime_risk_alpha=-0.05
strategy=softmax_t0.6
```

| Candidate | 91d mean | 91d std | 20d | 40d | 60d | 90d |
|---|---:|---:|---:|---:|---:|---:|
| relation 2.0 stable | `0.02210` | `0.05158` | `0.03279` | `0.00853` | `0.01821` | `0.02244` |
| relation 2.1 stable | `0.02216` | `0.05165` | `0.03286` | `0.00855` | `0.01829` | `0.02251` |

Current synced submission:

```csv
stock_id,weight
002422,0.4366668350683564
688981,0.26023331542128153
300433,0.13515239050984554
688008,0.08900648900769771
002049,0.07894096999281869
```

Additional artifacts:

```text
scripts/search_relation_v21.py
scripts/fast_relation_stability_search.py
configs/master_alpha_topk_v2.yaml
outputs/submissions/result_master_relation_v21_stable.csv
outputs/predictions/master_relation_v21_stable_metrics.json
outputs/relation_fast_stable_v21_small/
```

## Top-K Training and Regime Weighting Update

MASTER was updated to support date-batched Top-K/listwise training, so the listwise and pairwise top-k losses are computed inside same-day stock cross sections instead of random mixed batches. The first date-batched Top-K candidate did not improve validation performance:

```text
configs/master_alpha_topk_v2.yaml
mean_return = 0.01561
```

This model is retained as a reproducible experiment, but it is not used in the current submission.

Regime-aware weighting was then searched on top of the stronger relation score. A dynamic temperature switch improved some recent windows but failed the full stability constraint. The selected stable update instead combines a slightly stronger regime-risk score penalty with a more concentrated softmax temperature.

Selected relation 2.2 topk-regime parameters:

```text
alpha=-0.325
beta_alpha=0.05
vol_alpha=-0.20
liquidity_alpha=0.10
corr_alpha=0.125
regime_risk_alpha=-0.075
strategy=softmax_t0.55
```

| Candidate | 91d mean | 91d std | 20d | 40d | 60d | 90d |
|---|---:|---:|---:|---:|---:|---:|
| relation 2.1 stable | `0.02216` | `0.05165` | `0.03286` | `0.00855` | `0.01829` | `0.02251` |
| relation 2.2 topk-regime | `0.02219` | `0.05226` | `0.03323` | `0.00871` | `0.01849` | `0.02255` |

Current synced submission:

```csv
stock_id,weight
002422,0.4467639190029921
688981,0.2692725163001993
300433,0.12745631566083404
688008,0.08288095756045455
002049,0.07362629147552009
```

Additional artifacts:

```text
outputs/submissions/result_master_relation_v22_topk_regime.csv
outputs/predictions/master_relation_v22_topk_regime_metrics.json
outputs/predictions/master_relation_v22_topk_regime_scored.csv
outputs/regime_weighting_v1/
outputs/relation_fast_temp055/
scripts/search_regime_weighting.py
```
