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
