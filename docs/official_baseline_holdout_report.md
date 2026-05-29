# Official Baseline Holdout Report

Strict replay dates: 2026-04-17, 2026-04-24, 2026-04-30, 2026-05-08, 2026-05-15.

Important naming:

- `official_baseline_top5_equal`: official THU baseline submission style, Top5 equal weight.
- `official_baseline_top1`: first ranked stock from the official baseline output.
- `lightgbm_local_baseline`: our local LightGBM baseline row previously named `baseline`.
- `fallback_return`: MASTER-based fallback portfolio used by the dynamic-switch experiment, not the official baseline.

The official baseline evaluation uses the published official checkpoint and, for each T date, only feeds market data cut off at T. It does not train or infer using post-T data.

## Overall Summary

| model | hit_rate | mean_return | p05_return | negative_rate | max_drawdown | mean_excess_vs_fallback |
|---|---:|---:|---:|---:|---:|---:|
| stockmixer_official | 40.00% | 4.24% | -5.78% | 60.00% | -6.13% | 5.75% |
| stockmixer_official_multiseed | 40.00% | 2.88% | -5.71% | 60.00% | -11.22% | 4.39% |
| timexer | 40.00% | 2.59% | -6.29% | 60.00% | -3.75% | 4.10% |
| master | 60.00% | 2.01% | -3.38% | 40.00% | -3.74% | 3.52% |
| official_baseline_top1 | 80.00% | 1.62% | -8.39% | 20.00% | -10.55% | 3.12% |
| stockmixer_fast | 40.00% | 0.64% | -5.37% | 60.00% | -10.07% | 2.15% |
| official_baseline_top5_equal | 40.00% | 0.63% | -2.32% | 60.00% | -4.10% | 2.14% |
| timexer_multiseed | 40.00% | -0.21% | -8.56% | 60.00% | -8.88% | 1.30% |
| lightgbm_local_baseline | 40.00% | -0.28% | -7.03% | 60.00% | -13.52% | 1.23% |
| stockmixer_lite_multiseed | 20.00% | -0.81% | -4.41% | 80.00% | -9.69% | 0.70% |
| master_multiseed | 20.00% | -1.88% | -5.89% | 80.00% | -10.73% | -0.37% |
| stockmixer_lite | 20.00% | -3.27% | -6.54% | 80.00% | -14.44% | -1.76% |

## Per-Window Notes

### 2026-04-17

Fallback: -4.35%.

Top performers: `stockmixer_official_multiseed` +17.65%, `lightgbm_local_baseline` +12.69%, `stockmixer_fast` +6.04%, `master` +2.40%, `official_baseline_top5_equal` +2.35%.

### 2026-04-24

Fallback: -1.89%.

Top performers: `official_baseline_top1` +15.43%, `stockmixer_fast` +7.56%, `official_baseline_top5_equal` +4.94%, `stockmixer_lite` +1.23%.

### 2026-04-30

Fallback: +9.74%.

Top performers: `timexer_multiseed` +12.90%, `master` +9.74%, `stockmixer_official` +9.74%, `master_multiseed` +9.74%. Official baseline Top5 equal was -0.58%, below fallback by -10.31%.

### 2026-05-08

Fallback: -6.13%.

Top performers: `timexer` +20.30%, `master` +3.60%, `official_baseline_top1` +1.94%, `official_baseline_top5_equal` -0.90%. Although official Top5 was negative, it still beat fallback by +5.23%.

### 2026-05-15

Fallback: -4.90%.

Top performers: `stockmixer_official` +23.43%, `stockmixer_lite_multiseed` +8.37%, `stockmixer_official_multiseed` +8.37%, `timexer` +3.72%. Official Top5 equal was -2.67%, beating fallback by +2.23%; official Top1 was -10.55%.

## Readout

Official baseline Top5 equal is a useful conservative comparator: it has smaller downside than many single-stock Top1 models, but it is not the strongest selector here.

For competition submission strategy, the data supports keeping `stockmixer_official`, `stockmixer_official_multiseed`, `timexer`, and `master` as the main challenger set. Official baseline Top5 equal is more suitable as a conservative fallback/reference than as the primary alpha source.

Multi-seed should not be used blindly. It helped `stockmixer_official` on 2026-04-17 and 2026-05-15, but failed badly on 2026-04-30 when multiple models agreed on 300014 and the realized return was -4.01% versus fallback +9.74%. Seed/model agreement needs a market/fallback filter before triggering all-in.
