# LatestA Final Submission - 2026-05-30

## Submission Scope

This document records the final choice for the online A-stage submission window:

- Submission window: 2026-05-30 08:00 to 2026-05-31 23:59
- Data cutoff: 2026-05-29
- Target: T+5 open-to-open return after 2026-05-31
- Final files:
  - `app/model/result.csv`
  - `app/output/result.csv`

## Final Submitted Portfolio

```csv
stock_id,weight
300308,0.5721812227310785
002384,0.4278187772689215
```

The selected submission is the robust fusion candidate:

```text
75% MASTER single-seed score
25% StockMixer official multi-seed score
cross-sectional z-score transform
top2_softmax allocation
```

## Why This Was Selected

The hard all-in gate rejected single-stock all-in:

- no same-stock cross-family consensus;
- official baseline Top5 conflicted with the main alpha candidates;
- fallback support was weak;
- StockMixer single-seed `002384` was weakened by real multi-seed disagreement.

The portfolio selector then compared a small set of safer submission candidates:

| Candidate | Mean return | p05 return | Negative rate | Robust score |
|---|---:|---:|---:|---:|
| fusion_top2 | 3.6266% | 1.0165% | 0.0% | 0.0362 |
| master_top3 | 1.6337% | -1.5286% | 40.0% | 0.0013 |
| official_top5 | 0.6295% | -2.3151% | 60.0% | -0.0120 |
| master_ms_top3 | -0.2746% | -5.8845% | 60.0% | -0.0445 |

The fusion candidate is not all-in. It keeps the MASTER-supported `300308` as the anchor and caps the StockMixer attack signal `002384` below 50%.

## Multi-Seed Findings

Real multi-seed runs used seeds `42,52,62`.

| Model | Single-seed Top1 | Multi-seed Top1 | Seed consistency | Interpretation |
|---|---:|---:|---:|---|
| MASTER | 300308 | 300308 | 0.67, 2 unique | stable anchor |
| StockMixer official | 002384 | 600522 | 0.33, 3 unique | single-seed Top1 is unstable |
| TimeXer | 603296 | 603296 | 0.33, 3 unique | same Top1 after averaging, but weak seed agreement |

## Selector Summary

Three auxiliary selectors were used:

1. Gate selector: rejected all-in, support score `0.375`.
2. Portfolio selector: selected `fusion_top2`.
3. Weight meta-learner: recommended capped top2/top3 concentration with max single-stock weight `0.60`.

The selectors are auxiliary because only five recent online windows are available.

## Key Artifacts

```text
outputs/latestA_ensemble_weights_20260529_wide/candidate_1.csv
outputs/latestA_ensemble_weights_20260529_wide/leaderboard.json
outputs/latestA_ensemble_weights_20260529_wide/summary.json
outputs/final_submission_decider_latestA_20260529/candidate_diagnostic_table.csv
outputs/final_submission_decider_latestA_20260529/decision_summary.json
outputs/three_selectors_latestA_20260529/three_selector_report.json
scripts/search_latest_ensemble_weights.py
scripts/run_three_selectors.py
scripts/final_submission_decider.py
```
