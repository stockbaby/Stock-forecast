# Top1 Gate Recent Holdout Matrix

Windows: 2026-05-15

## Core Metrics

| source | n | hit_rate | mean_return | p05_return | negative_rate | max_drawdown | hit_vs_fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stockmixer_lite | 1 | 0.000000 | -0.110517 | -0.110517 | 1.000000 | 0.000000 | 0.000000 |
| stockmixer_lite_multiseed | 1 | 0.000000 | -0.110517 | -0.110517 | 1.000000 | 0.000000 | 0.000000 |
| dynamic_switch | 0 |  |  |  |  |  | 0.000000 |

## Model Top1 Rows

| date | model | missing | buy_date | sell_date | top1 | return | fallback_return | hit_vs_fallback | margin_z | top_strength | seed_vote_share | seed_unique_count | seed_top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-15 | master | True |  |  |  |  |  |  |  |  |  |  |  |
| 2026-05-15 | stockmixer_lite | False | 2026-05-18 | 2026-05-22 | 300394 | -0.110517 |  | False | 0.582402 | 4.666952 |  |  |  |
| 2026-05-15 | stockmixer_fast | True |  |  |  |  |  |  |  |  |  |  |  |
| 2026-05-15 | stockmixer_official | True |  |  |  |  |  |  |  |  |  |  |  |
| 2026-05-15 | timexer | True |  |  |  |  |  |  |  |  |  |  |  |
| 2026-05-15 | master_multiseed | True |  |  |  |  |  |  |  |  |  |  |  |
| 2026-05-15 | stockmixer_lite_multiseed | False | 2026-05-18 | 2026-05-22 | 300394 | -0.110517 |  | False | 0.582402 | 4.666952 | 1.000000 | 1.000000 | 300394 |
| 2026-05-15 | stockmixer_official_multiseed | True |  |  |  |  |  |  |  |  |  |  |  |
| 2026-05-15 | timexer_multiseed | True |  |  |  |  |  |  |  |  |  |  |  |

## Dynamic Switch Rows

| date | selected | top1 | return | fallback_return | hit_vs_fallback | candidate_top1_agreement | candidate_top1_agreement_share | best_margin_z | best_top_strength | allin_selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-15 | primary |  |  |  | False | 0 | 0.000000 | 0.000000 | 0.000000 | False |

## Interpretation

- `return` is all-in Top1 realized open-to-open return for the requested T window.
- `hit_vs_fallback` checks whether all-in Top1 beat the MASTER dynamic-risk-budget fallback on the same T.
- `candidate_top1_agreement_share` is the dynamic-switch cross-candidate consistency signal.
- Multi-seed rows are confirmation diagnostics; they should not be used as a standalone all-in trigger unless the matrix keeps showing positive p05 and low negative rate.