# Top1 Gate Recent Holdout Matrix

## Baseline Addendum

`fallback` in this report is the conservative MASTER-based fallback portfolio, not the LightGBM baseline model. The LightGBM baseline was added as a separate `baseline` row and was trained with the same strict per-T cutoff protocol.

Baseline result over the five windows:

- Mean return: `-0.002772`
- p05 return: `-0.070315`
- Negative-rate: `0.60`
- Hit-rate: `0.40`
- Beat fallback: `2/5`
- Average rank among model Top1 sources: `6.40`

Per-window baseline Top1:

| T | Top1 | Return | Fallback | Excess vs fallback |
|---|---:|---:|---:|---:|
| 2026-04-17 | 600482 | 0.126930 | -0.043546 | 0.170476 |
| 2026-04-24 | 000807 | -0.051170 | -0.018873 | -0.032296 |
| 2026-04-30 | 600150 | 0.004388 | 0.097394 | -0.093006 |
| 2026-05-08 | 002466 | -0.075101 | -0.061318 | -0.013783 |
| 2026-05-15 | 600150 | -0.018905 | -0.048987 | 0.030082 |

Interpretation: baseline is not a good primary challenger. It had one strong hit on `2026-04-17`, but overall it is below MASTER, StockMixer official, Timexer, and StockMixer fast on mean return and average rank. It can remain a weak diversity/reference signal, but should not drive the final submission gate.

Windows: 2026-04-17, 2026-04-24, 2026-04-30, 2026-05-08, 2026-05-15

## Core Metrics

| source | n | hit_rate | mean_return | p05_return | negative_rate | max_drawdown | hit_vs_fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stockmixer_official | 5 | 0.400000 | 0.042426 | -0.057764 | 0.600000 | -0.061318 | 0.400000 |
| stockmixer_official_multiseed | 5 | 0.400000 | 0.028817 | -0.057076 | 0.600000 | -0.112219 | 0.600000 |
| timexer | 5 | 0.400000 | 0.025917 | -0.062883 | 0.600000 | -0.037492 | 0.600000 |
| master | 5 | 0.600000 | 0.020122 | -0.033789 | 0.400000 | -0.037385 | 0.600000 |
| stockmixer_fast | 5 | 0.400000 | 0.006446 | -0.053680 | 0.600000 | -0.100709 | 0.600000 |
| timexer_multiseed | 5 | 0.400000 | -0.002108 | -0.085635 | 0.600000 | -0.088838 | 0.600000 |
| baseline | 5 | 0.400000 | -0.002772 | -0.070315 | 0.600000 | -0.135240 | 0.400000 |
| stockmixer_lite_multiseed | 5 | 0.200000 | -0.008063 | -0.044143 | 0.800000 | -0.096929 | 0.800000 |
| master_multiseed | 5 | 0.200000 | -0.018768 | -0.058852 | 0.800000 | -0.107302 | 0.000000 |
| dynamic_switch_allin_only | 3 | 0.000000 | -0.025331 | -0.038216 | 1.000000 | -0.054226 | 0.666667 |
| stockmixer_lite | 5 | 0.200000 | -0.032686 | -0.065416 | 0.800000 | -0.144433 | 0.600000 |
| dynamic_switch | 5 | 0.000000 | -0.037260 | -0.058852 | 1.000000 | -0.155709 | 0.400000 |

## Model Top1 Rows

| date | buy_date | sell_date | model | top1 | return | fallback_return | hit_vs_fallback | margin_z | top_strength | missing | seed_vote_share | seed_unique_count | seed_top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-17 | 2026-04-20 | 2026-04-24 | baseline | 600482 | 0.126930 | -0.043546 | True | 10.306204 | 13.153947 | False |  |  |  |
| 2026-04-17 | 2026-04-20 | 2026-04-24 | master | 688012 | 0.023968 | -0.043546 | True | 0.144685 | 3.227073 | False |  |  |  |
| 2026-04-17 | 2026-04-20 | 2026-04-24 | stockmixer_lite | 601899 | -0.024096 | -0.043546 | True | 0.670867 | 3.583115 | False |  |  |  |
| 2026-04-17 | 2026-04-20 | 2026-04-24 | stockmixer_fast | 300308 | 0.060365 | -0.043546 | True | 0.301170 | 4.831893 | False |  |  |  |
| 2026-04-17 | 2026-04-20 | 2026-04-24 | stockmixer_official | 002460 | -0.043546 | -0.043546 | False | 1.540003 | 7.298980 | False |  |  |  |
| 2026-04-17 | 2026-04-20 | 2026-04-24 | timexer | 600547 | -0.072822 | -0.043546 | False | 0.059851 | 3.155886 | False |  |  |  |
| 2026-04-17 | 2026-04-20 | 2026-04-24 | master_multiseed | 002460 | -0.043546 | -0.043546 | False | 0.443561 | 2.712972 | False | 0.333333 | 3.000000 | 688012;300308;600522 |
| 2026-04-17 | 2026-04-20 | 2026-04-24 | stockmixer_lite_multiseed | 601899 | -0.024096 | -0.043546 | True | 0.434461 | 4.028204 | False | 0.666667 | 2.000000 | 601899;601899;601138 |
| 2026-04-17 | 2026-04-20 | 2026-04-24 | stockmixer_official_multiseed | 600522 | 0.176471 | -0.043546 | True | 0.558510 | 4.088446 | False | 0.333333 | 3.000000 | 002460;000100;603986 |
| 2026-04-17 | 2026-04-20 | 2026-04-24 | timexer_multiseed | 600547 | -0.072822 | -0.043546 | False | 0.538072 | 3.611668 | False | 0.333333 | 3.000000 | 600547;601111;600026 |
| 2026-04-24 | 2026-04-27 | 2026-05-06 | baseline | 000807 | -0.051170 | -0.018873 | False | 0.113481 | 3.470316 | False |  |  |  |
| 2026-04-24 | 2026-04-27 | 2026-05-06 | master | 300394 | -0.037385 | -0.018873 | False | 0.370290 | 3.044185 | False |  |  |  |
| 2026-04-24 | 2026-04-27 | 2026-05-06 | stockmixer_lite | 603799 | 0.012314 | -0.018873 | True | 1.215267 | 6.056011 | False |  |  |  |
| 2026-04-24 | 2026-04-27 | 2026-05-06 | stockmixer_fast | 002460 | 0.075595 | -0.018873 | True | 0.231994 | 2.413324 | False |  |  |  |
| 2026-04-24 | 2026-04-27 | 2026-05-06 | stockmixer_official | 601899 | -0.014706 | -0.018873 | True | 2.079358 | 6.032201 | False |  |  |  |
| 2026-04-24 | 2026-04-27 | 2026-05-06 | timexer | 601899 | -0.014706 | -0.018873 | True | 1.332654 | 5.749697 | False |  |  |  |
| 2026-04-24 | 2026-04-27 | 2026-05-06 | master_multiseed | 300394 | -0.037385 | -0.018873 | False | 0.214225 | 3.200089 | False | 0.666667 | 2.000000 | 300394;300394;300014 |
| 2026-04-24 | 2026-04-27 | 2026-05-06 | stockmixer_lite_multiseed | 601899 | -0.014706 | -0.018873 | True | 0.098147 | 4.748670 | False | 0.333333 | 3.000000 | 603799;601899;000408 |
| 2026-04-24 | 2026-04-27 | 2026-05-06 | stockmixer_official_multiseed | 601899 | -0.014706 | -0.018873 | True | 0.996818 | 5.242138 | False | 0.666667 | 2.000000 | 601899;601899;300014 |
| 2026-04-24 | 2026-04-27 | 2026-05-06 | timexer_multiseed | 601899 | -0.014706 | -0.018873 | True | 2.037576 | 6.372757 | False | 1.000000 | 1.000000 | 601899;601899;601899 |
| 2026-04-30 | 2026-05-06 | 2026-05-12 | baseline | 600150 | 0.004388 | 0.097394 | False | 5.529590 | 9.822043 | False |  |  |  |
| 2026-04-30 | 2026-05-06 | 2026-05-12 | master | 603986 | 0.097394 | 0.097394 | False | 0.161392 | 2.546752 | False |  |  |  |
| 2026-04-30 | 2026-05-06 | 2026-05-12 | stockmixer_lite | 300014 | -0.040110 | 0.097394 | False | 0.284983 | 3.271556 | False |  |  |  |
| 2026-04-30 | 2026-05-06 | 2026-05-12 | stockmixer_fast | 603799 | -0.023127 | 0.097394 | False | 0.126383 | 3.406267 | False |  |  |  |
| 2026-04-30 | 2026-05-06 | 2026-05-12 | stockmixer_official | 603986 | 0.097394 | 0.097394 | False | 0.225493 | 3.204521 | False |  |  |  |
| 2026-04-30 | 2026-05-06 | 2026-05-12 | timexer | 603799 | -0.023127 | 0.097394 | False | 1.371396 | 5.654292 | False |  |  |  |
| 2026-04-30 | 2026-05-06 | 2026-05-12 | master_multiseed | 603986 | 0.097394 | 0.097394 | False | 0.295556 | 2.985209 | False | 0.666667 | 2.000000 | 603986;000977;603986 |
| 2026-04-30 | 2026-05-06 | 2026-05-12 | stockmixer_lite_multiseed | 300014 | -0.040110 | 0.097394 | False | 0.053550 | 2.916097 | False | 0.333333 | 3.000000 | 300014;002460;601899 |
| 2026-04-30 | 2026-05-06 | 2026-05-12 | stockmixer_official_multiseed | 300014 | -0.040110 | 0.097394 | False | 0.074545 | 2.926919 | False | 0.333333 | 3.000000 | 603986;300014;002460 |
| 2026-04-30 | 2026-05-06 | 2026-05-12 | timexer_multiseed | 000630 | 0.129032 | 0.097394 | True | 0.215688 | 4.387158 | False | 0.333333 | 3.000000 | 603799;000630;300394 |
| 2026-05-08 | 2026-05-11 | 2026-05-15 | baseline | 002466 | -0.075101 | -0.061318 | False | 0.451303 | 4.417031 | False |  |  |  |
| 2026-05-08 | 2026-05-11 | 2026-05-15 | master | 002371 | 0.036038 | -0.061318 | True | 1.820609 | 4.566978 | False |  |  |  |
| 2026-05-08 | 2026-05-11 | 2026-05-15 | stockmixer_lite | 000063 | -0.039796 | -0.061318 | True | 0.746550 | 4.693222 | False |  |  |  |
| 2026-05-08 | 2026-05-11 | 2026-05-15 | stockmixer_fast | 300014 | -0.061318 | -0.061318 | False | 0.961817 | 4.818914 | False |  |  |  |
| 2026-05-08 | 2026-05-11 | 2026-05-15 | stockmixer_official | 300014 | -0.061318 | -0.061318 | False | 0.007814 | 4.291916 | False |  |  |  |
| 2026-05-08 | 2026-05-11 | 2026-05-15 | timexer | 300394 | 0.203012 | -0.061318 | True | 0.085995 | 4.160629 | False |  |  |  |
| 2026-05-08 | 2026-05-11 | 2026-05-15 | master_multiseed | 300014 | -0.061318 | -0.061318 | False | 0.932630 | 3.722336 | False | 0.333333 | 3.000000 | 002371;300308;601766 |
| 2026-05-08 | 2026-05-11 | 2026-05-15 | stockmixer_lite_multiseed | 601899 | -0.045152 | -0.061318 | True | 0.197230 | 4.988823 | False | 0.333333 | 3.000000 | 000063;601899;300014 |
| 2026-05-08 | 2026-05-11 | 2026-05-15 | stockmixer_official_multiseed | 300014 | -0.061318 | -0.061318 | False | 0.653900 | 4.534543 | False | 0.333333 | 3.000000 | 300014;002475;000063 |
| 2026-05-08 | 2026-05-11 | 2026-05-15 | timexer_multiseed | 603799 | -0.088838 | -0.061318 | False | 1.980970 | 5.763909 | False | 0.333333 | 3.000000 | 300394;300308;603799 |
| 2026-05-15 | 2026-05-18 | 2026-05-22 | baseline | 600150 | -0.018905 | -0.048987 | True | 2.578548 | 6.145348 | False |  |  |  |
| 2026-05-15 | 2026-05-18 | 2026-05-22 | master | 300308 | -0.019407 | -0.048987 | True | 0.281444 | 2.801675 | False |  |  |  |
| 2026-05-15 | 2026-05-18 | 2026-05-22 | stockmixer_lite | 688223 | -0.071742 | -0.048987 | False | 0.049149 | 5.684632 | False |  |  |  |
| 2026-05-15 | 2026-05-18 | 2026-05-22 | stockmixer_fast | 603993 | -0.019284 | -0.048987 | True | 0.722937 | 5.505052 | False |  |  |  |
| 2026-05-15 | 2026-05-18 | 2026-05-22 | stockmixer_official | 300408 | 0.234305 | -0.048987 | True | 0.089604 | 5.624398 | False |  |  |  |
| 2026-05-15 | 2026-05-18 | 2026-05-22 | timexer | 002384 | 0.037226 | -0.048987 | True | 0.618823 | 4.550067 | False |  |  |  |
| 2026-05-15 | 2026-05-18 | 2026-05-22 | master_multiseed | 002475 | -0.048987 | -0.048987 | False | 0.261319 | 3.210437 | False | 0.666667 | 2.000000 | 300308;002475;002475 |
| 2026-05-15 | 2026-05-18 | 2026-05-22 | stockmixer_lite_multiseed | 688256 | 0.083750 | -0.048987 | True | 0.938181 | 5.244008 | False | 0.333333 | 3.000000 | 688223;603296;300274 |
| 2026-05-15 | 2026-05-18 | 2026-05-22 | stockmixer_official_multiseed | 688256 | 0.083750 | -0.048987 | True | 0.032816 | 4.475071 | False | 0.333333 | 3.000000 | 300408;603986;688256 |
| 2026-05-15 | 2026-05-18 | 2026-05-22 | timexer_multiseed | 601600 | 0.036792 | -0.048987 | True | 0.721301 | 4.678626 | False | 0.666667 | 2.000000 | 002384;601600;601600 |

## Dynamic Switch Rows

| date | selected | top1 | return | fallback_return | hit_vs_fallback | candidate_top1_agreement | candidate_top1_agreement_share | best_margin_z | best_top_strength | allin_selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-17 | stockmixer_portfolio_lite_rank | 002050 | -0.021179 | -0.043546 | True | 2 | 0.500000 | 0.124234 | 2.607396 | True |
| 2026-04-24 | stockmixer_portfolio_lite_rank | 601899 | -0.014706 | -0.018873 | True | 4 | 1.000000 | 0.048261 | 2.221666 | True |
| 2026-04-30 | stockmixer_portfolio_lite_rank | 300014 | -0.040110 | 0.097394 | False | 3 | 0.750000 | 0.021832 | 2.175908 | True |
| 2026-05-08 | primary | 300014 | -0.061318 | -0.061318 | False | 1 | 0.250000 | 0.026768 | 2.100824 | False |
| 2026-05-15 | primary | 002475 | -0.048987 | -0.048987 | False | 2 | 0.500000 | 0.068257 | 2.052206 | False |

## Interpretation

- `return` is all-in Top1 realized open-to-open return for the requested T window.
- `hit_vs_fallback` checks whether all-in Top1 beat the MASTER dynamic-risk-budget fallback on the same T.
- `candidate_top1_agreement_share` is the dynamic-switch cross-candidate consistency signal.
- Multi-seed rows are confirmation diagnostics; they should not be used as a standalone all-in trigger unless the matrix keeps showing positive p05 and low negative rate.
