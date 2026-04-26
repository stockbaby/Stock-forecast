# Official Baseline Strict Same-Window Comparison

## Scope

This document records the strict same-window comparison between:

- the official baseline repository: `external/THU-BDC2026`
- our current submissions in this repository

All methods use the same local price file and the same evaluation window.

## Reproduction Setup

- Official repo: `external/THU-BDC2026`
- Official training script: `external/THU-BDC2026/code/src/train.py`
- Official prediction script: `external/THU-BDC2026/code/src/predict.py`
- Official prediction output: `external/THU-BDC2026/output/result.csv`
- Shared price data: `data/raw/stock_data.csv`
- Evaluation script: `scripts/evaluate_submission_return.py`

Reference window:

- trade date `T`: `2026-04-17`
- buy date `T+1`: `2026-04-20`
- sell date: `2026-04-24`

## Official Baseline Result

Official baseline local return:

- `0.023500505822503896`

Official baseline selected stocks:

- `002493`
- `600482`
- `000301`
- `600489`
- `600346`

Official baseline weights:

- all equal weight `0.2`

## Same-Window Comparison

| Method | Submission File | Local Return | Improvement vs Official |
|---|---|---:|---:|
| Official baseline | `external/THU-BDC2026/output/result.csv` | 0.02350 | 0.00000 |
| LightGBM | `outputs/submissions/result_a_stage_round1.csv` | 0.03386 | +0.01036 |
| MASTER | `outputs/submissions/result_master_alpha.csv` | 0.06097 | +0.03747 |
| StockMixer | `outputs/submissions/result_stockmixer_alpha.csv` | 0.06332 | +0.03982 |

## Interpretation

- All three of our main submissions beat the official baseline under the same data, same date window, and same return formula.
- `LightGBM` is already above the official baseline, but the margin is modest.
- `MASTER` and the strengthened `StockMixer` are both clearly stronger than the official baseline.
- On this recent local window, `StockMixer` remains the best current submission.
- On validation-set aggregate metrics, `MASTER` remains the strongest overall model.

## Notes

- The official baseline repository required Windows compatibility fallbacks for `multiprocessing.Pool` in `train.py` and `predict.py`.
- The compatibility fallback does not change the training target, prediction logic, or portfolio construction rule; it only replaces failed multi-process feature engineering with a single-process path.
