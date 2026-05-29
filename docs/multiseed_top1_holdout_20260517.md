# Multi-Seed Top1 Holdout Replay 2026-05-17

Experiment target:

- T: 2026-05-15
- Buy: 2026-05-18 open
- Sell: 2026-05-22 open
- Seeds: 42, 52, 62
- Portfolio rule under inspection: all-in Top1

## Leaderboard

| Model | Mean Top1 | Return | Vote Top1 | Vote return | Vote share | Unique Top1 |
|---|---:|---:|---:|---:|---:|---:|
| stockmixer_lite | 688256 | 0.083750 | 688256 | 0.083750 | 0.667 | 2 |
| lstm | 600958 | 0.010471 | 688981 | 0.101618 | 0.333 | 3 |
| itransformer | 002415 | -0.018675 | 002415 | -0.018675 | 0.333 | 3 |
| timexer_fast | 002460 | -0.034589 | 300750 | -0.013981 | 0.333 | 3 |
| master_official | 600276 | -0.041465 | 002460 | -0.034589 | 0.333 | 3 |
| stockmixer_official | 300442 | -0.105785 | 300442 | -0.105785 | 0.667 | 2 |

## Notes

Multi-seed averaging did not reproduce the earlier single-seed/blend all-in `688981` result. The strongest 3-seed average candidate is `stockmixer_lite -> 688256`, with `0.083750` return. This is lower than `688981` all-in (`0.101618`) but still strong and has better seed agreement: seed 42 and seed 62 both select `688256`.

MASTER remains weak on this particular short window. Its three seed Top1 choices are all different (`002460`, `002384`, `600089`), and the mean Top1 shifts to `600276`, which loses `-0.041465`.

The result supports using multi-seed consistency as an all-in filter, but not naive averaging alone. A useful production gate should consider:

- seed Top1 vote share;
- whether the mean Top1 is also selected by at least one seed;
- mean Top1 margin/strength;
- cross-model agreement with StockMixer/ensemble candidates.

Artifacts:

- `outputs/holdout_20260517/multiseed_top1/leaderboard.csv`
- `outputs/holdout_20260517/multiseed_top1/summary.json`
- `scripts/run_multiseed_top1_holdout.py`
