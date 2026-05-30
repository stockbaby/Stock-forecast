# Final submission decider

## Decision

- decision: `top3_defensive`
- selected_submission: `master_top3`
- selected_stock: `300308`
- gated_candidate_stock: `002384`
- allin_allowed: `False`
- reason: 存在共识，但 fallback/官方防守信号不够配合，降为 Top3 分散。

## Gate facts

- independent stock agreement families: `1` on `300308`
- independent industry agreement families: `2` on `electronics`
- official completely different: `True`
- candidate in official Top5: `False`
- fallback ok: `False`
- max core top_strength / margin_z: `6.3779` / `0.4215`

## Candidate diagnostic table

| source                        | family            |   top1 | top1_industry   | top5                               | margin_z   | top_strength      | seed_consistency   |   same_top1_models |   top5_overlap_models | volume_price_confirmation           |
|:------------------------------|:------------------|-------:|:----------------|:-----------------------------------|:-----------|:------------------|:-------------------|-------------------:|----------------------:|:------------------------------------|
| master                        | master            | 300308 | telecom         | 300308;002050;600160;002466;300014 | 0.2462     | 2.218             | na                 |                  2 |                     8 | mixed; amount5/20=1.11; ret5=11.87% |
| master_multiseed              | master            | 300308 | telecom         | 300308;002460;002384;300014;600111 | 0.8531     | 3.9784            | 0.67 (2 unique)    |                  2 |                    11 | mixed; amount5/20=1.11; ret5=11.87% |
| stockmixer_official           | stockmixer        | 002384 | electronics     | 002384;600522;002371;000063;300308 | 0.1139     | 6.3779            | 1.00 (1 unique)    |                  1 |                    11 | mixed; amount5/20=0.95; ret5=-3.40% |
| stockmixer_official_multiseed | stockmixer        | 600522 | telecom         | 600522;000630;002384;002371;688008 | 2.2644     | 7.546             | 0.33 (3 unique)    |                  1 |                     9 | mixed; amount5/20=0.93; ret5=-3.72% |
| timexer                       | timexer           | 603296 | electronics     | 603296;603893;600460;688472;002049 | 0.4215     | 4.1866            | na                 |                  2 |                     7 | mixed; amount5/20=0.87; ret5=-2.33% |
| timexer_multiseed             | timexer           | 603296 | electronics     | 603296;002460;002475;688472;601138 | 0.1097     | 3.0799            | 0.33 (3 unique)    |                  2 |                     8 | mixed; amount5/20=0.87; ret5=-2.33% |
| lightgbm_local_baseline       | baseline          | 002625 | military        | 002625;001979;000002;600482;000975 | 2.5348     | 8.7596            | na                 |                  1 |                     5 | strong; amount5/20=1.53; ret5=5.18% |
| official_baseline_top5        | official_baseline | 002493 | oil_gas         | 002493;300347;600482;300251;600346 | na         | defensive         | na                 |                  0 |                     1 | weak; amount5/20=0.66; ret5=-1.66%  |
| relation_postprocess_best     | relation          | 300308 | telecom         | 300308;002460;600111;002384;300014 | na         | last_holdout=nan% | na                 |                  2 |                    11 | mixed; amount5/20=1.11; ret5=11.87% |
| dynamic_switch                | switch            | 300308 | telecom         | 300308                             | na         | selected=primary  | na                 |                  2 |                     3 | mixed; amount5/20=1.11; ret5=11.87% |

## Five-window reference

| model                         |   n |   hit_rate |   mean_return |   p05_return |   negative_rate |   max_drawdown |   mean_excess_vs_fallback |
|:------------------------------|----:|-----------:|--------------:|-------------:|----------------:|---------------:|--------------------------:|
| stockmixer_official           |   5 |        0.4 |    0.0424258  |   -0.0577637 |             0.6 |     -0.0613182 |                0.057492   |
| stockmixer_official_multiseed |   5 |        0.4 |    0.0288174  |   -0.0570764 |             0.6 |     -0.112219  |                0.0438835  |
| timexer                       |   5 |        0.4 |    0.0259168  |   -0.0628828 |             0.6 |     -0.0374924 |                0.040983   |
| master                        |   5 |        0.6 |    0.0201216  |   -0.0337891 |             0.4 |     -0.0373846 |                0.0351877  |
| official_baseline_top1        |   5 |        0.8 |    0.0161507  |   -0.0838628 |             0.2 |     -0.105466  |                0.0312168  |
| stockmixer_fast               |   5 |        0.4 |    0.00644638 |   -0.0536798 |             0.6 |     -0.100709  |                0.0215125  |
| official_baseline_top5_equal  |   5 |        0.4 |    0.00629494 |   -0.0231507 |             0.6 |     -0.0409807 |                0.0213611  |
| timexer_multiseed             |   5 |        0.4 |   -0.00210826 |   -0.085635  |             0.6 |     -0.0888383 |                0.0129579  |
| lightgbm_local_baseline       |   5 |        0.4 |   -0.00277154 |   -0.0703146 |             0.6 |     -0.13524   |                0.0122946  |
| stockmixer_lite_multiseed     |   5 |        0.2 |   -0.00806275 |   -0.0441435 |             0.8 |     -0.0969291 |                0.00700338 |
| master_multiseed              |   5 |        0.2 |   -0.0187684  |   -0.058852  |             0.8 |     -0.107302  |               -0.0037023  |
| stockmixer_lite               |   5 |        0.2 |   -0.0326861  |   -0.0654158 |             0.8 |     -0.144433  |               -0.0176199  |
