# Current Status

## LatestA Final Submission - 2026-05-30

The current files under `app/` are now the online A-stage latestA submission using data through 2026-05-29:

```csv
stock_id,weight
300308,0.5721812227310785
002384,0.4278187772689215
```

Decision:

- Hard all-in gate: rejected all-in.
- Portfolio selector: selected `fusion_top2`.
- Fusion recipe: `75% MASTER single-seed + 25% StockMixer official multi-seed`, z-score transform, `top2_softmax`.
- Five-window replay for the selected candidate: mean `3.6266%`, p05 `1.0165%`, negative rate `0.0%`.

Main record: `docs/final_submission_latestA_2026_05_30.md`.

## 当前阶段

项目已经完成从数据、模型、评估到 Docker 提交目录的闭环。正式提交后继续做了行业特征、组合优化、关系增强、TimeXer 和多模型融合实验。

当前最新推荐结果已同步到：

```text
app/model/result.csv
app/output/result.csv
```

## 当前最佳方案

当前最佳为：

```text
MASTER official-rank prediction
+ date z-score
+ industry-rank relation adjustment
+ softmax_t0.6 weighting
```

输出：

```csv
stock_id,weight
002422,0.5366080941073169
688981,0.18532079156062642
300433,0.14533404575723335
002049,0.06974993365670436
688008,0.06298713491811891
```

## 结果对比

| 候选 | 策略 | 验证均值 | 波动 | 判断 |
|---|---|---:|---:|---|
| 原提交 MASTER | `proportional_positive_thr0.0` | `0.01904` | `0.03941` | 已提交版本 |
| MASTER z-score + softmax + industry cap 3 | `softmax_t0.6` | `0.02146` | `0.04904` | 明显提升 |
| MASTER relation best | `softmax_t0.6` | `0.02177` | `0.05253` | 当前最佳 |

稳定性窗口：

| 候选 | 20日 | 40日 | 60日 | 90日 |
|---|---:|---:|---:|---:|
| 原提交 MASTER | `0.02431` | `0.00617` | `0.01309` | `0.01921` |
| MASTER z-score + softmax | `0.03002` | `0.00733` | `0.01680` | `0.02174` |
| MASTER relation best | `0.03275` | `0.00841` | `0.01821` | `0.02214` |

## 已完成实验

### 行业特征

- 行业映射已补强，HS300 中 `other` 从 146 只降到 3 只。
- 行业特征直接进入 StockMixer fast 后为负增益。
- 行业约束对 StockMixer 有帮助，但对 MASTER 不稳定。
- 最终保留行业关系后处理，而不是硬行业 cap。

### 关系增强

新增：

```text
scripts/relation_postprocess.py
```

当前最佳参数：

```text
mode = rank_ind_mix
alpha = -0.5
strategy = softmax_t0.6
```

### 组合搜索

新增：

```text
scripts/search_portfolio_candidates.py
scripts/postprocess_predictions.py
```

结论：多模型融合没有超过 MASTER 单模型关系增强。

### TimeXer

新增：

```text
src/models/timexer.py
scripts/train_timexer_backbone.py
configs/timexer_alpha_fast.yaml
configs/timexer_alpha_fast_v2.yaml
```

结论：TimeXer fast / v2 目前 RankIC 有信号，但 top-k 组合收益弱，暂不进主方案。

### 近期窗口训练

新增：

```text
configs/master_alpha_recent_fast.yaml
```

结论：近期窗口 MASTER RankIC 尚可，但 top-k 组合收益差，不进入当前主候选。

## 当前关键文件

```text
docs/experiment_report_2026_04_27.md
scripts/relation_postprocess.py
scripts/search_portfolio_candidates.py
src/features/industry_context.py
src/models/timexer.py
app/model/result.csv
app/output/result.csv
```

## 推荐下一步

1. 将 relation postprocess 正式接入 `app/code/train.py` 的训练后处理流程。
2. 做多 seed MASTER，并对 seed 平均分数再做 relation postprocess。
3. 把关系增强从行业扩展到行业 + 风格桶 + beta/流动性邻居。
4. 若继续 TimeXer，需要改成 top-k/listwise 目标，而不是只优化 RankIC。
5. 外源数据只在确认报备流程后引入。
