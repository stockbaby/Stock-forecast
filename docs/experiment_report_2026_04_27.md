# 2026-04-27 实验与提交后优化报告

## 摘要

本轮工作围绕提交后的继续提升展开，重点覆盖：

- 行业特征与行业约束
- 置信度过滤 / 留现金 / softmax 配权
- 关系增强（HIST 思路的行业 peer relation）
- 多模型融合搜索
- 近期窗口训练
- TimeXer-style 外生变量时序模型
- `app/` 当前结果更新与可复现脚本整理

当前最佳候选为：

```text
MASTER official-rank prediction
+ date z-score
+ industry-rank relation adjustment
+ softmax_t0.6 portfolio weighting
```

当前候选文件：

```text
app/model/result.csv
app/output/result.csv
outputs/submissions/result_master_relation_best.csv
```

当前候选内容：

```csv
stock_id,weight
002422,0.5366080941073169
688981,0.18532079156062642
300433,0.14533404575723335
002049,0.06974993365670436
688008,0.06298713491811891
```

## 关键结果对比

### 主候选对比

| 候选 | 策略 | 行业约束 | 91日均值 | 91日波动 | 备注 |
|---|---:|---:|---:|---:|---|
| 已提交 MASTER | `proportional_positive_thr0.0` | 无 | `0.01904` | `0.03941` | 原提交版 |
| MASTER z-score softmax | `softmax_t0.6` | `max_per_industry=3` | `0.02146` | `0.04904` | 分数截面标准化 + 更集中配权 |
| MASTER relation best | `softmax_t0.6` | 无硬 cap | `0.02177` | `0.05253` | 当前最佳 |

### 稳定性窗口

| 候选 | 20日 | 40日 | 60日 | 90日 |
|---|---:|---:|---:|---:|
| 已提交 MASTER | `0.02431` | `0.00617` | `0.01309` | `0.01921` |
| MASTER z-score softmax industry3 | `0.03002` | `0.00733` | `0.01680` | `0.02174` |
| MASTER relation best | `0.03275` | `0.00841` | `0.01821` | `0.02214` |

结论：关系增强候选在 20/40/60/90 日窗口均优于已提交版和纯 z-score softmax 版，但波动更大。

## 行业特征与行业约束

### 已完成改动

文件：

```text
src/features/industry_context.py
src/training/dataset_builder.py
scripts/train_stockmixer_baseline.py
configs/stockmixer_alpha_industry_fast.yaml
configs/stockmixer_alpha_fast.yaml
```

主要变化：

- 新增行业映射和行业 one-hot / 行业相对强弱特征。
- 对 HS300 股票简称增加显式行业 override，`other` 数量从 146 只降到 3 只。
- 深度训练脚本限定只使用数值特征，避免字符串行业名进入模型。
- 为大数据集增加 `reuse_processed` 和 `recent_train_days`，避免重复构建 5GB processed 文件。

### 行业特征进模型实验

同规格 fast 对照：

| 模型 | 特征数 | 训练样本 | 验证样本 | 最优策略 | 均值 |
|---|---:|---:|---:|---|---:|
| StockMixer fast 无行业 | 298 | 101898 | 21300 | `proportional_positive_thr0.0` | `0.01135` |
| StockMixer fast 行业特征 | 421 | 101898 | 21300 | `equal_weight` | `0.01053` |

结论：行业特征直接喂入 StockMixer 当前为负增益，暂不作为主线。

### 行业约束后处理

对完整 StockMixer：

- 无行业约束：`0.01469`
- `max_per_industry=1`：`0.01571`

对完整 MASTER：

- 无行业约束：`0.01904`
- `max_per_industry=1`：`0.01385`
- `max_per_industry=3`：`0.01893`

结论：行业分散对 StockMixer 有帮助，但对 MASTER 不稳定。最终最佳不是硬行业 cap，而是关系增强式软调整。

## 关系增强（HIST 思路）

文件：

```text
scripts/relation_postprocess.py
```

方法：

1. 对 MASTER 预测分数按交易日做 z-score。
2. 基于行业分组计算行业内 rank / 行业均值。
3. 搜索以下 relation score：

```text
smooth:       (1 - alpha) * score_z + alpha * industry_mean
neutral:      score_z - alpha * industry_mean
rank_mix:     (1 - alpha) * score_z + alpha * global_rank
rank_ind_mix: (1 - alpha) * score_z + alpha * industry_rank
```

当前最佳：

```text
mode = rank_ind_mix
alpha = -0.5
strategy = softmax_t0.6
max_per_industry = None
```

等价于：

```text
score_relation = 1.5 * score_z - 0.5 * industry_rank
```

解释：当前验证窗口里，单纯追行业内高 rank 有拥挤风险；对行业内 rank 做负向调整后，能提高头部组合收益。

复现命令：

```powershell
D:\anaconda\envs\stock-forecast\python.exe scripts\relation_postprocess.py --prediction-path outputs/predictions/master_alpha_official_rank_predictions.csv --output-path outputs/submissions/result_master_relation_best.csv --metrics-path outputs/predictions/master_relation_best_metrics.json --mode rank_ind_mix --alpha -0.5 --strategy softmax_t0.6
```

## 组合优化与融合搜索

文件：

```text
scripts/postprocess_predictions.py
scripts/search_portfolio_candidates.py
```

搜索范围：

- score transform: rank / zscore
- strategies: proportional positive / equal weight / softmax
- industry cap: None / 2 / 3
- models: MASTER official, StockMixer official, iTransformer, existing ensemble
- ensemble grid step: 0.25

核心搜索结果：

| 排名 | 候选 | transform | strategy | cap | 均值 |
|---:|---|---|---|---:|---:|
| 1 | MASTER single | zscore | `softmax_t0.6` | 3 | `0.02146` |
| 2 | MASTER single | zscore | `softmax_t0.6` | None | `0.02142` |
| 3 | MASTER single | zscore | `softmax_t0.6` | 2 | `0.02132` |
| 4 | 0.75 MASTER + 0.25 iTransformer | zscore | `softmax_t0.6` | None | `0.02005` |
| 5 | 0.75 MASTER + 0.25 iTransformer | zscore | `softmax_t0.6` | 3 | `0.02003` |

结论：融合没有超过 MASTER 单模型后处理；当前最优来自 MASTER 自身的分数变换和关系增强。

## TimeXer-style 实验

文件：

```text
src/models/timexer.py
scripts/train_timexer_backbone.py
configs/timexer_alpha_fast.yaml
configs/timexer_alpha_fast_v2.yaml
```

实现要点：

- 使用时间 patch token 表达股票自身历史序列。
- 使用市场 / 指数 / beta / style 等外生变量构建 exogenous token。
- 用 cross-attention 将外生 token 注入 patch token。
- 再经 Transformer encoder 输出回归分数。

结果：

| 模型 | 训练样本 | 验证样本 | RankIC | Precision@5 | 组合均值 |
|---|---:|---:|---:|---:|---:|
| TimeXer fast | 101898 | 21300 | `0.01583` | `0.05634` | `0.00571` |
| TimeXer fast v2 | 152538 | 18300 | `0.07611` | `0.00984` | `0.00451` |

结论：TimeXer v2 的 RankIC 高，但 top-k 组合收益弱，说明它学到了部分截面秩信息，但头部选股不适配当前目标。后处理和反向检查未改善，暂不进入主候选。

## 近期窗口训练

文件：

```text
configs/master_alpha_recent_fast.yaml
scripts/train_master_baseline.py
```

结果：

| 模型 | 训练样本 | 验证样本 | RankIC | Precision@5 | 组合均值 |
|---|---:|---:|---:|---:|---:|
| MASTER recent fast | 101898 | 21300 | `0.06106` | `0.03380` | `0.00517` |

结论：近期训练 RankIC 尚可，但 top-k 组合收益差；加入 MASTER 主模型融合也没有提升。

## 外源数据与数据增强判断

本轮没有引入新的外源数据集，原因：

- 赛事要求外部数据和模型需提前报备链接和 md5。
- 当前离线复现和提交风险优先级高于新增外源数据。
- 已有市场指数、行业关系、风格/相对强弱特征的边际收益更清晰。

建议下一步如继续外源路线，优先使用可报备、可离线固化的小体积数据：

- 指数/行业指数历史行情
- 中证/申万行业分类表
- 沪深 300 权重和成分调整历史
- 股票基本面静态分类字段

## 当前推荐

如果允许二次打包，推荐使用：

```text
app/model/result.csv
app/output/result.csv
```

这两个文件已同步为当前最佳关系增强候选。

如需复现当前最佳：

```powershell
D:\anaconda\envs\stock-forecast\python.exe scripts\relation_postprocess.py --prediction-path outputs/predictions/master_alpha_official_rank_predictions.csv --output-path outputs/submissions/result_master_relation_best.csv --metrics-path outputs/predictions/master_relation_best_metrics.json --mode rank_ind_mix --alpha -0.5 --strategy softmax_t0.6
Copy-Item outputs/submissions/result_master_relation_best.csv app/model/result.csv -Force
Copy-Item outputs/submissions/result_master_relation_best.csv app/output/result.csv -Force
D:\anaconda\envs\stock-forecast\python.exe app/code/test.py
```

## 后续优先级

1. 将 relation postprocess 接入 `app/code/train.py`，让完整训练后自动产出当前最佳关系增强结果。
2. 尝试更细粒度关系图：行业 + 同风格桶 + beta/流动性邻居，而不是只用行业 rank。
3. 对 MASTER 做多 seed 训练，再对 seed prediction 做 relation postprocess。
4. 若要继续 TimeXer，需改成直接优化 top-k/listwise 目标，而不是只看 RankIC。
5. 外源数据只建议在确认报备流程后引入。
