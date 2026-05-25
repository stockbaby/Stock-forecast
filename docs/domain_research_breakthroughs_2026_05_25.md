# 领域调研与突破点建议 - 2026-05-25

## 1. 项目约束再定义

本项目不是普通的股票收益率回归问题，而是一个更尖锐的截面决策问题：

```text
在预测日 T，只使用 T 及以前可得数据，选择不超过 5 只沪深300成分股并分配权重，
最大化 open(T+1) -> open(T+5) 的组合收益。
```

因此，`MSE`、`IC`、`RankIC` 只能作为辅助指标。真正值得优化的是：

- 头部股票是否命中；
- Top1/Top2/Top5 的风险收益权衡；
- 分数是否能稳定转化为权重；
- 新市场窗口下是否出现 regime shift；
- 是否避免未来信息、日期错位和验证集选择偏差。

本地实验已经证明了这一点：`master_official + top2_softmax` 在 91 日验证上均值约 `0.02145`、波动约 `0.04913`；`top1_weight` 的 A 阶段回放更高，但 91 日波动约 `0.06522`、负收益率更高。`master_multiseed + top2_softmax` 的历史均值更高，约 `0.02450`，但当前还缺少真实最新 T 日推理链路。

## 2. 外部研究脉络

### 2.1 股票专用模型仍比通用时序模型更贴近赛题

`MASTER` 明确面向股票预测，核心是用市场状态引导特征选择，并交替建模个股时间信息和股票间关联。它的论文目标与 CSI300/CSI800 场景贴近，适合继续作为主干模型，而不是被通用 Transformer 替代。

`HIST` 的价值在于关系建模：它认为股票间共享信息不仅来自预定义概念，也来自隐藏概念，且股票与概念的相关性会动态变化。这与本项目 relation postprocess 的有效性高度一致。

`StockMixer` 用 MLP 完成 indicator/time/stock mixing，论文强调简单结构在有限股票数据下更易优化。它适合作为强备选和集成成员，但本地结果显示目前还没有超过 MASTER relation 主线。

参考：

- MASTER: Market-Guided Stock Transformer for Stock Price Forecasting, AAAI 2024: https://arxiv.org/abs/2312.15235
- HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information: https://ideas.repec.org/p/arx/papers/2110.13716.html
- StockMixer: A Simple yet Strong MLP-Based Architecture for Stock Price Forecasting, AAAI 2024: https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/

### 2.2 排序目标比直接收益回归更值得继续打磨

RSR/Temporal Relational Ranking 把股票预测直接写成 learning-to-rank 问题，并结合股票关系做排序。后续的 stock selection hypergraph 工作也沿着“股票关系 + 学习排序”推进。这说明本项目此前尝试 top-k/listwise 是方向正确，但实现上要更抗噪声。

本地证据也支持这一判断：直接 `portfolio_return_loss` 过拟合明显，`master_portfolio` 各策略均明显低于 `master_official`。这不是“组合目标不对”，而是直接最大化短窗口收益太噪，需要更稳健的 surrogate。

参考：

- Temporal Relational Ranking for Stock Prediction: https://arxiv.org/abs/1809.09441
- Stock Selection via Spatiotemporal Hypergraph Attention Network: https://ojs.aaai.org/index.php/AAAI/article/view/16127
- Differentiable Ranking and Sorting using Optimal Transport: https://papers.nips.cc/paper/8910-differentiable-ranking-and-sorting-using-optimal-transport

### 2.3 通用时序模型适合借思想，不适合裸迁移

`iTransformer` 将变量作为 token，适合多变量相关性建模；但它不天然解决股票池、截面排名、权重构造和关系图问题。

`TimeXer` 专门处理外生变量，适合未来引入指数、行业、资金流、宏观、市场情绪等额外信息时参考。但本地 TimeXer fast/v2 已显示 RankIC 和 Top-K 收益可能脱钩，所以不能把它作为主线替换 MASTER。

参考：

- iTransformer: Inverted Transformers Are Effective for Time Series Forecasting: https://arxiv.org/abs/2310.06625
- TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables: https://arxiv.org/abs/2402.19072

### 2.4 市场漂移是下一阶段必须正面处理的问题

`DoubleAdapt` 聚焦金融时间序列中的分布漂移，用数据适配器和模型适配器处理增量学习中的 concept drift。对本项目来说，完整迁移 DoubleAdapt 成本偏高，但可以先吸收它的思想：让模型/组合策略根据近期市场状态选择不同候选，而不是固定一个权重规则。

参考：

- DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting: https://arxiv.org/abs/2306.09862

## 3. 可能突破点排序

### P0. 真实最新 T 日推理与验证闭环

优先级最高。此前最大问题是日期错位，说明任何新模型收益都可能被工程错位吞掉。

建议：

- 给 `master_multiseed` 补齐真实最新 T 日推理；
- 所有候选统一输出 latest predictions，而不是从验证集最后一天取结果；
- 在提交生成前强制检查 `latest_date == unlabeled T`；
- 把 `test.py` 或提交脚本中的结果日期校验写死。

预期收益：不是模型层面创新，但能立刻释放 `master_multiseed + top2_softmax` 的潜在候选价值。

### P1. 稳健 listwise/top-k surrogate

现有直接 portfolio-return loss 过拟合，下一步应避免直接把每日收益作为唯一梯度信号。

建议实验：

- `top_label_weighted_listnet`: 对同日股票按未来收益分桶或 rank label，头部给更高权重，标签裁剪极端值；
- `pairwise_margin_head_loss`: 只比较 top quantile 与 bottom/median，减少全截面噪声；
- `soft_ndcg_at_k`: 用可微排序近似优化 top-k 排序质量；
- `hybrid_loss = regression/rank + small_weight * topk_surrogate`，不要再让组合收益项占主导；
- 所有 loss 必须按日期 batch 计算，避免不同日期截面混在一起。

验证标准：

- 不只看 RankIC；
- 必须看 20/40/60/90 日窗口收益、负收益率、p05、latest A score；
- 如果 RankIC 提升但 top-k 收益下降，直接降级。

### P2. Relation 3.0：从线性后处理升级到可学习关系评分

relation 2.0/2.1/2.2 已证明行业、beta、波动、流动性、历史相关邻居和 regime risk 有稳定增益。下一阶段突破点不是再手调更多 alpha，而是把这些关系信号变成轻量可学习层。

建议路线：

- 输入：`score_z`、行业内 rank、beta rank、vol rank、liquidity rank、corr_peer_score、regime_risk、近期 momentum/breakout；
- 模型：LightGBM/LambdaMART 或一个很小的 MLP，按日期截面训练；
- 输出：relation-adjusted score；
- 训练目标：top-k surrogate 或同日收益分桶；
- 约束：使用 rolling/walk-forward，不能在全验证区间上一次性拟合。

为什么值得做：当前 relation 是手工线性组合，已经有收益；可学习关系层有机会学习“什么时候该信行业，什么时候该信动量/流动性/相关邻居”。

### P3. 候选切换与动态风险预算

本地结果已经出现典型冲突：`top1_weight` 最新窗口很强但历史波动高，`top2_softmax` 更稳，`confidence_topk` 介于两者之间。突破点可能不是找一个永远最强策略，而是做 regime-aware strategy selector。

建议：

- 先只在组合层做，不碰模型主干；
- 用最近 20/40/60 日的市场状态、候选分数间距、行业集中度、波动率、成交量突破、指数动量判断策略；
- 输出三档：`top1_weight`、`top2_softmax`、`confidence_topk`；
- 使用 walk-forward 只在过去窗口选择策略，再用于下一天，避免 hindsight。

评价：

- 平均收益要接近 top1；
- p05、负收益率、最大回撤要接近 top2；
- latest A 不应被过度牺牲。

### P4. 多 seed 不是简单平均，而是选择性集成

relation v2 报告显示 naive multi-seed average 稀释了强 seed 42 信号；但 `method_validation` 又显示 `master_multiseed + top2_softmax` 可能有更高历史均值。这说明多 seed 有信息，但不能简单平均。

建议：

- 计算 seed 间一致性：top-k overlap、rank correlation、top1 margin；
- 当强 seed 与其他 seed 一致时增强仓位；
- 当 seed 分歧大时降低 top1 集中度或切到 top2；
- 只选择过去 walk-forward 表现稳定的 seed 子集，而不是固定全 seed 平均。

这条线和 P3 可以合并：seed disagreement 是动态风险预算的重要输入。

### P5. 外生变量先做轻量特征，不急着上 TimeXer

TimeXer 的思想有价值，但当前项目缺少高质量、合规、可稳定复现的外生变量体系。直接上模型容易产生工程成本和过拟合。

建议先做：

- 沪深300指数/中证全指/创业板指数的短期动量、波动、成交额；
- 行业指数或行业等权组合的 5/10/20 日动量；
- 主题拥挤度：同一行业或同一动量簇内涨幅扩散程度；
- 北向资金、融资融券等数据只有在确认竞赛允许和可复现后再接入。

如果轻量外生特征在 relation layer 中有效，再考虑 TimeXer-style endogenous/exogenous 分离建模。

## 4. 建议的一周执行矩阵

| 顺序 | 任务 | 文件/模块 | 成功标准 |
|---:|---|---|---|
| 1 | 补齐 `master_multiseed` 最新 T 日推理 | `scripts/run_master_multiseed.py`, 训练脚本 | 生成真实 unlabeled T 的提交 |
| 2 | 给提交链路加日期校验 | `app/code/test.py` 或提交生成脚本 | 日期错位时直接失败 |
| 3 | 实现 listwise/top-label surrogate v1 | `src/training/metrics.py`, MASTER config | 91 日 top-k 收益不低于 official |
| 4 | 做 relation 3.0 轻量学习器 | 新脚本或 `scripts/search_relation_v3.py` | walk-forward 不低于 relation 2.2 |
| 5 | 做动态策略选择器 | `src/portfolio/construct.py` 或新脚本 | p05 改善且 mean 不明显下降 |
| 6 | 多 seed 一致性过滤 | multi-seed 输出分析脚本 | 避免 naive average 稀释强信号 |
| 7 | 外生轻量特征验证 | `src/features/market_context.py` | relation layer 中产生稳定增益 |

## 5. 不建议优先投入的方向

- 直接堆更大的 Transformer：任务瓶颈更像头部排序与组合决策，不是 backbone 容量不足。
- 继续直接最大化 portfolio-return loss：已有结果显示过拟合，需要换更稳健 surrogate。
- 完整复现 DoubleAdapt/TimeXer 作为主线：思想值得吸收，但现阶段工程成本高，且不一定对 Top-5 组合收益直接有效。
- 盲目外部数据扩展：若竞赛合规、复现链路、日期对齐无法保证，风险高于收益。

## 6. 结论

最可能的突破点不是“换一个全新的 SOTA 模型”，而是围绕当前强主线做四件事：

1. 让 `MASTER/multiseed/relation` 的真实最新推理链路绝对可靠；
2. 用稳健 top-k/listwise surrogate 替代噪声很大的直接收益最大化；
3. 把 relation 2.2 从手调线性组合升级为 walk-forward 可学习关系层；
4. 在组合层做 regime-aware 策略切换和 seed-disagreement 风险预算。

如果只能选一条主攻线，我建议先做：

```text
master_multiseed latest inference
+ relation 3.0 lightweight learner
+ dynamic top1/top2/confidence strategy selector
```

这条路线最贴合已有有效证据，工程风险最低，也最可能在短期内带来可验证提升。
