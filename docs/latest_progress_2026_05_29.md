# 最新进展 2026-05-29

## 当前判断

本轮工作的核心已经从“固定相信哪个单模型”，转向“判断当天哪个 Top1 信号值得 all-in”。

当前主候选路线是：

```text
动态候选切换
+ StockMixer/ensemble 挑战者 gate
+ 候选 Top1 一致性
+ 多 seed Top1 诊断
+ gate 通过时允许 all-in
```

2026-05-17 holdout 回放窗口：

- T：2026-05-15
- 买入：2026-05-18 开盘
- 卖出：2026-05-22 开盘
- 候选提交：`688981,1.0`
- 实测收益：`0.101618`

关键结论不是“永远 all-in”，而是：

- 当多个模型/融合候选收敛到同一只 Top1 时，允许 all-in；
- seed 一致性是确认和风险信号，不是单独放行 all-in 的理由；
- 如果模型分歧大，则退回 `top2_softmax`、`dynamic_risk_budget` 或 MASTER/relation 稳健主线。

## 本轮已完成

对应提交：`2297f48`

- `top1_weight` 已正式进入 MASTER 和 StockMixer 的主候选策略，不再只是高风险备选。
- MASTER 和 StockMixer 支持 `top1_margin_weight`、`top1_margin_target`。
- 训练评估指标新增 `top1_margin_z` 和 `top1_portfolio_return`。
- `scripts/dynamic_candidate_switch.py` 支持读取多 seed 分数列，计算 seed Top1 投票，并用候选 Top1 一致性控制 all-in。
- `scripts/run_multiseed_top1_holdout.py` 会输出 all-in gate 诊断文件：
  - `outputs/holdout_20260517/multiseed_top1/allin_candidates.csv`
  - `outputs/holdout_20260517/multiseed_top1/allin_stock_gate.csv`
  - `outputs/holdout_20260517/multiseed_top1/allin_recommendation.json`

## Holdout 证据

### 动态候选切换

最新输出：

```csv
stock_id,weight
688981,1.0
```

本轮 latest gate 选择的是：

```text
stockmixer_portfolio_lite_rank
```

四个 challenger 候选的 Top1 都是：

```text
688981
```

因此候选 Top1 一致性为：

```text
4 / 4 = 1.0
```

这是本窗口支持 all-in 的最强证据。

### 多 Seed Top1 回放

aggregate-only 多 seed 排行：

| 模型 | mean Top1 | mean 收益 | vote Top1 | vote 收益 | vote share | Top1 唯一数 |
|---|---:|---:|---:|---:|---:|---:|
| stockmixer_lite | 688256 | 0.083750 | 688256 | 0.083750 | 0.667 | 2 |
| lstm | 600958 | 0.010471 | 688981 | 0.101618 | 0.333 | 3 |
| itransformer | 002415 | -0.018675 | 002415 | -0.018675 | 0.333 | 3 |
| timexer_fast | 002460 | -0.034589 | 300750 | -0.013981 | 0.333 | 3 |
| master_official | 600276 | -0.041465 | 002460 | -0.034589 | 0.333 | 3 |
| stockmixer_official | 300442 | -0.105785 | 300442 | -0.105785 | 0.667 | 2 |

重要观察：

- `stockmixer_lite -> 688256` 有较强 seed 一致性，且实测收益不错。
- `stockmixer_official -> 300442` 也有较强 seed 一致性，但实测收益很差。
- 所以 seed-only agreement 不能作为 all-in 的充分条件。

更严格的 aggregate gate 会把 `688981` 标记为高收益观察，但保持 `allin_allowed=false`。原因是它在 aggregate 表里没有足够的基础模型共同投票支持。这个设计是有意保守的：最终 all-in 权限应该主要来自跨模型/融合候选一致，而不是单个模型内部的 seed 共振。

## 为什么 MASTER 不总是最优

MASTER 仍然是稳健底座，但它不一定是短窗口 Top1 最尖的模型。

可能原因：

- MASTER 更擅长学习较宽的跨股票结构和 market-guided 关系，有利于平均排序质量，但可能会平滑掉短期主题爆发。
- 当前比赛窗口只有约 5 个交易日，最终收益主要取决于能不能抓到单日最强的那只股票。
- StockMixer/ensemble 候选可能更敏捷，因为它们用较轻结构混合 stock、time、indicator 信号，短线响应更快。
- relation postprocess 是规则/统计后处理，可以提高稳健性，但如果没有显式 all-in gate，也可能稀释很尖的 Top1 信号。

所以当前最合理的定位是：

```text
MASTER/relation = 稳健默认底座
StockMixer/ensemble = 短窗口挑战者
gate = 决定 all-in 还是回退稳健组合
```

## 后续调研方向

### P0：扩展 Holdout 窗口

不能只在 2026-05-15 这一个窗口上调 gate。

下一步实验：

- 回放多个近期 T 日，例如 2026-04-17、2026-04-24、2026-04-30、2026-05-08、2026-05-15；
- 每个窗口都跑同一套 multi-seed 和 dynamic-switch 流程；
- 记录“候选一致性”是否真的能预测 profitable all-in。

建议产出：

```text
scripts/run_recent_holdout_matrix.py
docs/top1_gate_holdout_matrix.md
```

核心指标：

```text
all-in 命中率、平均收益、p05 收益、负收益比例、最大回撤
```

### P1：把 Gate 做成可学习选择器

当前 gate 是规则式的，好处是透明，但无法学习复杂交互，例如：

- margin 高但流动性差；
- 多个候选一致，但其实来自同一模型家族；
- 行业动量强，但指数 regime 弱；
- seed 一致性对某个模型有用，对另一个模型反而危险。

下一版可以训练一个小型 logistic/GBDT selector：

- 目标：当天 all-in Top1 是否优于 fallback 组合；
- 特征：
  - candidate Top1 agreement count/share；
  - seed vote share；
  - top1 margin z；
  - top strength；
  - model family；
  - market risk/trend；
  - industry collective momentum；
  - volume confirmation；
  - liquidity / volatility。

注意：selector 要小，只能 walk-forward 训练。它只决定组合模式，不替代股票预测模型。

### P2：继续强化 Top1 导向训练

这次新增的 `top1_margin_weight` 只是第一步。

后续值得实现的 loss：

- `top_hit_loss`：直接奖励真实头部股票进入预测头部；
- pairwise top-vs-bottom loss：强制真实头部桶高于底部桶；
- listwise Top1 temperature loss：更尖锐地优化榜首位置；
- Top1 margin calibration：只在真实 label 也拉开差距时要求预测分数拉开，避免噪声日硬拉 margin。

训练规则：

```text
按同日截面 batch
优化截面排序
验证时重点看 Top1/all-in return，而不只看 RankIC
```

### P3：短窗口主题/量价确认特征

all-in 策略需要回答“为什么今天就是这只股票”。

建议新增或强化：

- 3/5/10 日相对 HS300 强度；
- 换手率和成交额加速；
- 如果数据支持，加入 gap/open 强度；
- 行业 collective momentum；
- 同行业 breadth 和 volume confirmation；
- 涨停接近度 / 近期突破风格特征；
- 对脆弱拉升加入流动性惩罚。

这些特征应同时进入模型和 gate。

### P4：升级 Relation Layer

当前 relation postprocess 有效，但仍然偏手工。

下一步 relation 方向：

- 行业关系；
- beta 邻居关系；
- 波动率邻居关系；
- 流动性邻居关系；
- 历史收益相关性邻居；
- 参考 HIST 思路的 hidden concept neighbors。

近期实现仍建议保持轻量：

```text
先作为 relation postprocess 验证
只有稳健有效的关系信号再提升为模型特征
```

### P5：漂移与在线适配

近期窗口比远期历史更重要，但 naive recent-only training 已经表现不稳。

可研究方向：

- 保留 full-history 模型作为 anchor；
- 增加 recent-window adapter 或 residual head；
- walk-forward 调 adapter 权重；
- 对比 DoubleAdapt 式增量适配。

这条线适合解决短窗口比赛中 market regime 快速切换的问题。

### P6：最终提交策略

最终提交不应该固定成某个模型名。

推荐流程：

```text
1. 生成 MASTER/relation 稳健提交。
2. 生成 StockMixer 和 ensemble challenger 提交。
3. 运行 multi-seed 诊断。
4. 运行 dynamic candidate switch。
5. 如果跨模型/融合候选 Top1 一致性强，提交 all-in。
6. 如果分歧大，提交 top2/top3 或 MASTER/relation。
7. 保留等权 Top5 作为保守备选。
```

## 已检查外部参考

- MASTER 官方代码：https://github.com/SJTU-DMTai/MASTER
- MASTER 论文页：https://ojs.aaai.org/index.php/AAAI/article/view/27767
- StockMixer 官方代码：https://github.com/SJTU-DMTai/StockMixer
- StockMixer 论文页：https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/
- HIST 官方代码：https://github.com/Wentao-Xu/HIST
- HIST arXiv：https://arxiv.org/abs/2110.13716
- DoubleAdapt 官方代码：https://github.com/SJTU-DMTai/DoubleAdapt
- DoubleAdapt arXiv：https://arxiv.org/abs/2306.09862

## 立即执行项

1. 做一个多窗口 holdout matrix runner。
2. 用多窗口结果只调 gate 阈值，不调模型权重。
3. 真实训练 Top1-margin MASTER/StockMixer 配置，再跑同一套 matrix。
4. 增加短窗口主题/量价确认特征。
5. 如果规则阈值仍然脆弱，把 rule gate 升级成小型 walk-forward selector。
