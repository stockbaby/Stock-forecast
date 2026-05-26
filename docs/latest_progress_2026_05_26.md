# 最新进展 - 2026-05-26

## 对齐调研文档的后续落地

本轮继续沿 `docs/domain_research_breakthroughs_2026_05_25.md` 推进，重点覆盖 P1、P3、P4。

## P3/P4: 动态风险预算与 seed 分歧输入

新增组合策略 `dynamic_risk_budget`，用于在 `top1_weight`、`top2_softmax`、`confidence_topk` 三档之间做规则化切换：

- 分数边际高、Top1 强度高、seed 分歧低时，使用 Top1 集中仓位；
- 分数边际低或 seed 分歧高时，降级为 `top2_softmax`；
- 中间状态使用 confidence-style 分散权重。

该策略会读取 multi-seed 聚合输出中的 `score_std`。同时修复了组合评估函数过去丢弃 `score_std` 的问题，使 seed disagreement 可以真正进入验证和提交构造链路。

补充：`dynamic_risk_budget` 现已接入市场状态输入。新生成的序列模型预测会从 meta 透传以下上下文列：

- 指数/市场状态：`regime_trend`、`regime_vol_ratio`、`regime_drawdown`、`regime_score`、`regime_is_high_vol`
- 指数动量与回撤：`index_ret_5`、`index_ret_10`、`index_ret_20`、`index_drawdown_20`
- 截面量价状态：`ret_5`、`ret_20`、`volume_ratio_5`、`volume_ratio_20`、`amount_ratio_5`、`amount_ratio_20`
- 行业/主题拥挤：`industry_id`、`industry_collective_momentum`、`industry_collective_volume_confirm`

组合层会用这些输入构造 `market_risk` 与 `market_momentum`。高波动、深回撤、量能异常或行业集中时降低 Top1 集中度；正向趋势和低风险时允许更集中。旧预测文件没有这些列时会自动回退到分数边际和 `score_std` 逻辑。

## 风险指标补齐

`evaluate_portfolio_strategy` 现在除均值和波动外，还输出：

- `p05_return`
- `negative_rate`
- `max_drawdown`

`scripts/validate_multiple_methods.py` 和 `scripts/simulate_online_windows.py` 也同步输出这些指标，便于按调研文档要求比较 20/40/60/90 日窗口的收益和尾部风险。

快速校验结果显示，在旧的 `master_official` 91 日窗口上：

| strategy | mean | std | p05 | neg_rate | latest A |
|---|---:|---:|---:|---:|---:|
| top2_softmax | 0.021445 | 0.049134 | -0.047065 | 0.318681 | 0.049636 |
| top1_weight | 0.021298 | 0.065220 | -0.069151 | 0.373626 | 0.106689 |
| confidence_topk | 0.019817 | 0.052214 | -0.050367 | 0.362637 | 0.062790 |
| dynamic_risk_budget | 0.019429 | 0.056633 | -0.051942 | 0.384615 | 0.049636 |

结论：`dynamic_risk_budget` 已经能在 latest 场景降级到稳健档，但旧窗口均值和风险还没有超过 `top2_softmax`，后续需要继续调阈值或引入市场状态特征。

## P1: Top-label weighted ListNet v1

新增 `top_label_weighted_listnet` surrogate：

- 使用同日截面 rank label，而不是直接使用未来收益幅度；
- 对头部 rank 给更高权重；
- 支持按日期 batch 时逐日计算；
- 作为 hybrid loss 的小权重项使用，避免重走 `portfolio_return_loss` 过拟合路径。

新增配置：

- `configs/master_alpha_top_label_listnet.yaml`
- `configs/holdout_20260517_master_alpha_top_label_listnet.yaml`

默认权重较保守：`top_label_listnet_weight: 0.12`，下一步需要跑训练和 20/40/60/90 日验证，若 top-k 收益或 p05 不优于 official，则降级。

## 验证

已通过：

```text
python -m py_compile src/models/master.py src/models/stockmixer.py scripts/train_master_baseline.py scripts/train_stockmixer_baseline.py src/portfolio/construct.py scripts/simulate_online_windows.py scripts/validate_multiple_methods.py
python -m py_compile src/models/deep_sequence.py src/portfolio/construct.py
```

并用现有 `master_alpha_official_rank` 预测文件跑通：

```text
scripts/validate_multiple_methods.py --strategies top1_weight,top2_softmax,confidence_topk,dynamic_risk_budget
scripts/simulate_online_windows.py --strategies top1_weight,top2_softmax,confidence_topk,dynamic_risk_budget --selection-windows 20,40
```

并用最小样例验证：低风险强趋势输入会给 Top1 满仓；高波动、回撤和负动量输入会降低 Top1 权重并分散到后续候选。

## 下一步

1. 跑 `master_alpha_top_label_listnet` 完整训练，比较 official、top_hit、top_label_listnet。
2. 在 `master_multiseed_latest_predictions.csv` 生成后，优先验证 `dynamic_risk_budget` 是否能同时利用 `score_std` 与市场状态改善 top1 风险。
3. 若市场状态输入有效，再把阈值从手工规则升级为 walk-forward 可学习 selector。
