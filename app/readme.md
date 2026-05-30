# 代码说明

## 环境配置

- Python 3.11
- 主要依赖：pandas、numpy、PyYAML、scikit-learn、lightgbm、torch
- Docker 镜像名：`bdc2026`
- 预测阶段不联网、不训练，直接读取 `/app/model/result.csv` 并写出 `/app/output/result.csv`。

## 数据

训练阶段默认读取赛题下发或本地准备的数据：

- `/app/data/stock_data.csv`
- `/app/data/hs300_stock_list.csv`
- `/app/data/hs300_index.csv`，如存在则使用

提交镜像中不依赖额外未报备外部数据或预训练模型。

## 预训练模型

本方案不使用预训练模型。`/app/model/result.csv` 保存训练和后处理后的最终组合结果。

## 算法

### 整体思路

主模型为 MASTER-style 时序模型，使用量价特征、市场指数上下文、个股相对指数收益、beta、风格分组等特征预测未来开盘到开盘收益。

当前提交目录中的最终结果来自：

1. MASTER official-rank 训练输出预测分数。
2. 对每日截面预测分数做 z-score 标准化。
3. 引入行业内 rank、beta、波动率、流动性、历史相关邻居和 regime 风险作为关系增强信号。
4. 使用 relation 2.2 topk-regime 后处理得到关系增强分数。
5. 使用 `softmax_t0.55` 生成不超过 5 只股票、权重和不超过 1 的组合。

### 关系增强

当前最佳后处理公式：

```text
score_relation =
  1.325 * score_z
  -0.325 * industry_rank
  +0.05 * beta_20_rank
  -0.20 * volatility_20_rank
  +0.10 * liquidity_20_rank
  +0.125 * corr_peer_score
  -0.075 * high_risk_regime * volatility_20_rank
```

其中：

- `score_z` 是 MASTER 分数的日截面 z-score。
- `industry_rank` 是同一交易日、同行业内的分数百分位排名。
- `corr_peer_score` 是历史收益正相关邻居的分数确认信号。
- `high_risk_regime` 由滚动波动率和 beta 的市场状态代理得到。

该关系增强在验证窗口中优于原始 MASTER 动态权重和纯 z-score softmax 版本。

### 损失函数

MASTER 训练阶段使用：

- 回归损失
- pairwise rank loss
- 相关性损失
- official top-k listwise/pairwise 加权排序损失

### 模型集成

研发阶段评估过 StockMixer、iTransformer、TimeXer、近期窗口 MASTER、多模型融合和 Top-K/listwise 增强 MASTER；当前提交版使用单主模型加 relation 2.2 关系增强后处理。

## 训练流程

运行：

```bash
bash /app/train.sh
```

训练入口会运行 MASTER official-rank 配置，并生成 `/app/model/result.csv` 和 `/app/output/result.csv`。

## 推理流程

运行：

```bash
bash /app/test.sh
```

推理阶段：

1. 读取 `/app/model/result.csv`
2. 校验股票代码、股票数、权重和、非负权重
3. 写出 `/app/output/result.csv`

## 当前结果

```csv
stock_id,weight
002493,1.0
```

Latest note: the current result is the aggressive Top1 allocation generated after fixing true latest inference for unlabeled `T=2026-04-24`. See `docs/latest_progress_2026_05_06.md` for the replay score, fallback strategies, and multi-method validation.

## 其他注意事项

- 最终输出文件必须是 `/app/output/result.csv`。
- 预测阶段满足 5 分钟限制。
- 如后续引入外部数据或预训练模型，需要按赛事要求报备链接和 md5。

# LatestA 当前提交结果

当前 `app/model/result.csv` 和 `app/output/result.csv` 已更新为 2026-05-30 线上赛 A 阶段 latestA 融合提交：

```csv
stock_id,weight
300308,0.5721812227310785
002384,0.4278187772689215
```

生成逻辑：

- 数据截止日：2026-05-29
- all-in gate：拒绝单股 all-in
- portfolio selector：选择 `fusion_top2`
- 融合权重：`75% MASTER single-seed + 25% StockMixer official multi-seed`
- 分数处理：日截面 z-score 后使用 `top2_softmax`

详细记录：`docs/final_submission_latestA_2026_05_30.md`。

`train.sh` 会确定性写出上述最终结果和 `result.metadata.json`；`test.sh` 只做离线校验和复制，满足预测阶段快速复现要求。
