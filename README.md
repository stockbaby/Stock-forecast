# Stock Forecast for CSI 300 Competition

面向“基于历史数据预测未来股价收益”赛题的研究与工程仓库。当前目标是基于沪深 300 成分股公开可获取数据，预测未来一周收益较高的股票，并生成符合赛题要求的 `result.csv`。

当前仓库已经跑通：
- 数据抓取与清洗
- 特征工程与标签构建
- `LightGBM / MLP / LSTM / StockMixer / MASTER` 实验
- 组合构建与提交文件校验
- 赛题公式收益回看
- 官方 `app/` 打包目录对齐

## 1. 当前结论

截至目前，仓库里的结论可以概括为：

- `LightGBM` 是稳定强基线，排序能力好。
- `LSTM` 作为时序深度基线有一定组合收益潜力，但整体不如强化版 `StockMixer`。
- `MASTER` 是当前验证集表现最强的模型。
- 强化版 `StockMixer` 是当前最近窗口本地回看最强的单模型提交稿。
- 集成工具已经实现，但在当前这轮最近窗口回看下，还没有稳定超过最强单模型提交稿。
- “训练层面的 recent bias” 已验证不适合作为当前主方向；“选择层面的 recent-aware 候选筛选”保留为辅助工具。

## 2. 当前最好结果

### 2.1 强化版 StockMixer

主指标：
- `RankIC = 0.0591`
- `Precision@5 = 0.0462`
- `Top-k portfolio return = 0.0103`

最优组合策略：
- `proportional_positive_thr0.0`

验证期该策略均值：
- `0.01085`

当前主提交稿：
- `outputs/submissions/result_stockmixer_alpha.csv`

### 2.2 MASTER

主指标：
- `RankIC = 0.0348`
- `Precision@5 = 0.0725`
- `Top-k portfolio return = 0.0176`

最优组合策略：
- `proportional_positive_thr0.0`

验证期该策略均值：
- `0.01831`

当前 MASTER 提交稿：
- `outputs/submissions/result_master_alpha.csv`

说明：
- 这是当前验证集综合表现最强的模型
- 但在最近单窗口本地回看里，略弱于当前强化版 `StockMixer`

### 2.3 集成模型

当前 recent-decay 集成最优权重：
- `LightGBM 0.4 + LSTM 0.6 + StockMixer 0.0`

主指标：
- `RankIC = 0.0290`
- `Precision@5 = 0.0791`
- `Top-k portfolio return = 0.0101`

说明：
- 集成在某些验证指标上更稳
- 但在当前最近窗口真实回看里，仍未超过最强单模型 `StockMixer`

主集成稿：
- `outputs/submissions/result_ensemble_alpha.csv`

候选集成稿：
- `outputs/submissions/ensemble_candidates/candidate_1.csv`
- `outputs/submissions/ensemble_candidates/candidate_2.csv`
- `outputs/submissions/ensemble_candidates/candidate_3.csv`

## 3. 模型对比快照

| 模型 | RankIC | Precision@5 | Top-k Return | 最近窗口本地回看 |
|---|---:|---:|---:|---:|
| LightGBM | 0.0210 | 0.0767 | -0.0016 | 0.03386 |
| LSTM | 0.0136 | 0.0356 | 0.0060 | 未作为主稿保留 |
| StockMixer | 0.0591 | 0.0462 | 0.0103 | 0.06332 |
| MASTER | 0.0348 | 0.0725 | 0.0176 | 0.06097 |
| iTransformer | 0.0087 | 0.0242 | 0.0053 | -0.01166 |
| Ensemble | 0.0290 | 0.0791 | 0.0101 | 当前主稿不稳定 |

当前判断：
- 如果更看重验证期稳健性，`MASTER` 最强
- 如果更看重最近窗口本地回看，`StockMixer` 最强

## 4. 近期回看结论

按赛题收益公式，在最近可回看窗口：
- 参考交易日 `2026-04-17`
- 买入日 `2026-04-20`
- 卖出日 `2026-04-24`

当前最佳单模型提交稿 `result_stockmixer_alpha.csv` 的本地回看收益约为：
- `0.06332`

`MASTER` 当前本地回看收益约为：
- `0.06097`

这一轮 recent-decay 集成主稿未超过该值，因此：
- 当前更推荐保留 `StockMixer` 作为主提交方向
- 集成作为候选与验证辅助工具继续保留

## 5. 相对赛题基准代码

基准仓库：
- [Sherlock1956/THU-BDC2026](https://github.com/Sherlock1956/THU-BDC2026)

按官方基准仓库 `README`，其 baseline 方案核心是：
- `StockTransformer`
- 过去 `60` 个交易日序列
- `39` 或 `158+39` 特征
- 排序损失
- 输出前 `5` 只股票，固定等权 `0.2`

我们当前相对基准代码的主要增强点：
- 额外做了 `LightGBM / LSTM / StockMixer / MASTER` 多模型路线
- 加了市场上下文、regime、style-group、相对指数特征
- 支持多种组合策略，而不是只做固定等权
- 支持赛题公式本地回看与候选提交管理
- 支持集成、rolling/recent 验证

当前需要诚实说明的一点：
- 我们已经做了“结构和结果层面的增强”
- 但还没有在仓库内完成一次“官方 baseline 原仓复现后再做严格同仓数值对比”

所以当前能给出的结论是：
- 从方法和实验宽度上，我们已经明显超过赛题基准代码
- 从严格数值同仓对比角度，还建议后续补一次官方 baseline 原样复现

## 6. 目录说明

```text
.
├── README.md
├── docs/
├── configs/
├── data/
├── outputs/
├── scripts/
├── src/
└── app/
```

重点目录：
- `docs/`: 文档、调研、路线、当前状态
- `configs/`: 训练、集成、提交配置
- `scripts/`: 数据抓取、训练、集成、验证脚本
- `src/`: 特征、模型、训练、组合核心代码
- `outputs/predictions/`: 预测结果与 metrics
- `outputs/submissions/`: 提交文件与候选提交
- `app/`: 对齐官方复现与 Docker 打包规范的入口目录

## 7. 常用脚本

### 5.1 抓取比赛基准数据

```powershell
python scripts/fetch_benchmark_data.py --data-dir data/raw --output-name stock_data.csv --start-date 2015-01-01 --end-date 2026-04-24
```

### 5.2 抓取沪深 300 指数

```powershell
python scripts/fetch_hs300_index.py --output-path data/raw/hs300_index.csv --start-date 2015-01-01 --end-date 2026-04-24
```

### 5.3 构建训练数据

```powershell
python scripts/build_dataset.py --config configs/a_stage_round1.yaml
```

### 5.4 训练 LightGBM 基线

```powershell
python scripts/train_baseline.py --config configs/a_stage_round1.yaml
```

### 5.5 训练 LSTM 基线

```powershell
python scripts/train_lstm_baseline.py --config configs/lstm_alpha.yaml
```

### 7.6 训练强化版 StockMixer

```powershell
python scripts/train_stockmixer_baseline.py --config configs/stockmixer_alpha.yaml
```

### 7.7 训练 MASTER

```powershell
python scripts/train_master_baseline.py --config configs/master_alpha.yaml
```

### 7.8 运行集成

```powershell
python scripts/run_ensemble.py --config configs/ensemble_alpha.yaml
```

### 7.9 校验提交文件

```powershell
python scripts/validate_submission.py --input outputs/submissions/result_stockmixer_alpha.csv
```

### 7.10 按赛题公式回看收益

```powershell
python scripts/evaluate_submission_return.py --data-path data/raw/stock_data.csv --submission-path outputs/submissions/result_stockmixer_alpha.csv --trade-date 2026-04-17 --buy-offset 1 --sell-offset 5 --sell-fallback-offset 4
```

## 8. 关键配置

当前最重要的配置文件：
- `configs/a_stage_round1.yaml`
- `configs/lstm_alpha.yaml`
- `configs/stockmixer_alpha.yaml`
- `configs/master_alpha.yaml`
- `configs/ensemble_alpha.yaml`

其中：
- `a_stage_round1.yaml` 对齐当前 A 阶段特例：`T+5` 缺失时回退到 `T+4`
- `stockmixer_alpha.yaml` 是当前主力单模型配置
- `master_alpha.yaml` 是当前验证表现最强模型配置
- `ensemble_alpha.yaml` 是当前集成与候选提交配置

## 9. 当前最推荐的使用顺序

如果现在要准备比赛提交，建议：

1. 主提交候选优先看 `result_stockmixer_alpha.csv`
2. 同时保留 `result_master_alpha.csv` 作为强备选
3. 集成候选仅作参考，不建议当前直接压过单模型主稿
4. 提交前再用最新公开数据重新抓数、重训、重出稿一次

## 10. 文档索引

建议优先看：
- `docs/current_status.md`
- `docs/benchmark_comparison.md`
- `docs/benchmark_same_window_comparison.md`
- `docs/roadmap.md`
- `docs/model_survey.md`
- `docs/sota_survey.md`
- `docs/data_pipeline.md`
- `docs/code_spec_alignment.md`

## 11. 下一步建议

当前最值得继续投入的方向：

1. 在官方 baseline 已完成复现的基础上，继续做更多窗口的同仓对比
2. 更贴比赛的排序目标函数，但不要继续做过强的 recent-biased training
3. 更稳的多候选提交管理
4. 若比赛允许且工程成本可控，再引入额外结构化数据

当前不建议继续作为主线投入的方向：
- 训练层面的 recent bias
- 为了“看起来更像近期”而明显牺牲整体稳健性的目标改造

## 12. 官方 Baseline 严格同仓对比

本仓库已经完成官方 baseline 仓库 `Sherlock1956/THU-BDC2026` 的本地复现，并在**同一数据文件、同一参考交易日、同一买卖窗口**下完成严格数值对比。

对比口径：
- 官方仓库：`external/THU-BDC2026`
- 官方训练数据：使用同一份 `data/raw/stock_data.csv` 生成的 `external/THU-BDC2026/data/train.csv`
- 官方预测输出：`external/THU-BDC2026/output/result.csv`
- 参考交易日 `T`：`2026-04-17`
- 买入日 `T+1`：`2026-04-20`
- 卖出日：按本轮规则回看到 `2026-04-24`

官方 baseline 本地回看收益：
- `0.02350`

我们当前主要结果：

| Submission | Local Return | vs Official Baseline |
|---|---:|---:|
| Official baseline | 0.02350 | 0.00000 |
| LightGBM | 0.03386 | +0.01036 |
| MASTER | 0.06097 | +0.03747 |
| StockMixer | 0.06332 | +0.03982 |

当前结论：
- 严格同仓对比下，`LightGBM / MASTER / StockMixer` 都已经超过官方 baseline。
- 当前最近窗口本地回看最强提交稿仍然是 `outputs/submissions/result_stockmixer_alpha.csv`。
- 当前验证集综合最强模型仍然是 `MASTER`。

## 14. Backbone 实验：iTransformer

本轮已按文档中的通用 backbone 方向实现并评测 `iTransformer`：
- 配置：`configs/itransformer_alpha.yaml`
- 训练脚本：`scripts/train_itransformer_backbone.py`
- 提交文件：`outputs/submissions/result_itransformer_alpha.csv`

结果：
- `RankIC = 0.0087`
- `Precision@5 = 0.0242`
- `Top-k portfolio return = 0.0053`
- 最近窗口本地回看：`-0.01166`
- 相对官方 baseline：`-0.03516`

判断：
- 它作为通用时序 backbone 能学到一定信号，但当前对这道题的适配度明显不如 `StockMixer / MASTER`。
- 目前不建议把 `iTransformer` 继续作为主力提交方向。
- 更适合把它保留为“后续研究型 backbone 参考”，而不是当前比赛主线。

## 13. 官方 Baseline 值得借鉴的点

虽然当前结果已经超过官方 baseline，但官方方法里仍有几处很值得继续吸收：

1. 任务定义对题
- 官方直接把问题建成“同日截面排序”，而不是普通价格回归。
- 这和比赛的 Top-5 组合目标高度一致。

2. Top-5 倾斜的排序损失
- 官方 `WeightedRankingLoss` 会对头部样本施加更高权重。
- 这个思想比单纯做收益回归更贴比赛目标。

3. 60 日序列输入
- 官方默认使用过去 `60` 个交易日序列。
- 这说明中短期趋势、波动和量价结构在官方设计里被认为需要更长上下文。

4. 简洁稳健的提交逻辑
- 官方预测端固定输出 Top-5 且等权 `0.2`。
- 这不一定是收益最优，但非常稳、非常容易复现，也适合作为保守备选提交稿。

当前我们准备吸收的方向：
- 把官方这种 Top-5 倾斜思想继续迁移到 `StockMixer / MASTER` 的目标函数中。
- 增加“固定等权备选 submission”，作为与当前动态权重稿并行的稳健候选。

已完成的一次迁移实验：
- 配置文件：`configs/stockmixer_alpha_official_rank.yaml`
- 结果：验证集指标优于原始强化版 `StockMixer`
  - `RankIC: 0.0591 -> 0.0657`
  - `Precision@5: 0.0462 -> 0.0527`
  - `Top-k return: 0.0103 -> 0.0142`
- 但最近单窗口本地回看从 `0.06332` 降到 `0.03964`

当前判断：
- 这说明“官方 Top-5 加权损失”作为训练增强是有价值的。
- 但在本轮提交窗口上，它还不适合直接替换当前主提交稿。
- 因此目前保留为**可选实验分支**，不覆盖当前主稿 `result_stockmixer_alpha.csv`。

在 `MASTER` 上也完成了同类迁移实验：
- 配置文件：`configs/master_alpha_official_rank.yaml`
- 验证集指标有小幅提升
  - `RankIC: 0.0348 -> 0.0418`
  - `Top-k return: 0.0176 -> 0.0183`
- 但最近单窗口本地回看从 `0.06097` 直接降到 `-0.02259`

因此当前结论更明确：
- 官方 Top-5 加权损失思想对验证集排序有帮助
- 但在当前提交窗口里，迁移到 `StockMixer / MASTER` 后都没有转化成更强的真实回看收益
- 所以目前统一保留为**实验配置**，不升级为主稿

## 15. 固定等权备选 Submission 对比

已经为当前强模型都生成了固定等权备选稿：
- `outputs/submissions/result_stockmixer_alpha_equal_weight.csv`
- `outputs/submissions/result_master_alpha_equal_weight.csv`
- `outputs/submissions/result_master_alpha_official_rank_equal_weight.csv`

最近窗口本地回看结果：

| Submission | Local Return | 结论 |
|---|---:|---|
| StockMixer dynamic | 0.06332 | 当前最佳 |
| StockMixer equal-weight | 0.06077 | 略弱，可作稳健备选 |
| MASTER dynamic | 0.06097 | 当前强备选 |
| MASTER equal-weight | 0.04982 | 明显弱于动态 |
| MASTER official-rank dynamic | -0.02259 | 不保留 |
| MASTER official-rank equal-weight | -0.02150 | 略好于动态，但仍不保留 |

当前判断：
- 在我们当前最强几条线上，动态权重仍普遍优于固定等权。
- 固定等权可以保留为“保守候选稿”，但不建议替换主提交稿。
