# Stock Forecast for CSI 300 Competition

面向“基于历史数据预测未来股价收益”赛题的研究与工程仓库。当前目标是基于沪深 300 成分股公开可获取数据，预测未来一周收益较高的股票，并生成符合赛题要求的 `result.csv`。

当前仓库已经跑通：
- 数据抓取与清洗
- 特征工程与标签构建
- `LightGBM / MLP / LSTM / StockMixer` 实验
- 组合构建与提交文件校验
- 赛题公式收益回看
- 官方 `app/` 打包目录对齐

## 1. 当前结论

截至目前，仓库里的结论可以概括为：

- `LightGBM` 是稳定强基线，排序能力好。
- `LSTM` 作为时序深度基线有一定组合收益潜力，但整体不如强化版 `StockMixer`。
- 强化版 `StockMixer` 是当前最强单模型。
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

### 2.2 集成模型

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

## 3. 近期回看结论

按赛题收益公式，在最近可回看窗口：
- 参考交易日 `2026-04-17`
- 买入日 `2026-04-20`
- 卖出日 `2026-04-24`

当前最佳单模型提交稿 `result_stockmixer_alpha.csv` 的本地回看收益约为：
- `0.06332`

这一轮 recent-decay 集成主稿未超过该值，因此：
- 当前更推荐保留 `StockMixer` 作为主提交方向
- 集成作为候选与验证辅助工具继续保留

## 4. 目录说明

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

## 5. 常用脚本

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

### 5.6 训练强化版 StockMixer

```powershell
python scripts/train_stockmixer_baseline.py --config configs/stockmixer_alpha.yaml
```

### 5.7 运行集成

```powershell
python scripts/run_ensemble.py --config configs/ensemble_alpha.yaml
```

### 5.8 校验提交文件

```powershell
python scripts/validate_submission.py --input outputs/submissions/result_stockmixer_alpha.csv
```

### 5.9 按赛题公式回看收益

```powershell
python scripts/evaluate_submission_return.py --data-path data/raw/stock_data.csv --submission-path outputs/submissions/result_stockmixer_alpha.csv --trade-date 2026-04-17 --buy-offset 1 --sell-offset 5 --sell-fallback-offset 4
```

## 6. 关键配置

当前最重要的配置文件：
- `configs/a_stage_round1.yaml`
- `configs/lstm_alpha.yaml`
- `configs/stockmixer_alpha.yaml`
- `configs/ensemble_alpha.yaml`

其中：
- `a_stage_round1.yaml` 对齐当前 A 阶段特例：`T+5` 缺失时回退到 `T+4`
- `stockmixer_alpha.yaml` 是当前主力单模型配置
- `ensemble_alpha.yaml` 是当前集成与候选提交配置

## 7. 当前最推荐的使用顺序

如果现在要准备比赛提交，建议：

1. 先以 `result_stockmixer_alpha.csv` 作为主提交候选
2. 同时保留集成候选作对照
3. 提交前再用最新公开数据重新抓数、重训、重出稿一次

## 8. 文档索引

建议优先看：
- `docs/current_status.md`
- `docs/roadmap.md`
- `docs/model_survey.md`
- `docs/sota_survey.md`
- `docs/data_pipeline.md`
- `docs/code_spec_alignment.md`

## 9. 下一步建议

当前最值得继续投入的方向：

1. `MASTER`
2. 更贴比赛的排序目标函数，但不要继续做过强的 recent-biased training
3. 更稳的多候选提交管理
4. 若比赛允许且工程成本可控，再引入额外结构化数据

当前不建议继续作为主线投入的方向：
- 训练层面的 recent bias
- 为了“看起来更像近期”而明显牺牲整体稳健性的目标改造

