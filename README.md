# Stock Forecast for CSI 300 Competition

本仓库用于沪深 300 成分股未来收益预测与 Top-5 组合生成，覆盖数据处理、特征工程、深度模型训练、组合后处理、Docker 提交目录整理和实验记录。

当前推荐结果已经同步到：

```text
app/model/result.csv
app/output/result.csv
```

当前最佳候选为 **MASTER official-rank 预测 + 日截面 z-score + 行业关系增强 + softmax 配权**。

## 当前最佳

最终候选：

```csv
stock_id,weight
002422,0.5366080941073169
688981,0.18532079156062642
300433,0.14533404575723335
002049,0.06974993365670436
688008,0.06298713491811891
```

验证对比：

| 候选 | 策略 | 91 日均值 | 91 日波动 | 结论 |
|---|---|---:|---:|---|
| 已提交 MASTER | `proportional_positive_thr0.0` | `0.01904` | `0.03941` | 原提交版 |
| MASTER z-score softmax | `softmax_t0.6` | `0.02146` | `0.04904` | 分数标准化有效 |
| MASTER relation best | `softmax_t0.6` | `0.02177` | `0.05253` | 当前最佳 |

更完整的实验报告见：

```text
docs/experiment_report_2026_04_27.md
```

## 主要结论

- `MASTER` 是当前最强主模型，后处理后超过之前提交版。
- 行业信息直接作为 StockMixer 特征输入暂时是负增益。
- 行业关系作为后处理信号有效，当前最佳为 `rank_ind_mix alpha=-0.5`。
- 多模型融合、TimeXer fast、近期窗口 MASTER 暂未超过 MASTER relation best。
- 当前 `app/` 已更新到最新最佳结果，可用于二次打包。

## 目录

```text
.
|-- app/                      # 官方提交目录
|-- configs/                  # 训练与实验配置
|-- data/                     # 本地数据，不建议提交大文件
|-- docs/                     # 实验、状态、路线文档
|-- outputs/                  # 预测与提交结果
|-- scripts/                  # 数据、训练、评估、后处理脚本
|-- src/                      # 特征、模型、训练、组合核心代码
|-- Dockerfile                # Docker 打包入口
|-- requirements.txt
|-- requirements-docker.txt
```

## 常用命令

### 复现当前最佳后处理

```powershell
D:\anaconda\envs\stock-forecast\python.exe scripts\relation_postprocess.py --prediction-path outputs/predictions/master_alpha_official_rank_predictions.csv --output-path outputs/submissions/result_master_relation_best.csv --metrics-path outputs/predictions/master_relation_best_metrics.json --mode rank_ind_mix --alpha -0.5 --strategy softmax_t0.6
Copy-Item outputs/submissions/result_master_relation_best.csv app/model/result.csv -Force
Copy-Item outputs/submissions/result_master_relation_best.csv app/output/result.csv -Force
D:\anaconda\envs\stock-forecast\python.exe app/code/test.py
```

### 训练 MASTER official-rank

```powershell
D:\anaconda\envs\stock-forecast\python.exe scripts\train_master_baseline.py --config configs/master_alpha_official_rank.yaml
```

### 训练 StockMixer official-rank

```powershell
D:\anaconda\envs\stock-forecast\python.exe scripts\train_stockmixer_baseline.py --config configs/stockmixer_alpha_official_rank.yaml
```

### 运行组合候选搜索

```powershell
D:\anaconda\envs\stock-forecast\python.exe scripts\search_portfolio_candidates.py --output-dir outputs/portfolio_search_core --models master_official stockmixer_official itransformer ensemble_alpha --grid-step 0.25 --max-ensemble-models 3 --top-n 20
```

### 训练 TimeXer fast

```powershell
D:\anaconda\envs\stock-forecast\python.exe scripts\train_timexer_backbone.py --config configs/timexer_alpha_fast.yaml
```

## Docker 打包

在 `D:\data_competition` 下执行：

```powershell
docker buildx build --platform linux/amd64 -t bdc2026 .
docker run --rm -v D:\data_competition\app\output:/app/output -v D:\data_competition\app\temp:/app/temp bdc2026 bash /app/test.sh
docker save -o PastoralBabyBoom.tar bdc2026:latest
```

当前 `test.sh` 会读取 `/app/model/result.csv` 并生成 `/app/output/result.csv`，预测阶段不重新训练。

## 文档索引

- `docs/experiment_report_2026_04_27.md`：最新实验、评估、比对与结论
- `docs/current_status.md`：当前状态摘要
- `docs/model_survey.md`：模型路线与实验判断
- `docs/code_spec_alignment.md`：提交规范对齐
- `docs/data_pipeline.md`：数据流程
- `docs/benchmark_comparison.md`：早期 benchmark 对比

## 注意

- 大文件、tar 包和中间搜索输出不应提交到 git。
- 外部数据或预训练模型如需进入正式方案，必须按赛事要求报备链接和 md5。
- 当前最佳是后处理候选，不代表 TimeXer / HIST 完整训练路线已被充分穷尽。
