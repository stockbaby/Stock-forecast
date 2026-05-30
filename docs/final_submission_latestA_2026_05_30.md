# LatestA 最终提交说明 - 2026-05-30

## 提交口径

本文记录线上赛 A 阶段 latestA 的最终提交选择：

- 提交窗口：2026-05-30 08:00 至 2026-05-31 23:59
- 数据截止日：2026-05-29
- 预测目标：2026-05-31 之后的 T+5 开盘到开盘收益
- 最终同步文件：
  - `app/model/result.csv`
  - `app/output/result.csv`

## 最终提交组合

```csv
stock_id,weight
300308,0.5721812227310785
002384,0.4278187772689215
```

最终选择的是稳健融合候选：

```text
75% MASTER single-seed 分数
25% StockMixer official multi-seed 分数
日截面 z-score 标准化
top2_softmax 配权
```

## 选择理由

硬 all-in gate 拒绝单股 all-in，主要原因是：

- 没有跨模型家族的同股共识；
- 官方 baseline Top5 与主 alpha 候选冲突；
- fallback 支持不足；
- StockMixer 单 seed 的 `002384` 被真实 multi-seed 分歧削弱。

随后 portfolio selector 在更安全的小候选集合中比较：

| 候选 | 平均收益 | p05 收益 | 负收益率 | 稳健分 |
|---|---:|---:|---:|---:|
| fusion_top2 | 3.6266% | 1.0165% | 0.0% | 0.0362 |
| master_top3 | 1.6337% | -1.5286% | 40.0% | 0.0013 |
| official_top5 | 0.6295% | -2.3151% | 60.0% | -0.0120 |
| master_ms_top3 | -0.2746% | -5.8845% | 60.0% | -0.0445 |

融合候选不是 all-in。它保留 MASTER 支持的 `300308` 作为锚，同时把 StockMixer 的进攻信号 `002384` 控制在 50% 以下。

## Multi-Seed 发现

真实 multi-seed 使用 seeds `42,52,62`。

| 模型 | 单 seed Top1 | multi-seed Top1 | seed 一致性 | 解读 |
|---|---:|---:|---:|---|
| MASTER | 300308 | 300308 | 0.67, 2 unique | 稳健锚点 |
| StockMixer official | 002384 | 600522 | 0.33, 3 unique | 单 seed Top1 不稳定 |
| TimeXer | 603296 | 603296 | 0.33, 3 unique | 均值后 Top1 保持，但 seed 共识弱 |

## Selector 结论

本次使用了三类辅助 selector：

1. Gate selector：拒绝 all-in，支持分 `0.375`。
2. Portfolio selector：选择 `fusion_top2`。
3. Weight meta-learner：建议 capped top2/top3 集中度，单股最大权重 `0.60`。

这些 selector 只作为辅助判断，因为当前只有 5 个近期线上窗口，样本仍然很少。

## 关键产物

```text
outputs/latestA_ensemble_weights_20260529_wide/candidate_1.csv
outputs/latestA_ensemble_weights_20260529_wide/leaderboard.json
outputs/latestA_ensemble_weights_20260529_wide/summary.json
outputs/final_submission_decider_latestA_20260529/candidate_diagnostic_table.csv
outputs/final_submission_decider_latestA_20260529/decision_summary.json
outputs/three_selectors_latestA_20260529/three_selector_report.json
scripts/search_latest_ensemble_weights.py
scripts/run_three_selectors.py
scripts/final_submission_decider.py
```

## 打包与复现检查

提交目录已整理到：

```text
submission_package_latestA_20260530/
```

包内保留 Dockerfile 所需的最小运行内容：

```text
Dockerfile
requirements-docker.txt
app/
```

本地复现检查已通过：

```powershell
D:\anaconda\envs\stock-forecast\python.exe submission_package_latestA_20260530\app\code\train.py --model-result submission_package_latestA_20260530\app\model\result.csv --final-output submission_package_latestA_20260530\app\output\result.csv
D:\anaconda\envs\stock-forecast\python.exe submission_package_latestA_20260530\app\code\test.py --model-result submission_package_latestA_20260530\app\model\result.csv --final-output submission_package_latestA_20260530\app\output\result.csv
```

输出保持为：

```csv
stock_id,weight
300308,0.5721812227310785
002384,0.4278187772689215
```

Docker 客户端可用，但当前机器 Docker daemon 未启动，`docker build` 未能连接到 `dockerDesktopLinuxEngine`，因此容器内构建验证需要在 Docker Desktop 启动后再执行。
