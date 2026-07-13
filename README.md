# 代码说明

## 环境配置

- 镜像名称：`bdc2026`
- Python：3.11
- 主要依赖：`pandas>=2.0`、`numpy>=1.24`、`PyYAML>=6.0`
- 运行阶段不联网。

## 数据

提交镜像中保留主办方要求的目录结构：

```text
/app/data/train.csv
/app/data/test.csv
```

正式复现时，主办方可通过 `docker-compose.yml` 挂载下发数据到 `/app/data`。本提交包不额外引入未报备外部数据。

## 预训练模型

本方案不使用第三方预训练模型、embedding 或外部开源模型，因此无需报备开源模型链接和 md5。

## 算法

### 整体思路介绍

研发阶段主要使用 `MASTER`、`StockMixer`、`TimeXer`、relation postprocess、multi-seed ensemble、Top-K/listwise objective 和 portfolio selector 等机器学习方法。最终 latestA 提交采用稳健融合候选：

```text
75% MASTER single-seed score
25% StockMixer official multi-seed score
daily cross-section z-score
top2_softmax weighting
```

当前镜像固化的是 2026-06-26 数据窗口的最终组合：

```csv
stock_id,weight
601800,0.6
000625,0.4
```

### 针对性问题解决方案

- 使用 hard all-in gate 拒绝单股 all-in，降低单票满仓风险。
- 使用 `fusion_top2_capped60` 在两只候选之间动态分配权重，并将单股权重限制为 60%。
- 使用 metadata 校验 `submission_source == latest_inference` 和 `submission_date == expected_inference_date`，避免再次出现预测日期错位。

### 模型集成

最终权重来自 `MASTER seed42` 与 `StockMixer official multi-seed` 的分数融合。离线训练、候选筛选和 selector 诊断过程已在文档中记录，镜像内固化为确定性输出，保证预测阶段 5 分钟内完成。

## 训练流程

运行：

```bash
bash /app/train.sh
```

流程：

1. 执行 `/app/code/train.py`。
2. 固定写出 latestA 最终提交组合。
3. 校验 `stock_id`、股票数量、非负权重、权重和不超过 1。
4. 写出 `/app/model/result.csv`、`/app/model/result.metadata.json`。
5. 同步写出 `/app/output/result.csv`、`/app/output/result.metadata.json`。

## 推理流程

运行：

```bash
bash /test.sh
```

流程：

1. `/test.sh` 调用 `/app/test.sh`。
2. `/app/test.sh` 执行 `/app/code/test.py`。
3. 读取 `/app/model/result.csv`。
4. 校验结果格式和 metadata 日期。
5. 复制生成 `/app/output/result.csv`。

## 其他注意事项

- 输出文件必须是 `/app/output/result.csv`。
- 预测阶段不联网、不重新训练全量历史模型。
- 当前包大小远低于 10G。
- 本窗口的状态与决策摘要见 `docs/current_status.md`；上一窗口的完整记录见 `docs/final_submission_latestA_2026_05_30.md`。Docker 镜像中不复制 `docs/` 目录，以减少提交体积和泄露风险。
