# 代码说明

## 环境配置

- Python 3.11
- 主要依赖：pandas、numpy、PyYAML、scikit-learn、lightgbm、torch
- Docker 镜像名：`bdc2026`
- 复现阶段不需要联网；数据需提前放在 `/app/data` 挂载目录下。

## 数据

使用赛题下发的沪深 300 日线数据，默认读取：

- `/app/data/stock_data.csv`：股票日线数据
- `/app/data/hs300_stock_list.csv`：沪深 300 成分股列表，用于行业关键词映射
- `/app/data/hs300_index.csv`：如存在，则用于构造指数和市场状态特征

没有使用额外付费数据、未报备的外部数据或预训练模型。

## 预训练模型

本方案不使用预训练模型。`/app/model/result.csv` 仅保存 `train.sh` 训练后固化的最终组合结果，供 `test.sh` 在 5 分钟内快速生成 `/app/output/result.csv`。

## 算法

### 整体思路介绍

提交版采用当前验证收益最高的 MASTER-style 时序模型：

1. 读取股票日线数据并按股票、日期排序。
2. 构造量价因子、滚动收益/波动/成交量因子、截面排名和标准化特征。
3. 如存在指数数据，加入市场趋势、波动、回撤、个股相对指数收益、beta、特质收益等市场上下文特征。
4. 用未来开盘到开盘收益构造训练标签，并对 A 阶段末尾缺失交易日使用配置化 fallback。
5. 训练 MASTER-style Transformer 回归模型，同时使用回归损失、排序损失、相关性损失和 top-k 官方目标加权损失。
6. 在验证集上选择组合构建策略，最终输出不超过 5 只股票，权重和不超过 1。

### 针对性问题解决方案

- 针对榜单目标，引入 official top-k 加权训练目标，提高模型对前 5 名股票排序的关注度。
- 针对市场风格切换，引入指数上下文和个股相对指数特征。
- 针对过度集中风险，研发版已加入行业约束和置信度留现金机制；提交版保留可复现的最优 MASTER 结果作为主方案。
- 针对预测时间限制，`test.py` 不重新训练，只校验并复制训练阶段生成的 `model/result.csv`。

### 网络结构

MASTER-style 模型使用：

- 30 日 lookback 时序输入
- 特征投影层
- 多头注意力编码层
- 市场状态门控
- MLP 回归头

### 损失函数

训练目标由以下部分加权组成：

- 回归损失
- pairwise rank loss
- 预测与标签相关性损失
- official top-k listwise/pairwise 加权排序损失

### 数据扩增

未使用随机数据扩增。训练过程固定随机种子以保证可复现。

### 模型集成

提交版不做模型集成，使用单个 MASTER-style 模型。研发阶段评估过 StockMixer、iTransformer、LSTM 和组合后处理策略。

## 训练流程

运行：

```bash
bash /app/train.sh
```

主要步骤：

1. 调用 `/app/code/train.py`
2. 使用 `/app/configs/master_alpha_official_rank.yaml`
3. 构建训练特征和标签
4. 训练 MASTER-style 模型并生成验证期预测
5. 根据验证期组合收益选择组合策略
6. 写出 `/app/model/result.csv`，并同步写出 `/app/output/result.csv`

## 推理流程

运行：

```bash
bash /app/test.sh
```

主要步骤：

1. 读取 `/app/model/result.csv`
2. 校验列名、股票数、股票代码、权重和、非负权重
3. 写出 `/app/output/result.csv`

## 其他注意事项

- 预测阶段不联网、不训练，满足 5 分钟预测时间限制。
- 训练阶段按固定随机种子运行，目标是在 8 小时限制内复现。
- 最终输出文件必须是 `/app/output/result.csv`，包含 `stock_id,weight` 两列。
- 如后续启用额外开源数据或模型，需要按赛事要求邮件报备链接和 md5。
