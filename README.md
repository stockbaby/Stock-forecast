# Stock Forecast for CSI 300 Competition

面向“基于历史数据预测未来股价收益”竞赛的研究型仓库。目标是基于沪深 300 成分股的公开可获取数据，预测未来一周收益率，并输出不超过 5 只股票的投资组合及权重。

本仓库当前阶段先完成方案设计、研究综述与实验路线规划，随后再逐步落地：

1. 数据下载与清洗
2. 特征工程与标签构建
3. 基线模型训练
4. 深度学习模型迭代
5. 排序与组合优化
6. 回测与提交生成

## 1. 比赛任务理解

这道题不是传统的“价格预测”问题，而是一个更接近量化选股的任务：

- 预测目标不是未来价格本身，而是未来一周收益率
- 评估核心不是均方误差，而是最终组合收益率
- 输出是 `Top-5` 以内股票及其权重，而不是全市场逐股打分表
- 因此模型的核心能力应当是：
  - 做好截面排序
  - 找到高收益尾部样本
  - 控制组合构建时的噪声与过拟合

这意味着我们的建模目标要从“回归误差最小”转向“排序质量更高、Top-K 命中更好、组合收益更强”。

## 2. 总体研究路线

我们计划采用“三层递进”的方案：

### 第一层：强基线

先用低成本、强鲁棒的方法建立可靠基线：

- `LightGBM / XGBoost`
- 以日频 OHLCV + 技术指标 + 截面标准化因子为主
- 目标先做未来 5 个交易日收益率回归
- 再将预测值转为 `Top-5` 组合

这样做的好处是：

- 训练快
- 调参成本低
- 能快速验证数据处理是否正确
- 在量化任务里常常比“未经充分调优的深度模型”更强

### 第二层：面向截面排序的深度模型

在强基线稳定后，再进入深度学习阶段，优先考虑：

- `MASTER`：市场引导的股票 Transformer
- `StockMixer`：低成本、效果强的 MLP-based 股票预测模型
- `iTransformer`：适合多变量时序建模的通用骨干
- `Temporal Relational Ranking (RSR)`：更贴近“排序”目标的关系建模方法

我们会优先选择“算力投入合理、复现实操性高”的路线，而不是盲目追求模型复杂度。

### 第三层：组合优化与创新

在预测分数之上，再做组合层优化：

- `Top-K` 选股
- 权重分配策略
- 行业/风格约束
- 不确定性过滤
- 排序学习或端到端组合优化

最终成绩通常不只取决于模型本身，也取决于“如何从预测值变成组合”。

## 3. 推荐参考的开源项目

以下项目适合作为本赛题的直接参考对象：

### 3.1 Microsoft Qlib

- 仓库：<https://github.com/microsoft/qlib>
- 价值：
  - 量化研究领域最成熟的开源框架之一
  - 内置 `Alpha158`、`Alpha360` 等经典特征体系
  - 支持中国市场数据与 `CSI300` 股票池
  - 能快速搭建从数据、特征、标签到训练评估的完整流水线

适合作为本项目的“实验平台底座”。

### 3.2 MASTER

- 仓库：<https://github.com/SJTU-DMTai/MASTER>
- 论文：AAAI 2024, *MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*
- 价值：
  - 明确面向股票预测
  - 强调市场整体信息对个股预测的引导
  - 很适合沪深 300 这种“指数成分股 + 明确市场上下文”的任务

### 3.3 StockMixer

- 仓库：<https://github.com/SJTU-DMTai/StockMixer>
- 论文：AAAI 2024, *StockMixer: A Simple yet Strong MLP-Based Architecture for Stock Price Forecasting*
- 价值：
  - 参数效率高
  - 训练成本明显低于复杂 Transformer / GNN
  - 对“股票间关系 + 时间混合 + 指标混合”都有建模
  - 很适合当作本项目的“低成本强模型”

### 3.4 iTransformer

- 仓库：<https://github.com/thuml/iTransformer>
- 论文：ICLR 2024, *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*
- 价值：
  - 是通用多变量时序预测的强骨干
  - 适合拿来改造成“全市场截面联合建模”的 backbone
  - 更适合做方法迁移，而不是直接拿来当股票任务终版

### 3.5 Temporal Relational Ranking

- 仓库：<https://github.com/fulifeng/Temporal_Relational_Stock_Ranking>
- 论文：TOIS 2019, *Temporal Relational Ranking for Stock Prediction*
- 价值：
  - 非常贴合本题“最终看排序结果”的本质
  - 提醒我们不能只做回归，还要重视 `Top-K` 排名质量
  - 对后续设计 `ranking loss` 很有参考意义

## 4. 论文与方法选择建议

需要先强调一点：

**股票预测领域没有脱离数据集与回测设定的“统一 SOTA”。**

很多论文在自己的数据切分、股票池、交易规则和评价指标下领先，但未必能直接迁移到本赛题。因此我们更重视：

- 是否贴近 A 股日频任务
- 是否适合截面排序
- 是否容易复现
- 是否训练成本可控

基于这些标准，本项目建议优先级如下：

### 优先级 A：最值得落地

1. `LightGBM + Alpha 因子`
2. `Qlib Alpha158/Alpha360 + MLP/LSTM baseline`
3. `StockMixer`
4. `MASTER`

### 优先级 B：作为增强或研究方向

1. `iTransformer`
2. `GAT / STGNN`
3. `Listwise / Pairwise ranking loss`
4. `Mixture of Experts for market regimes`

### 不建议一开始就重仓的方向

1. 纯新闻大模型/NLP 多模态
2. 高频数据建模
3. 强化学习直接学组合

原因不是这些方向不高级，而是：

- 成本高
- 数据清洗复杂
- 很容易过拟合
- 对当前这类比赛未必比“强基线 + 合理组合优化”更有效

## 5. 数据策略：低成本但尽量有效

### 5.1 必选主数据

第一阶段先用最稳的数据源：

- 比赛提供的 `baostock` 历史日线
- 沪深 300 成分股列表及其时间变化

如果成分股历史变动难以完全恢复，初版可以先采用“统一股票池近似法”，后续再逐步升级为“时变股票池”。

### 5.2 强烈建议补充的数据

优先补充“结构化、低频、易对齐”的公开数据，而不是先上文本：

1. 行业分类
2. 指数级别特征
3. 北向资金/资金流特征
4. 财务因子或估值因子

推荐来源：

- `Qlib` 现成数据与特征体系
- `AkShare`
- 公开可下载的指数与宏观数据

### 5.3 为什么这样更划算

相比海量新闻文本或复杂图谱，这些结构化特征：

- 更容易清洗
- 更容易对齐到日频
- 对沪深 300 更有现实意义
- 更适合样本量有限的比赛环境

## 6. 标签与评价指标设计

本项目不会只做一个标签，而是会并行准备多种标签体系：

### 6.1 主标签

- `y_ret_5d_open_open`
  - 从 `T+1` 开盘买入，到 `T+5` 开盘卖出的收益率

这个标签要尽量与比赛收益定义保持一致。

### 6.2 辅助标签

- `y_ret_5d_close_close`
- `y_rank_cs_5d`
- `y_excess_ret_vs_csi300`
- `y_updown_5d`

辅助标签的作用：

- 让模型更稳定
- 支持多任务学习
- 帮助我们判断模型是在学市场方向，还是在学个股超额收益

### 6.3 训练阶段的评价指标

除了 `MSE / MAE`，更应关注：

- `IC` / `RankIC`
- `Precision@5`
- `Top-5 portfolio return`
- `Hit rate of top decile`
- `Excess return vs baseline`

## 7. 特征工程设计

### 7.1 第一批特征

第一批建议控制在可解释、稳定、低泄漏风险的范围内：

- 收益率特征：1/3/5/10/20 日收益
- 波动率特征：5/10/20 日波动率
- 趋势特征：MA, EMA, MACD
- 反转特征：乖离率、短期反转
- 量价特征：成交量变化、量比、价量协同
- K 线结构：上下影线、实体比例、振幅
- 相对强弱：相对沪深 300 的超额收益

### 7.2 截面特征处理

股票任务中，截面标准化往往非常关键：

- 行业内标准化
- 全市场截面 `z-score`
- 分位数归一化
- winsorize 去极值

这一步可以显著提升排序模型稳定性。

### 7.3 第二批增强特征

在基线稳定后加入：

- 市场状态特征：指数收益、指数波动、成交额
- 资金特征：北向资金、主力净流入
- 行业轮动特征
- 财务/估值因子

## 8. 模型策略建议

### 8.1 基线组

- 线性模型：`Ridge / Lasso`
- 树模型：`LightGBM / XGBoost / CatBoost`
- 简单深度模型：`MLP / LSTM / GRU`

目标是先跑通数据、评估、提交与回测闭环。

### 8.2 主力深度模型组

- `StockMixer`
- `MASTER`
- `iTransformer` 改造版

建议顺序：

1. 先复现 `StockMixer`
2. 再做 `MASTER`
3. 最后再尝试更重的 Transformer/GNN 变体

### 8.3 排序优化组

如果普通回归模型已经不错，下一步最值得投入的是：

- `pairwise ranking loss`
- `listwise ranking loss`
- `top-k aware loss`
- 回归分数 + 二阶段排序头

这往往比盲目换更复杂 backbone 更有效。

## 9. 创新性建议

比赛里“创新”不一定等于“最复杂”。更好的路径通常是：

### 9.1 目标对齐创新

不只预测收益率，还直接优化：

- `Top-K` 排序
- 超额收益
- 组合收益

这是最贴近赛题本质的创新。

### 9.2 市场状态分层

先判断市场 regime，再用不同专家模型预测：

- 上行市
- 震荡市
- 下行市

这比让单一模型硬吃全部市场状态更合理。

### 9.3 组合构建层创新

在模型分数之外加入：

- 不确定性惩罚
- 行业分散度约束
- 单票极端暴露约束

这样能明显减少“模型选出的 5 只股票全押一个短期热门板块”的风险。

### 9.4 数据层创新

重点不是“数据越多越好”，而是“正交信息越多越好”：

- 量价因子之外加入市场上下文
- 个股之外加入行业与指数层特征
- 低频基本面做前向填充

## 10. 拟定实施路线

### Phase 0：文档与方案

- 明确标签定义
- 明确数据源名单
- 明确实验指标
- 明确目录结构

### Phase 1：最小可运行版本

- 读取原始数据
- 清洗与对齐
- 构造基础特征
- 训练 LightGBM baseline
- 生成 `result.csv`

### Phase 2：可靠基线

- 加入 Qlib 风格特征
- 做时间切分验证
- 做滚动回测
- 建立实验记录系统

### Phase 3：深度模型

- 复现 `StockMixer`
- 复现/改造 `MASTER`
- 加入市场上下文和截面结构

### Phase 4：排序与组合优化

- 优化 `Top-5` 命中
- 研究权重分配策略
- 做模型集成与稳健性测试

## 11. 计划中的仓库结构

```text
Stock-forecast/
├─ README.md
├─ docs/
│  ├─ roadmap.md
│  └─ model_survey.md
├─ configs/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ notebooks/
├─ src/
│  ├─ data/
│  ├─ features/
│  ├─ models/
│  ├─ training/
│  ├─ portfolio/
│  └─ utils/
├─ outputs/
│  ├─ models/
│  ├─ predictions/
│  └─ submissions/
└─ scripts/
```

## 12. 近期最重要的工作

在真正开始编码前，我们先把下面几件事做扎实：

1. 写清楚训练样本与标签定义
2. 定义严格的时间切分验证方案
3. 明确要接入的免费公开数据源
4. 决定第一阶段的 baseline 与第一阶段主力深度模型

目前我的建议是：

1. 基线先上 `LightGBM + Alpha 类特征`
2. 深度模型首选 `StockMixer`
3. 第二主力模型选 `MASTER`
4. 创新点重点放在：
   - 排序目标
   - 市场状态建模
   - 组合构建优化

## 13. 参考链接

- Qlib: <https://github.com/microsoft/qlib>
- Qlib Data Docs: <https://qlib.readthedocs.io/en/latest/component/data.html>
- MASTER: <https://github.com/SJTU-DMTai/MASTER>
- StockMixer: <https://github.com/SJTU-DMTai/StockMixer>
- iTransformer: <https://github.com/thuml/iTransformer>
- Temporal Relational Ranking: <https://github.com/fulifeng/Temporal_Relational_Stock_Ranking>

## 14. 下一步

下一步建议直接进入文档细化阶段：

1. 写 `docs/roadmap.md`：完整工作拆解与时间表
2. 写 `docs/model_survey.md`：模型与论文调研结论
3. 再开始初始化项目代码结构与数据流水线

## 15. 环境与数据启动

当前仓库已经补充了环境文件与数据流水线脚手架：

- 环境文件：[environment.yml](D:\data_competition\environment.yml)
- 环境说明：[environment.md](D:\data_competition\docs\environment.md)
- 数据流水线说明：[data_pipeline.md](D:\data_competition\docs\data_pipeline.md)

建议启动顺序：

1. 创建并激活 `stock-forecast` conda 环境
2. 将比赛原始 CSV 放入 `data/raw/`
3. 执行：

```powershell
python scripts/build_dataset.py --config configs/baseline.yaml
python scripts/train_baseline.py --config configs/baseline.yaml
```

4. 检查生成物：
   - `data/processed/model_dataset.csv`
   - `outputs/predictions/baseline_metrics.json`
   - `outputs/submissions/result.csv`

## 16. A Stage Round 1 Note

根据官方通知与 Q&A：

- 官方正式评测时会提供“过去十年到比赛提交最后一天”的历史数据
- 因此我们建议将抓数截止日期至少扩展到 `2026-04-24`
- 对于 `2026-04-25` 到 `2026-04-26` 的第一次提交，`T+5` 直接复用 `T+4` 数据

仓库中已提供专用配置：

- [a_stage_round1.yaml](D:\data_competition\configs\a_stage_round1.yaml)

推荐流程：

```powershell
python scripts/fetch_benchmark_data.py --data-dir data/raw --output-name stock_data.csv --start-date 2015-01-01 --end-date 2026-04-24
python scripts/build_dataset.py --config configs/a_stage_round1.yaml
python scripts/train_baseline.py --config configs/a_stage_round1.yaml
```

## 17. Official Packaging Alignment

为对齐官方《代码规范2026》中的复现与打包要求，仓库已新增：

- [code_spec_alignment.md](D:\data_competition\docs\code_spec_alignment.md)
- [app/](D:\data_competition\app)

其中 `app/` 目录用于后续 Docker 打包与离线复现审核，保留了官方要求的关键结构：

- `app/code/src`
- `app/code/featurework.py`
- `app/code/train.py`
- `app/code/test.py`
- `app/data`
- `app/model`
- `app/output`
- `app/temp`
- `app/init.sh`
- `app/train.sh`
- `app/test.sh`
- `app/readme.md`

当前研发工作仍建议在仓库根目录下继续推进，最终提交时再通过 `app/` 入口完成对齐。
