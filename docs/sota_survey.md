# SOTA Survey for CSI300 Stock Return Forecasting

## 1. 文档目的

本文档用于梳理股票预测与选股领域中，**对本项目最有参考价值**的代表性工作，并从工程落地角度评估它们是否适合迁移到本次比赛任务：

- 任务：预测沪深 300 成分股未来一周收益率
- 输出：不超过 5 只股票及其权重
- 目标：提升 `Top-K` 选股质量与最终组合收益

需要特别说明：

**股票预测领域不存在脱离数据集、股票池、标签定义、回测规则的统一“绝对 SOTA”。**

因此，本调研重点不在于简单比较谁“最好”，而在于判断：

1. 是否贴近 A 股日频选股任务
2. 是否重视截面排序与组合收益
3. 是否有开源代码可复现
4. 训练成本是否适合比赛环境
5. 是否值得作为本项目的主力路线或增强方向

## 2. 任务适配标准

我们用以下标准衡量一篇工作对本比赛的参考价值：

- `目标对齐`：是否优化排序、超额收益、组合收益，而不是只做点预测误差
- `市场贴近性`：是否在股票任务，尤其是 A 股 / CSI300 / CSI800 上验证
- `可复现性`：是否有官方代码、数据说明和明确实验设置
- `训练成本`：是否能在单卡或有限算力条件下运行
- `工程迁移性`：是否容易改造到我们的数据与评估体系

## 3. 代表工作总览

| 方法 | 年份 | 核心思想 | 主要数据集 | 官方代码 | 训练成本 | 对本任务适配度 | 是否建议复现 |
|---|---:|---|---|---|---|---|---|
| Temporal Relational Ranking (RSR) | 2019 | 排序学习 + 股票关系建模 | NASDAQ, NYSE | 是 | 低-中 | 高 | 作为方法参考，部分复现 |
| HIST | 2021 | 概念图 + 隐式共享信息建模 | CSI100, CSI300 | 是 | 中 | 很高 | 建议重点参考 |
| DoubleAdapt | 2023 | 应对分布漂移的增量元学习 | Qlib crowd data, CSI300/CSI500 | 是 | 中-高 | 中高 | 作为增强方向 |
| MASTER | 2024 | 市场引导 gating + 时序/截面交替聚合 | CSI300, CSI800 | 是 | 中-高 | 很高 | 强烈建议复现 |
| StockMixer | 2024 | 轻量 MLP 做 indicator/time/stock mixing | NASDAQ, NYSE, S&P500 | 是 | 低-中 | 很高 | 强烈建议复现 |
| iTransformer | 2024 | 变量作为 token 的通用时序 backbone | 通用多变量时序基准 | 是 | 中 | 中 | 作为 backbone 参考 |
| TimeXer | 2024 | 外生变量增强的 Transformer | 通用时序 + exogenous | 是 | 中 | 中高 | 若引入外部数据可参考 |

## 4. 逐项分析

### 4.1 Temporal Relational Ranking (RSR)

- 论文：Temporal Relational Ranking for Stock Prediction
- 时间：2019
- 代码：<https://github.com/fulifeng/Temporal_Relational_Stock_Ranking>
- 论文摘要页：<https://ideas.repec.org/p/arx/papers/1809.09441.html>

| 维度 | 结论 |
|---|---|
| 特点 | 把股票预测直接视为“排序问题”，并用 Temporal Graph Convolution 联合建模时序与股票关系 |
| 优点 | 非常贴合“Top-K 选股”本质；很早就强调不能只做回归；关系建模思想仍然有效 |
| 缺点 | 数据和市场环境偏旧；主要是美股；官方实现基于 TensorFlow 1.x，工程维护成本较高 |
| 数据集 | NASDAQ、NYSE；历史 EOD 数据 + 行业关系 + Wiki 关系 |
| 代码情况 | 有官方代码，包含数据、预处理和训练脚本 |
| 训练成本 | 以今天标准看不高，单卡即可；但老框架会增加复现摩擦 |
| 准确性/结果 | 论文摘要报告在 NYSE/NASDAQ 回测中平均收益率分别达到 98% 和 71%；但评价体系与本赛题不完全一致 |
| 对本任务适配度 | 高，尤其适合作为排序目标设计的参考 |
| 是否建议复现 | **不建议作为首发主模型完整复现**，但非常建议吸收其 ranking loss 与关系建模思路 |

### 4.2 HIST

- 论文：HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information
- 时间：2021
- 代码：<https://github.com/Wentao-Xu/HIST>
- 论文摘要页：<https://www.emergentmind.com/papers/2110.13716>

| 维度 | 结论 |
|---|---|
| 特点 | 通过图结构同时利用预定义概念关系和隐藏概念关系，挖掘股票间共享信息 |
| 优点 | 直接面向 CSI100/CSI300；与 Qlib/Alpha360 兼容；很适合 A 股成分股任务 |
| 缺点 | 需要构建概念或图关系；工程复杂度明显高于表格模型或轻量 MLP |
| 数据集 | CSI100、CSI300；Qlib 的 Alpha360 特征 |
| 代码情况 | 有官方代码，包含多种 baseline 复现入口 |
| 训练成本 | 中等；通常需要单卡 GPU，但不属于超重模型 |
| 准确性/结果 | 官方代码可复现 HIST、MLP、LSTM、GAT、Transformer 等基线；原工作在 CSI 数据集上优于多种对照方法 |
| 对本任务适配度 | 很高，尤其适合做“股票间关系”增强 |
| 是否建议复现 | **建议重点参考**；可作为第二阶段图模型路线的重要候选 |

### 4.3 DoubleAdapt

- 论文：DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting
- 时间：2023
- 代码：<https://github.com/SJTU-DMTai/DoubleAdapt>
- KDD 页面：<https://www.kdd.org/kdd2023/wp-content/uploads/2023/08/toc.html>

| 维度 | 结论 |
|---|---|
| 特点 | 通过数据适配器与模型适配器来缓解金融时间序列的 concept drift / distribution shift |
| 优点 | 很适合金融市场非平稳问题；对真实比赛后段行情切换可能很有帮助 |
| 缺点 | 更像“训练框架增强器”而不是首个预测模型；调参更复杂；对 rank label 的支持也需要小心 |
| 数据集 | Qlib crowd-source data；支持 Alpha158/Alpha360；示例中支持 csi300 |
| 代码情况 | 有官方代码；有较详细的参数和资源说明 |
| 训练成本 | 中到高；官方说明在某些设置下需要约 8GB RAM、最高 10GB 显存；小步长实验耗时较长 |
| 准确性/结果 | 官方摘要称在真实股票数据上实现了 SOTA 级表现，并兼顾效率 |
| 对本任务适配度 | 中高，更适合在强基线或主力模型之后作为增强模块接入 |
| 是否建议复现 | **不建议首发复现**，建议在 baseline 和主力深度模型稳定后再接入 |

### 4.4 MASTER

- 论文：MASTER: Market-Guided Stock Transformer for Stock Price Forecasting
- 时间：2024
- 论文页：<https://ojs.aaai.org/index.php/AAAI/article/view/27767>
- 代码：<https://github.com/SJTU-DMTai/MASTER>

| 维度 | 结论 |
|---|---|
| 特点 | 利用市场状态做 gating 以动态筛选有效特征，并交替进行 intra-stock 与 inter-stock 聚合 |
| 优点 | 与 CSI300/CSI800 任务高度贴近；显式利用市场上下文；同时关注 RankIC 和组合收益指标 |
| 缺点 | 对数据组织和预处理较敏感；官方仓库后续披露过验证/测试数据导出流程问题，复现时需谨慎 |
| 数据集 | CSI300、CSI800；2008-2022；Alpha158 + 63 个市场特征；lookback=8，horizon=5 |
| 代码情况 | 有官方代码和预训练权重；也提供了处理后的数据说明 |
| 训练成本 | 中到高；论文使用 V100 16GB，最多训练 40 epochs，并做 5 次重复实验 |
| 准确性/结果 | 在 CSI300 上报告 IC=0.064、RankIC=0.076、AR=0.27、IR=2.4；在 CSI800 上也显著领先 |
| 对本任务适配度 | 很高，是最贴近本次比赛任务的 stock-specific 深度模型之一 |
| 是否建议复现 | **强烈建议复现**；优先级非常高 |

### 4.5 StockMixer

- 论文：StockMixer: A Simple yet Strong MLP-Based Architecture for Stock Price Forecasting
- 时间：2024
- 论文页：<https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/>
- 代码：<https://github.com/SJTU-DMTai/StockMixer>

| 维度 | 结论 |
|---|---|
| 特点 | 用更轻量的 MLP 结构完成 indicator mixing、time mixing、stock mixing |
| 优点 | 结构简单、训练更稳、参数效率高；官方明确强调在降低内存与运行成本的同时超过多种 SOTA |
| 缺点 | 主要在美股基准上验证；没有像 MASTER 那样直接在 CSI300/CSI800 上提供主结果 |
| 数据集 | NASDAQ、NYSE、S&P500；官方 repo 附带预处理后数据格式 |
| 代码情况 | 有官方代码和 supplementary material |
| 训练成本 | 低到中，是这批深度模型里性价比最高的一类 |
| 准确性/结果 | 官方摘要称其在真实股票基准上以显著优势超过多种 SOTA，同时降低 memory usage 和 runtime cost |
| 对本任务适配度 | 很高，尤其适合作为有限算力条件下的主力模型 |
| 是否建议复现 | **强烈建议复现**；建议作为第一主力深度模型 |

### 4.6 iTransformer

- 论文：iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
- 时间：2024
- 论文页：<https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html>
- 代码：<https://github.com/thuml/iTransformer>

| 维度 | 结论 |
|---|---|
| 特点 | 将变量而非时间点作为 token，擅长多变量相关性建模 |
| 优点 | 是通用多变量时序预测的强 backbone；适合迁移到多股票联合建模 |
| 缺点 | 不是股票专用模型；不直接解决排序、行业关系、组合构建问题 |
| 数据集 | 多个通用多变量时序预测基准 |
| 代码情况 | 有官方代码，生态成熟 |
| 训练成本 | 中等 |
| 准确性/结果 | 在通用多变量时序预测任务上达到强结果和较好的泛化能力 |
| 对本任务适配度 | 中等，需要做股票任务定制化改造 |
| 是否建议复现 | **不建议早期直接复现为主力模型**；更适合作为后续方法创新 backbone |

### 4.7 TimeXer

- 论文：TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables
- 时间：2024
- 代码：<https://github.com/thuml/TimeXer>
- 论文摘要页：<https://huggingface.co/papers/2402.19072>

| 维度 | 结论 |
|---|---|
| 特点 | 专门面向带外生变量的时序预测，将 endogenous 与 exogenous 信息分开建模再融合 |
| 优点 | 如果引入指数、行业、宏观、北向资金等外部特征，会很有参考价值 |
| 缺点 | 不是股票专用；仍需要自己补足排序头、组合构建和股票池逻辑 |
| 数据集 | 通用时序与带外生变量的预测基准 |
| 代码情况 | 有官方代码 |
| 训练成本 | 中等 |
| 准确性/结果 | 官方 repo 称其在 12 个 benchmark 上达到稳定强表现 |
| 对本任务适配度 | 中高，前提是项目确实引入较丰富的外生变量 |
| 是否建议复现 | **不建议首发复现**；适合作为多源数据增强后的后续研究方向 |

## 5. 综合比较结论

### 5.1 从“贴近本赛题”角度看

最贴近本赛题的工作是：

1. `MASTER`
2. `HIST`
3. `RSR`

原因：

- 明确关注股票间关系或市场上下文
- 更重视排序和投资结果
- 与成分股截面任务契合度高

### 5.2 从“算力性价比”角度看

最具性价比的工作是：

1. `StockMixer`
2. `LightGBM / XGBoost + Alpha 因子`
3. `MASTER`

原因：

- `StockMixer` 更轻、更稳、更容易训出可用结果
- 表格模型在量化任务中往往极强
- `MASTER` 虽然更重，但任务贴合度极高

### 5.3 从“创新空间”角度看

最适合作为创新切入点的是：

1. `MASTER` 的市场引导思想
2. `RSR` 的排序目标
3. `HIST` 的图关系建模
4. `DoubleAdapt` 的分布漂移适配
5. `TimeXer` 的外生变量建模

## 6. 建议的首发实验矩阵

我们的首发实验目标不是一次性追求最复杂模型，而是：

1. 先建立可靠闭环
2. 再验证最值得投入的深度模型
3. 最后把创新点压到最可能带来收益的模块

### 6.1 实验矩阵总表

| 实验编号 | 类型 | 模型 | 输入特征 | 目标 | 预期用途 | 优先级 |
|---|---|---|---|---|---|---:|
| E01 | 基线 | LightGBM | 基础量价 + 技术指标 | 5日收益回归 | 跑通全链路 | 1 |
| E02 | 基线增强 | LightGBM | Alpha158 风格特征 + 市场特征 | 5日收益回归 | 建立强传统基线 | 2 |
| E03 | 深度基线 | MLP / LSTM | 与 E02 相同 | 5日收益回归 | 验证深度模型下限 | 3 |
| E04 | 主力深度 | StockMixer | 个股特征 + 市场汇总特征 | 5日收益回归 | 首个主力模型 | 4 |
| E05 | 主力深度 | MASTER | Alpha158 + 市场信息 | 5日收益回归/排序 | 第二主力模型 | 5 |
| E06 | 图增强 | HIST 风格模型 | Alpha360 + 行业/概念关系 | 趋势/收益预测 | 股票关系增强 | 6 |
| E07 | 目标增强 | E04/E05 + ranking loss | 同上 | 排序优化 | 提升 Top-5 命中 | 7 |
| E08 | 训练增强 | DoubleAdapt + 主力模型 | 同上 | 增量适配 | 应对市场漂移 | 8 |
| E09 | 数据增强 | TimeXer / 自定义 exogenous 模型 | 加入指数/宏观/资金流 | 收益预测 | 外部信息增强 | 9 |
| E10 | 组合优化 | 任意高质量预测器 + 组合层 | 预测分数 | Top-5 组合收益 | 冲榜关键模块 | 10 |

### 6.2 每个实验的目标说明

#### E01：LightGBM 最小可运行版本

目的：

- 验证数据读取、标签构造、时间切分、提交生成全部正确

建议输入：

- 收益率特征
- 波动率特征
- 均线/MACD/振幅
- 相对沪深300超额收益

建议输出：

- 未来 5 个交易日收益率预测

成功标准：

- 稳定生成 `result.csv`
- 能完成滚动验证
- `RankIC` 和 `Top-5` 组合收益不为随机水平

#### E02：Alpha 因子强化版 LightGBM

目的：

- 建立一个足够强的传统基线，防止后续深度模型“复杂但不涨分”

建议输入：

- Alpha158 风格因子
- 指数上下文特征
- 行业 one-hot 或行业统计特征

成功标准：

- 显著超过 E01
- 成为后续深度模型需要超越的参考线

#### E03：简单深度基线

目的：

- 校验深度学习训练链路是否稳定
- 检查是否存在严重过拟合或标签错误

候选模型：

- MLP
- LSTM
- GRU

#### E04：StockMixer 首发深度模型

目的：

- 以较低成本获取比简单深度模型更强的结果

预期：

- 在有限算力下最可能成为第一个真正有竞争力的深度模型

#### E05：MASTER 首发高贴合模型

目的：

- 通过市场引导和截面相关建模，进一步提高排序质量和组合收益

预期：

- 更适合沪深300这类指数成分股场景

#### E06：HIST 风格图增强

目的：

- 引入股票间结构性关系，验证行业/概念图是否有额外价值

注意：

- 这一阶段工程成本会明显上升，应放在主力模型稳定后

#### E07：Ranking Loss 增强

目的：

- 把训练目标从“回归均值更准”推进到“Top-K 排名更准”

候选方案：

- Pairwise ranking loss
- Listwise ranking loss
- 回归头 + 排序头多任务联合训练

#### E08：DoubleAdapt 增强

目的：

- 缓解市场风格切换造成的泛化下降

适用条件：

- 当我们发现不同年份表现差异很大时

#### E09：外生变量增强

目的：

- 验证指数、行业、北向资金、宏观变量能否提供正交增益

#### E10：组合层优化

目的：

- 把预测分数转换成更强的最终提交结果

候选方案：

- 等权 Top-5
- 归一化 softmax 权重
- 温度缩放权重
- 不确定性过滤后再分配权重
- 行业分散约束

## 7. 优先复现顺序

### 第一梯队：必须先做

1. `LightGBM + Alpha 风格特征`
2. `StockMixer`
3. `MASTER`

原因：

- 这三条路线覆盖了“强传统基线 + 低成本强深度模型 + 高贴合深度模型”
- 能尽快形成可靠结果对比
- 适合当前比赛目标与算力条件

### 第二梯队：在主力模型稳定后做

1. `ranking loss`
2. `HIST`
3. `组合优化模块`

原因：

- 这几项最可能带来实战收益提升
- 但前提是基础预测器已经足够稳定

### 第三梯队：作为后续增强或创新储备

1. `DoubleAdapt`
2. `TimeXer`
3. `iTransformer` 改造版

原因：

- 这些方向更偏“研究型增强”
- 开发与验证成本更高
- 应在前两梯队完成后再投入

## 8. 当前建议的项目主路线

基于现有调研，当前最合理的主路线如下：

### 路线 A：强基线

- `LightGBM + Alpha158 风格特征 + 市场上下文特征`

### 路线 B：主力深度模型

- `StockMixer`

### 路线 C：高贴合增强模型

- `MASTER`

### 路线 D：冲榜增强

- `ranking loss`
- `组合构建优化`
- `HIST 风格关系增强`

## 9. 执行建议

接下来真正落地时，建议按以下顺序推进：

1. 先把数据字典、标签定义和时间切分严格写清楚
2. 先做 `E01-E02`，建立不可动摇的 baseline
3. 再上 `E04`，快速验证 `StockMixer`
4. 再上 `E05`，验证 `MASTER`
5. 最后围绕 `E07-E10` 做赛题导向优化

如果只有有限时间，我建议本项目最终优先投入：

1. `LightGBM baseline`
2. `StockMixer`
3. `MASTER`
4. `ranking loss + Top-5 组合优化`

## 10. 参考链接

- RSR code: <https://github.com/fulifeng/Temporal_Relational_Stock_Ranking>
- RSR abstract: <https://ideas.repec.org/p/arx/papers/1809.09441.html>
- HIST code: <https://github.com/Wentao-Xu/HIST>
- HIST abstract: <https://www.emergentmind.com/papers/2110.13716>
- DoubleAdapt code: <https://github.com/SJTU-DMTai/DoubleAdapt>
- DoubleAdapt KDD page: <https://www.kdd.org/kdd2023/wp-content/uploads/2023/08/toc.html>
- MASTER paper: <https://ojs.aaai.org/index.php/AAAI/article/view/27767>
- MASTER code: <https://github.com/SJTU-DMTai/MASTER>
- StockMixer paper: <https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/>
- StockMixer code: <https://github.com/SJTU-DMTai/StockMixer>
- iTransformer paper: <https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html>
- iTransformer code: <https://github.com/thuml/iTransformer>
- TimeXer code: <https://github.com/thuml/TimeXer>
- TimeXer abstract: <https://huggingface.co/papers/2402.19072>
