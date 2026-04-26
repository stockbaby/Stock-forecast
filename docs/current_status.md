# Current Status

## 1. 项目当前状态

仓库已经从“方案讨论”进入“可训练、可评估、可提交”的阶段。

已完成：
- 比赛基准数据抓取与扩展更新
- 数据清洗、字段标准化、训练数据集构建
- Alpha 风格特征、市场上下文特征、截面特征
- `LightGBM / MLP / LSTM / StockMixer` 训练
- 组合构建与提交文件校验
- 赛题公式收益回看
- 官方代码规范对齐的 `app/` 目录结构

## 2. 当前模型结果

### LightGBM baseline

- `RankIC = 0.0210`
- `Precision@5 = 0.0767`
- `Top-k portfolio return = -0.00157`

特点：
- 排序能力稳定
- 仍然是重要参照线
- 单独作为最终主提交稿时，当前不如强化版 StockMixer

### MLP baseline

- `RankIC ≈ -0.0011`
- `Precision@5 ≈ 0.0183`
- 表现明显弱于树模型与后续深度模型

结论：
- 作为对照实验保留
- 不再作为主线继续投入

### LSTM baseline

- `RankIC = 0.0136`
- `Precision@5 = 0.0356`
- `Top-k portfolio return = 0.00603`

特点：
- 整体排序不如 LightGBM
- 组合收益一度优于 LightGBM
- 是进入时序深度模型阶段的重要过渡模型

### 强化版 StockMixer

- `RankIC = 0.0591`
- `Precision@5 = 0.0462`
- `Top-k portfolio return = 0.01033`
- 最优策略：`proportional_positive_thr0.0`
- 验证期均值：`0.01085`

结论：
- 当前最强单模型
- 当前主提交稿优先级最高

### MASTER

- `RankIC = 0.0348`
- `Precision@5 = 0.0725`
- `Top-k portfolio return = 0.01762`
- 最优策略：`proportional_positive_thr0.0`
- 验证期均值：`0.01831`

结论：
- 当前验证集综合表现最强的模型
- 已经具备作为正式主力模型的价值
- 但在最近单窗口本地回看里，略低于强化版 `StockMixer`

## 3. 集成阶段结论

已实现：
- 分数归一化
- 网格搜索权重
- recent-window 验证
- 候选提交导出

当前 best ensemble：
- 权重：`LightGBM 0.4 + LSTM 0.6 + StockMixer 0.0`
- `RankIC = 0.0290`
- `Precision@5 = 0.0791`
- `Top-k portfolio return = 0.01013`

结论：
- 集成在部分验证指标上更稳
- 但当前最近窗口真实回看仍未超过强化版 StockMixer
- 因此暂不作为主提交方案

## 4. 关于 recent-biased objective 的结论

已尝试：
- 在 StockMixer 训练中加入更强的近期样本偏置
- 在集成选择中加入 recent-decay 验证打分

结果：
- 训练层面的 recent bias 会显著拉差 StockMixer
- 选择层面的 recent-aware 候选筛选可以保留

结论：
- 不再继续强化训练层 recent bias
- 后续只在验证与候选管理层使用“近期偏好”

## 5. 当前建议的提交优先级

### 第一优先级

- `outputs/submissions/result_stockmixer_alpha.csv`

### 第一优先级并列备选

- `outputs/submissions/result_master_alpha.csv`

### 第二优先级

- `outputs/submissions/ensemble_candidates/candidate_1.csv`
- `outputs/submissions/ensemble_candidates/candidate_2.csv`
- `outputs/submissions/ensemble_candidates/candidate_3.csv`

用途：
- 候选对照
- 提交前最后复查与切换参考

## 6. 下一步可继续推进的方向

### A. 官方 baseline 原仓复现

现在最值得补的一步是：
- 把赛题基准代码原样跑一次
- 在相同数据窗口下做更严格的数值对照

### B. 更贴比赛的目标函数

可以做，但建议控制强度，不再明显 recent-bias：
- 超额收益回归
- pairwise ranking loss
- listwise ranking loss
- 回归 + 排序混合目标

### C. 提交管理与稳健性

这条线很务实，也很值得：
- 固定保留多个候选提交
- 做更标准的 rolling retrain
- 提交前自动回看最近窗口
- 固定生成“主稿 + 备选稿”

### D. 额外结构化数据

若比赛规则允许且时间足够，可考虑：
- 行业分类
- 北向资金
- 资金流
- 低频财务/估值因子

## 7. 我们目前最重要的判断

当前这个仓库已经具备：
- 研究闭环
- 工程闭环
- 提交闭环

也就是说，后续工作不再是“从零到一搭框架”，而是：
- 选择更高价值的新方向
- 控制试错成本
- 围绕比赛得分做更精细的优化
## 8. 官方 Baseline 同仓结果

官方 baseline 原仓复现已经完成，且已经按同一回看口径和我们当前结果做了严格对比。

对比窗口：
- 参考交易日 `T = 2026-04-17`
- 买入日 `2026-04-20`
- 卖出日 `2026-04-24`

结果如下：

| Method | Local Return | vs Official |
|---|---:|---:|
| Official baseline | 0.02350 | 0.00000 |
| LightGBM | 0.03386 | +0.01036 |
| MASTER | 0.06097 | +0.03747 |
| StockMixer | 0.06332 | +0.03982 |

这意味着：
- 当前所有主力提交稿都已超过官方 baseline。
- `StockMixer` 仍是当前最近窗口本地回看最强稿。
- `MASTER` 仍是当前验证集综合最强稿。
