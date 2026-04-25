# Data Pipeline

## 目标

本项目的数据流水线目标是：

1. 将比赛原始股票日线数据统一整理到 `data/raw/`
2. 通过固定规则清洗与校验
3. 构造模型训练数据集并输出到 `data/processed/`
4. 保证后续 baseline 与深度模型可以共享同一份标准化输入

## 目录约定

```text
data/
├─ raw/        # 原始输入数据，不直接手改
├─ interim/    # 清洗或合并过程中的中间产物
└─ processed/  # 可直接用于训练的结构化数据
```

## 一、原始数据要求

当前 baseline 脚手架默认读取 `data/raw/` 下的多个 CSV 文件，并要求至少能映射出以下字段：

| 标准字段 | 含义 | 可兼容别名 |
|---|---|---|
| `date` | 交易日期 | `datetime`, `trade_date`, `日期` |
| `stock_id` | 股票代码 | `code`, `ts_code`, `股票代码` |
| `open` | 开盘价 | 无 |
| `high` | 最高价 | 无 |
| `low` | 最低价 | 无 |
| `close` | 收盘价 | 无 |
| `volume` | 成交量 | `vol`, `成交量` |

也就是说，如果你的比赛原始数据列名不是完全一致，只要落在这些兼容别名里，当前脚手架就能识别。

对于比赛基准代码导出的中文列，当前加载器已直接兼容：

| 基准字段 | 归一化后字段 |
|---|---|
| `股票代码` | `stock_id` |
| `日期` | `date` |
| `开盘` | `open` |
| `收盘` | `close` |
| `最高` | `high` |
| `最低` | `low` |
| `成交量` | `volume` |
| `成交额` | `amount` |
| `振幅` | `amplitude_pct` |
| `涨跌额` | `change_amount` |
| `换手率` | `turnover_rate_pct` |
| `涨跌幅` | `pct_chg` |

此外，当前代码也兼容：

- `UTF-8 with BOM`
- 日期格式 `YYYY/M/D`
- 股票代码 `600000` 或 `sh.600000`

## 二、推荐的原始数据组织方式

建议使用以下两种之一：

### 方式 A：每只股票一个 CSV

例如：

```text
data/raw/
├─ sh.600000.csv
├─ sz.000001.csv
├─ sz.000858.csv
└─ ...
```

### 方式 B：一个合并后的总表 CSV

例如：

```text
data/raw/
└─ daily_prices.csv
```

当前脚手架对两种方式都兼容，因为它会读取 `data/raw/` 下的所有 CSV 再拼接。

对于你提到的比赛基准代码，最推荐的放置方式是直接把它导出的主文件放进：

```text
data/raw/
└─ stock_data.csv
```

或者：

```text
data/raw/
└─ hs300_history_2015_2026.csv
```

二者当前都能直接被我们的数据入口读取。

## 三、当前已实现的清洗逻辑

在 [io.py](D:\data_competition\src\data\io.py) 中，当前已实现：

1. 自动识别常见列名别名
2. 自动兼容比赛基准代码的中文字段
3. 自动尝试 `utf-8-sig / utf-8 / gbk` 编码读取
2. 将 `date` 统一转为时间格式
3. 将价格和成交量转为数值格式
4. 删除关键价格缺失的行
5. 将 `sh.600000`、`sz.000001` 统一规整为 6 位数字代码
6. 按 `stock_id + date` 排序

## 四、当前已实现的特征工程

在 [alpha_factors.py](D:\data_competition\src\features\alpha_factors.py) 中，当前已实现第一版轻量特征：

- `ret_1`
- `open_to_close`
- `high_to_low`
- `amplitude`
- `ret_3 / ret_5 / ret_10 / ret_20`
- `volatility_3 / 5 / 10 / 20`
- `ma_ratio_*`
- `volume_ratio_*`
- 每日截面 `z-score`
- 每日截面分位数 `rank`

## 五、当前已实现的标签

在 [labels.py](D:\data_competition\src\features\labels.py) 中，当前默认主标签为：

- `y_ret_5d_open_open`

其定义是：

- 从 `T+1` 开盘买入
- 到 `T+5` 开盘卖出
- 收益率为 `(open[T+5] / open[T+1]) - 1`

这与比赛规则是直接对齐的。

## 六、构建数据集的命令

如果你希望直接在本仓库里调用与比赛基准逻辑兼容的数据下载脚本，可以先执行：

```powershell
python scripts/fetch_benchmark_data.py --data-dir data/raw --output-name stock_data.csv
```

常用可选参数：

```powershell
python scripts/fetch_benchmark_data.py `
  --data-dir data/raw `
  --output-name stock_data.csv `
  --start-date 2015-01-01 `
  --end-date 2026-02-28
```

这个脚本会：

- 登录 `baostock`
- 获取当前沪深300成分股列表
- 下载后复权日线数据
- 导出 `hs300_stock_list.csv`
- 导出 `stock_data.csv`
- 支持增量更新和失败记录

结合当前赛事通知，A 阶段第一次提交建议至少把数据补到：

- `2026-04-24`

推荐命令：

```powershell
python scripts/fetch_benchmark_data.py `
  --data-dir data/raw `
  --output-name stock_data.csv `
  --start-date 2015-01-01 `
  --end-date 2026-04-24
```

然后再运行：

在环境准备好后，运行：

```powershell
python scripts/build_dataset.py --config configs/baseline.yaml
```

成功后会生成：

- `data/processed/model_dataset.csv`

如果是本次 A 阶段第一次提交的特例数据集，请改用：

```powershell
python scripts/build_dataset.py --config configs/a_stage_round1.yaml
```

这是因为官方通知明确说明：

- `2026-04-25` 到 `2026-04-26` 这次提交中
- `T+5` 直接复用 `T+4` 数据

因此我们专门提供了：

- [a_stage_round1.yaml](D:\data_competition\configs\a_stage_round1.yaml)

它会在标签构造时采用：

- 正常卖出日偏移 `5`
- 当 `T+5` 不可用时回退到 `T+4`

如果你直接使用比赛基准代码下载的数据，那么推荐流程是：

1. 先用基准代码生成 `stock_data.csv` 或类似主数据文件
2. 放入 `data/raw/`
3. 再执行本项目的 `build_dataset.py`

这样我们既继承了比赛官方推荐的数据获取方式，也复用了当前项目的特征和训练流水线。

## 七、训练 baseline 的命令

```powershell
python scripts/train_baseline.py --config configs/baseline.yaml
```

成功后会生成：

- `outputs/predictions/baseline_predictions.csv`
- `outputs/predictions/baseline_metrics.json`
- `outputs/submissions/result.csv`

## 八、下一步数据增强计划

当基础日线数据跑通后，建议按以下顺序扩展：

1. 沪深300指数特征
2. 行业分类特征
3. 指数滚动统计特征
4. 少量资金流特征
5. Alpha158 风格更完整因子

## 九、当前最需要你提供的内容

为了真正开始数据处理，你需要把比赛原始数据放进：

- `D:\data_competition\data\raw\`

并尽量告诉我：

1. 原始文件数量和命名方式
2. 列名长什么样
3. 是否已有指数数据
4. 是否已经有成分股列表

一旦原始数据到位，我就可以继续帮你把字段映射和数据清洗规则精确改到比赛数据上。
