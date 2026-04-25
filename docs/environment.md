# Environment Setup

## 目标环境

本项目建议使用独立的 Conda 环境，环境名称统一为：

- `stock-forecast`

推荐版本：

- Python `3.11`

## 一、创建环境

如果你的终端已经能直接使用 `conda`，在仓库根目录执行：

```powershell
conda env create -f environment.yml
conda activate stock-forecast
```

如果后续更新了 `environment.yml`，可执行：

```powershell
conda env update -f environment.yml --prune
conda activate stock-forecast
```

## 二、验证环境

激活环境后，建议至少验证以下命令：

```powershell
python --version
python -c "import pandas, numpy, yaml, sklearn; print('ok')"
python -c "import lightgbm; print(lightgbm.__version__)"
```

如果你计划接入 AkShare：

```powershell
python -c "import akshare; print(akshare.__version__)"
```

## 三、如果终端无法识别 conda

当前项目环境里已经观察到一种常见情况：

- 终端里输入 `conda --version` 提示找不到命令

这通常说明：

1. 本机没有安装 Conda
2. 已安装，但没有加入 PATH
3. 当前终端没有加载 Conda 初始化脚本

建议按顺序排查：

### 方案 A：先查 Conda 是否已安装

常见路径包括：

- `C:\Users\<你的用户名>\miniconda3`
- `C:\Users\<你的用户名>\anaconda3`
- `D:\miniconda3`
- `D:\anaconda3`

如果找到了，通常可以直接调用：

```powershell
<conda安装目录>\Scripts\conda.exe --version
```

例如：

```powershell
C:\Users\Administrator\miniconda3\Scripts\conda.exe --version
```

### 方案 B：手动初始化 PowerShell

如果能找到 `conda.exe`，可以执行：

```powershell
<conda安装目录>\Scripts\conda.exe init powershell
```

然后重启终端，再执行：

```powershell
conda activate stock-forecast
```

### 方案 C：临时用完整路径创建环境

即使 PATH 没配好，也可以先用完整路径：

```powershell
<conda安装目录>\Scripts\conda.exe env create -f environment.yml
<conda安装目录>\Scripts\conda.exe activate stock-forecast
```

## 四、环境文件说明

项目根目录中的 [environment.yml](D:\data_competition\environment.yml) 是主环境定义文件。

当前依赖分为三类：

1. 基础数值与表格库
   - `numpy`
   - `pandas`
   - `PyYAML`
2. baseline 模型库
   - `scikit-learn`
   - `lightgbm`
3. 数据获取与分析辅助
   - `akshare`
   - `jupyterlab`
   - `ipykernel`

## 五、后续扩展原则

后续新增依赖时，请遵循：

1. 优先写入 `environment.yml`
2. 避免随手 `pip install` 后不记录
3. 优先保持环境轻量，符合比赛算力限制

## 六、当前与比赛约束的关系

比赛明确限制：

- 训练时间不超过 `8` 小时
- 预测时间不超过 `5` 分钟
- 机器资源为 `16GB RAM + 8GB VRAM`

因此环境配置也遵循轻量原则：

- 不预装重型深度学习生态
- 先围绕数据处理、树模型 baseline 和轻量研究展开
- 深度学习依赖可在进入 `StockMixer`/`MASTER` 阶段后再追加
