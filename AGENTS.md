# 协作指南（AGENTS.md）

## 项目概述

本项目是基于 [helme/ecg_ptbxl_benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking) 的 Fork，围绕 PTB-XL 心电图数据集进行深度学习基准测试。核心改动是新增了**纯推理管线**，支持从预训练权重和 PyTorch Lightning 检查点直接推理，无需重新训练。

## 目录结构

```
├── code/                         # 所有 Python 代码
│   ├── experiments/
│   │   └── scp_experiment.py     # 实验编排器（数据准备→训练/推理→评估）
│   ├── models/
│   │   ├── base_model.py         # 模型基类（fit / predict 接口）
│   │   ├── fastai_model.py       # fastai 封装，驱动大部分架构（resnet/inception/rnn）
│   │   ├── xresnet1d.py          # xresnet1d 模型定义
│   │   ├── inception1d.py        # inception1d 模型定义
│   │   ├── resnet1d.py           # resnet1d_wang 模型定义
│   │   ├── rnn1d.py              # LSTM/GRU 定义
│   │   ├── basic_conv1d.py       # 基础1D卷积模块
│   │   ├── timeseries_utils.py   # 时间序列数据集、裁剪、聚合
│   │   ├── wavelet.py            # 小波+NN 模型
│   │   └── your_model.py         # 自定义模型模板
│   ├── configs/
│   │   ├── fastai_configs.py     # 所有 fastai 模型的配置
│   │   ├── wavelet_configs.py    # 小波模型配置
│   │   └── your_configs.py       # 自定义配置模板
│   ├── utils/
│   │   ├── utils.py              # 数据加载、预处理、标准化、评估、阈值计算
│   │   ├── convert_ICBEB.py      # ICBEB 数据集格式转换
│   │   └── stratisfy.py         # 分层抽样
│   ├── reproduce_results.py      # 完整训练复现入口
│   ├── run_inference.py          # fastai 模型推理（skip_training=True）
│   ├── run_lightning_inference.py # Lightning 检查点推理（不依赖 fastai）
│   └── test_evaluate_exp0.py     # 对已有预测结果进行评估
├── tests/
│   └── test_utils.py             # 工具函数单元测试
├── output/                       # 训练输出（Git 忽略）
│   └── exp0/                     # 实验结果
│       ├── data/                 # y_train/val/test.npy, mlb.pkl, standard_scaler.pkl
│       └── models/
│           ├── naive/            # 基线预测（训练集均值）
│           ├── fastai_xresnet1d101/
│           │   ├── models/       # 模型权重 .pth
│           │   └── results/      # te_results.csv 等评估结果
│           └── ensemble/         # 集成预测（各模型均值）
├── data/                         # 数据集（Git 忽略）
│   └── ptbxl_clean_no_noise/
│       ├── ptbxl_database_clean_no_noise.csv
│       └── records100/
├── get_datasets.sh               # 数据集下载脚本
├── ecg_env.yml                   # Conda 环境（当前 Fork 版本）
├── ecg_env_original.yml          # 原始环境（封存）
├── README.md                     # 当前 README（中文）
└── README_original.md            # 原始 README（封存）
```

## 核心概念

### 1. 模型接口 (`code/models/base_model.py:2`)

所有模型必须实现：

```python
class ClassificationModel:
    def fit(self, X_train, y_train, X_val, y_val):
        """X: list of np.ndarray [T, C]; y: np.ndarray [N, n_classes] one-hot"""

    def predict(self, X, full_sequence=True) -> np.ndarray:
        """返回 [N, n_classes] 概率预测"""
```

**现有模型**都通过 `fastai_model.py` 封装，实际网络定义在 `xresnet1d.py`/`inception1d.py`/`resnet1d.py`/`rnn1d.py` 中。

### 2. 模型配置

配置格式为 `dict`，包含 `modelname`、`modeltype`、`parameters`：

```python
conf_fastai_xresnet1d101 = {
    'modelname': 'fastai_xresnet1d101',
    'modeltype': 'fastai_model',
    'parameters': dict()  # 传给模型构造函数的额外参数
}
```

`modeltype` 在 `scp_experiment.py` 中决定实例化哪个类。

### 3. 实验编排 (`scp_experiment.py`)

三步流程：
- **`prepare()`** — 加载数据、预处理、标准化、划分 train/val/test
- **`perform()`** — 训练（或加载权重推理）、生成预测 `.npy`
- **`evaluate()`** — 用预测和标签计算 macro_auc / Fmax 等指标

数据按 fold 划分：folds 1-8 训练，fold 9 验证，fold 10 测试。

## 关键工作流

### 工作流 A：完整训练（paper 复现）

```
reproduce_results.py
  → SCP_Experiment.prepare()
    → utils.load_dataset()              # 加载 WFDB 信号 + CSV 标签
    → utils.compute_label_aggregations() # 根据 task 类别聚合 SCP 编码
    → utils.select_data()               # 过滤 + one-hot 编码
    → utils.preprocess_signals()        # StandardScaler 标准化
  → SCP_Experiment.perform()
    → model.fit() / model.predict()     # 训练 + 预测（或加载权重）
    → 生成 y_train/val/test_pred.npy
  → SCP_Experiment.evaluate()
    → utils.generate_results()          # 计算 macro_auc、Fmax
    → 输出 te_results.csv
```

### 工作流 B：纯推理（本 Fork 新增）

```
run_inference.py
  → SCP_Experiment(skip_training=True)
  → prepare() 同上
  → perform()
    → _predict_with_pretrained()        # 从 .pth 加载权重
    → model.predict()                   # 生成预测
  → evaluate(bootstrap_eval=False)     # 不 bootstrap，直接单次评估
```

### 工作流 C：Lightning 检查点推理（不依赖 fastai）

```
run_lightning_inference.py
  → 手动加载数据 + 标准化器
  → load_lightning_checkpoint()         # 解析 state_dict，映射到 XResNetLightning
  → predict_with_lightning()            # 用 TimeseriesDatasetCrops 分块推理
  → 输出 y_test_pred_lightning.npy
```

**模型架构映射关系**：`scp_experiment.py` 中各种模型配置对应的是 fastai 版本的同一模型。输出目录结构相同：`output/<exp_name>/models/<modelname>/`

### 工作流 D：快速评估已有结果（不重新推理）

```
test_evaluate_exp0.py
  → SCP_Experiment.evaluate(bootstrap_eval=False)
  → 读取已有 y_test_pred.npy → 评估 → 打印结果
```

## 数据流

| 阶段 | 输入 | 输出 |
|---|---|---|
| 加载 | WFDB `.dat`/`.hea` + CSV | `X` (list of [T,12] ndarray), `raw_labels` (DataFrame) |
| 标签聚合 | `raw_labels`, task 名称 | `labels` (DataFrame with diagnostic_superclass 等列) |
| 数据筛选 | `X`, `labels`, min_samples | 按 fold 划分的 `X_train/val/test`, `y_train/val/test` (one-hot) |
| 标准化 | `X_train/val/test` | StandardScaler 后的标准化数据 |
| 推理 | 标准化数据 + 模型权重 | `y_*_pred.npy` |
| 评估 | `y_*_pred.npy`, `y_*.npy` | `te_results.csv` |

## 添加新模型

1. 在 `code/models/` 中创建 `your_new_model.py`，实现 `fit()` 和 `predict()`
2. 在 `code/configs/your_new_configs.py` 中定义配置
3. 在 `code/experiments/scp_experiment.py` 的 `perform()` 中添加 `elif modeltype == "your_type":`

## 测试

```bash
cd tests && pytest -q
```

CI 在 `.github/workflows/python-tests.yml`，只测试工具函数（`test_apply_thresholds`），不涉及重型的模型训练/推理，因为原始仓库在 GitHub Actions 上没有安装 fastai 和 pytorch。

如果需要本地跑完整的模型流程验证，应使用 conda 环境并确保 `data/` 目录已就绪。

## 代码约定

- **Python 3.8+**（conda 环境锁定 3.8.6）
- 无类型注解（原项目风格，不要加）
- 不加英文注释（原项目也不加）
- 推荐使用 `pathlib.Path` 而非 `os.path`（本 Fork 已有部分使用）
- `output/` 和 `data/` 目录加入 `.gitignore`，不要提交
- Fastai v1 兼容（`ecg_env.yml` 使用 `fastai=1.0.61`，`pytorch=1.4.0`）
- 推理脚本中有 Monkey-patch 来兼容 PyTorch ≥2.6（`weights_only=False`）

## 环境说明

| 文件 | 用途 |
|---|---|
| `ecg_env.yml` | 当前 Fork 环境（不含 torchvision） |
| `ecg_env_original.yml` | 原始仓库环境（含 torchvision，仅封存参考） |

## Git 协作注意事项

- **不要在 master 上直接推送**，使用分支 + PR
- 提交消息用中文简洁描述，格式：`类型: 描述`，如 `fix: 修复推理时类别数不匹配`、`feat: 添加新模型配置`
- **禁止提交 `output/`、`data/`、`__pycache__/`、`.ipynb_checkpoints/`、模型权重（`.pth`/`.ckpt`/`.pt`）**
- 原始 README 已封存到 `README_original.md`，当前 README 为中文版本
- 如果有需要持久化的配置文件，路径以 `../data/` 开头（相对于 `code/` 目录）

## 常用命令速查

```bash
# 环境
conda env create -f ecg_env.yml && conda activate ecg_env

# 数据
./get_datasets.sh

# 测试
cd tests && pytest -q

# 推理
cd code && python run_inference.py         # fastai 推理
cd code && python run_lightning_inference.py  # Lightning 推理

# 评估
cd code && python test_evaluate_exp0.py     # 快速评估

# 完整训练（需要 GPU，耗时数小时）
cd code && python reproduce_results.py
```
