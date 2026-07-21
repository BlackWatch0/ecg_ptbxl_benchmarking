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

## 原作者模型三测试域基准结果契约

入口为 `run_original_models_benchmark_colab.sh`。训练固定使用 clean PTB-XL folds 1-8，fold 9 选择 validation-loss 最优 checkpoint，fold 10 在 clean、mixed-noise 和 denoised 三个测试域评估。noisy/denoised 均包含 24、12、6、0、-6 dB，并且必须按 `ecg_id` 对齐。

数据下载地址统一维护在 `configs/datasets.json`，Shell 脚本不得再次硬编码 Drive ID。固定键为 `ptbxl_original`、`ptbxl_noisy`、`ptbxl_denoised`；每项必须包含 `role`、`url`、`drive_id`、`archive_name` 和 `format`。后续特征提取文件添加到 `feature_archives` 数组，建议每项沿用相同字段并增加唯一 `name` 和适用的 `scenario`，不得覆盖原始三数据集配置。

已登记的 Wavelet 特征归档为 `wavelet_feature_extraction`（`1mGZRk_SJ20miD8DNvK_BjGtQhoJsA60O`）。它当前仅作为配置资产登记，不能在未实现并验证 ID/feature-column 对齐前替代直接 ECG Wavelet 特征提取。旧脚本仍硬编码的待甄别链接见 `docs/DOWNLOAD_LINKS_REVIEW.md`；收到替代链接后才可迁移到统一配置。

必须包含七个原论文模型：`xresnet1d101`、`resnet1d_wang`、`lstm`、`lstm_bidir`、`fcn_wang`、`inception1d`、`wavelet_nn`。不得用 unsupported 状态或空行伪装 Wavelet+NN 已完成。

固定 Google Drive 根目录：

```text
/content/drive/MyDrive/ECG/original_models_benchmark/
```

原始运行产物组成：

```text
original_models_benchmark/
├── config/                 # resolved config、数据完整性、标准化器、smoke test
├── checkpoints/            # best/last checkpoint、thresholds、model_info
├── training_logs/          # 每模型逐 epoch train/valid loss 与 learning rate
├── features/wavelet_nn/    # 按 ID 对齐的 Wavelet 特征缓存
├── predictions/            # validation 和三测试域逐记录概率/预测
├── metrics/                # 三阈值整体、逐类、复杂度、完整性指标
├── errors/                 # 失败 traceback
├── completed_models.json
├── final_report/
└── original_models_benchmark_report.zip
```

必须保存三种阈值策略：`threshold_0.5`、`best_global_threshold`、`per_class_thresholds`。最终主比较使用 validation set 选择的 `per_class_thresholds`，阈值不得在 test set 上优化。

整体指标至少包含：`macro_roc_auc`、`micro_roc_auc`、`macro_pr_auc`、`micro_pr_auc`、`macro_f1`、`micro_f1`、`samples_f1`、`label_accuracy`、`exact_match_accuracy`、`predicted_positive_rate`、`mean_predicted_labels`、`all_zero_prediction_rate`。

逐类指标必须按 `NORM, MI, STTC, CD, HYP` 输出：`roc_auc`、`pr_auc`、`precision`、`recall`、`specificity`、`f1`、`support_positive`、`support_negative`、`predicted_positive_count`。

复杂度字段至少包含：`parameter_count`、`trainable_parameter_count`、`training_time_seconds`、`best_epoch`、`best_valid_loss`、`inference_time_per_sample_ms`、`actual_batch_size`。Wavelet+NN 的特征提取与分类器信息必须保留在配置和模型信息中。

`final_report/` 不得使用简化输出，必须包含：

```text
benchmark_summary.csv
clean_comparison.csv
noisy_snr_comparison.csv
denoised_snr_comparison.csv
denoising_contributions.csv
robustness_metrics.csv
mean_domain_metrics.csv
per_class_metrics.csv
model_complexity.csv
best_model_summary.json
ORIGINAL_MODELS_BENCHMARK_RESULTS.md
figures/*.png
figures/*.pdf
```

图表必须覆盖 noisy/denoised 的 Macro ROC-AUC 与 Macro F1 vs SNR、相对 clean 的下降、clean 与 -6 dB 逐类表现、参数量/推理时间权衡、每个模型 train/valid loss、noisy 与 denoised 对比。坐标和图例使用可读英文显示名，不能直接使用内部变量名作为最终轴标签。

最终 ZIP 固定为：

```text
/content/drive/MyDrive/ECG/original_models_benchmark/original_models_benchmark_report.zip
```

ZIP 包含 `final_report/`、`metrics/`、`predictions/`、`training_logs/` 和 `config/`。checkpoint 和大型 Wavelet 缓存保留在 Drive 根目录，不放入 ZIP，也不得提交到 Git。

## 添加新模型

1. 在 `code/models/` 中创建 `your_new_model.py`，实现 `fit()` 和 `predict()`
2. 在 `code/configs/your_new_configs.py` 中定义配置
3. 在 `code/experiments/scp_experiment.py` 的 `perform()` 中添加 `elif modeltype == "your_type":`
4. **同步更新根目录的 `model_architecture_summary.md` 和 `model_architecture_summary.json`。** 该要求适用于新增模型、可训练配置变体、factory/registry builder 注册项、attention、feature branch、fusion 或 classifier 的实质变更。
5. 架构报告必须基于实际构造函数、配置和训练入口追踪；记录模型定义/训练文件、输入输出 shape、主干、attention、特征分支、fusion、分类头、训练配置、参数量或无法实例化原因、以及静态风险。不要仅按文件名登记模型。
6. 如环境允许，使用随机小张量做一次不加载数据、不训练的 dummy `forward`；否则在报告中写明完整失败摘要。同步更新 Markdown 的人工说明和 JSON 的机器可读条目。
7. 为新模型提供 `evaluation/model_registry.py` 可使用的 factory 或 adapter，并通过统一 `evaluation/evaluate.py` 验证 CPU 单输入/feature-only/late-fusion 调用、严格 checkpoint 加载、输出类别数和标准结果 schema。不要为新模型另写一套不兼容的指标或测试目录。

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
