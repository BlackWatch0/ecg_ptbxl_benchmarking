# 协作指南（AGENTS.md）

## 项目概述

本仓库基于 `helme/ecg_ptbxl_benchmarking`，用于 PTB-XL 多标签心电分类、噪声鲁棒性和去噪收益基准。当前代码同时包含原始 FastAI 复现路径、直接 PyTorch 训练、attention/特征融合消融、Wavelet+NN、独立检查点推理和统一评估；不要把任一单独路径描述成全部模型的唯一实现。

标准数据划分为 folds 1-8 训练、fold 9 验证、fold 10 测试。测试域包括 clean、mixed-noise 和 denoised，所有跨域比较必须按 `ecg_id` 对齐，阈值只能由 validation set 选择。

## 目录结构

```text
├── code/
│   ├── configs/                    # 旧 SCP/FastAI 与模型配置
│   ├── experiments/                # SCP_Experiment 旧实验编排
│   ├── models/                     # PyTorch、FastAI adapter、Lightning、Wavelet 模型
│   ├── task_manager/               # canonical taskmanager 核心
│   ├── time_domain_robustness/     # 时域特征鲁棒性分析
│   ├── utils/                      # 数据、标签、标准化和特征工具
│   ├── run_original_models_benchmark.py
│   ├── run_ablation_study.py
│   ├── run_wavelet_ablation_study.py
│   ├── run_inference.py
│   └── run_lightning_inference.py
├── configs/                        # 数据资产、消融和统一评估配置
│   └── taskmanager/                # 可跟踪的 taskmanager YAML 配置
├── evaluation/                     # 与训练解耦的统一评估、adapter、指标和报告
├── docs/TASK_MANAGER_CONFIG.md     # taskmanager YAML 全量参数参考
├── environments/
│   ├── ecg-training.yml            # Python 3.10 直接训练环境
│   └── legacy/                     # Python 3.8/FastAI v1 历史环境快照
├── tests/                          # 轻量单元、模型 forward 和评估契约测试
├── data/                           # 本地数据，不提交
├── output/                         # 旧实验输出，不提交
├── results/                        # 统一评估输出，不提交
├── scripts/legacy/                 # 历史 Colab/Shell 入口
├── taskmanager.py                  # taskmanager CLI 入口
└── model_architecture_summary.{md,json}
```

## 模型与接口

仓库存在多种真实入口，修改前必须沿训练脚本、factory/adapter 和配置追踪实际调用：

- `code/models/base_model.py` 的 `ClassificationModel.fit/predict` 是旧 SCP 实验接口。
- `code/models/fastai_model.py` 只服务旧 FastAI v1 路径，不代表全部模型。
- `code/models/original_model_factory.py` 直接构建原论文六个 raw-waveform PyTorch 模型；Wavelet+NN 使用独立特征提取和 TensorFlow/Keras 分类器。
- `code/models/cbam_xresnet1d.py` 及相关文件支持 ECG-only、CBAM、SE、feature-only 和 late-fusion 变体。
- `evaluation/model_registry.py` 是训练外统一评估的 canonical model adapter/registry。

模型输出必须保持 `[N, n_classes]`。PTB-XL superdiagnostic 的固定类别顺序为 `NORM, MI, STTC, CD, HYP`；不得依赖字典、文件系统或编码器的偶然顺序。

## 主要工作流

### 原模型三域基准

canonical 入口为根目录 `taskmanager.py` 和 `configs/taskmanager/original_models_benchmark.yaml`，核心训练器为 `code/run_original_models_benchmark.py`。训练只使用 clean folds 1-8，fold 9 按 validation loss 选择 checkpoint，fold 10 在 clean、noisy 和 denoised 上评估。`scripts/legacy/run_original_models_benchmark_colab.sh` 仅为历史归档。

### Attention 与特征融合消融

当前消融仍由 `code/run_ablation_study.py`、`code/run_wavelet_ablation_study.py` 及 `configs/ablation_*.yaml` 直接驱动，尚未纳入 taskmanager。`scripts/legacy/` 下的同名 Shell 仅供追溯，不是受支持入口。配置、输入特征 schema、fusion mode 和 checkpoint 必须一起验证。

### 统一评估

`evaluation/evaluate.py` 使用 `configs/evaluation/*.yaml`，不调用训练。新模型优先接入 `evaluation/model_registry.py`，并使用统一结果 schema、严格 checkpoint 加载、固定类别顺序和 validation-derived thresholds。

### 旧复现与推理

`code/reproduce_results.py`、`code/run_inference.py` 和 `code/experiments/scp_experiment.py` 保留原始 SCP/FastAI 工作流。`code/run_lightning_inference.py` 独立加载 Lightning checkpoint。只有维护这些入口时才使用 `environments/legacy/` 环境。

## 数据与配置

- 数据集下载元数据的唯一来源是 `configs/datasets.json`；Shell 脚本不得再次硬编码 Drive ID。
- 固定数据键为 `ptbxl_original`、`ptbxl_noisy`、`ptbxl_denoised`，每项包含 `role`、`url`、`drive_id`、`archive_name` 和 `format`。
- 新特征归档添加到 `feature_archives`，必须有唯一 `name`、适用 `scenario` 和对齐说明，不得覆盖三项原始数据配置。
- 已登记的 `wavelet_feature_extraction` 只能在验证 `ecg_id` 和 feature-column 对齐后使用。
- 训练标准化器只能在 clean folds 1-8 拟合，并原样用于 validation 和全部 test domain。
- `data/`、`output/`、`results/`、checkpoint、预测数组和本地缓存不得提交。

## Canonical Taskmanager 约定

- 根目录 `taskmanager.py` 是 CLI 入口，`code/task_manager/` 是唯一核心实现；不要在 `scripts/`、`docs/` 或临时目录复制第二套 runner。
- 可复现任务定义放在 `configs/taskmanager/*.yaml`，属于可跟踪输入，不能被 `.gitignore` 的宽泛规则吞掉。
- taskmanager 只编排现有 Python 入口，不调用 `scripts/legacy/`，也不复制 Colab、Google Drive 或 S3 传输逻辑。
- 每次运行必须使用显式 `output_dir`。当前运行状态为 `<output_dir>/task_status.json`，日志为 `<output_dir>/task_logs/`，resolved config 位于 `<output_dir>/config/resolved_config.yaml`。dry-run 单独写 `task_plan.json` 和 `task_plan_logs/`。
- taskmanager 状态和日志是本地生成物并由 Git 忽略；源 YAML 配置和需要持久化的结论必须保留在受控产物中。
- 除非任务明确要求，不修改 `taskmanager.py`、`code/task_manager/`、状态格式或调度语义。

## 原模型三测试域结果契约

noisy/denoised 均必须包含 24、12、6、0、-6 dB，并与 clean fold 10 按 `ecg_id` 严格对齐。必须包含七个原论文模型：`xresnet1d101`、`resnet1d_wang`、`lstm`、`lstm_bidir`、`fcn_wang`、`inception1d`、`wavelet_nn`；不得用 unsupported 状态或空行代替 Wavelet+NN 结果。

taskmanager 运行根目录由 YAML 的 `output_dir` 显式指定；本地、EBS 和历史 Drive 路径不得写死在训练代码中：

```text
<output_dir>/
```

运行产物必须包含：

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

必须保存 `threshold_0.5`、`best_global_threshold`、`per_class_thresholds` 三种策略。最终主比较使用 validation set 选择的 `per_class_thresholds`，禁止在 test set 优化阈值。

整体指标至少包含：`macro_roc_auc`、`micro_roc_auc`、`macro_pr_auc`、`micro_pr_auc`、`macro_f1`、`micro_f1`、`samples_f1`、`label_accuracy`、`exact_match_accuracy`、`predicted_positive_rate`、`mean_predicted_labels`、`all_zero_prediction_rate`。

逐类指标按 `NORM, MI, STTC, CD, HYP` 输出：`roc_auc`、`pr_auc`、`precision`、`recall`、`specificity`、`f1`、`support_positive`、`support_negative`、`predicted_positive_count`。

复杂度字段至少包含：`parameter_count`、`trainable_parameter_count`、`training_time_seconds`、`best_epoch`、`best_valid_loss`、`inference_time_per_sample_ms`、`actual_batch_size`。Wavelet+NN 的特征提取和分类器信息必须同时写入配置与模型信息。

`final_report/` 必须包含：

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

图表必须覆盖 noisy/denoised 的 Macro ROC-AUC 与 Macro F1 vs SNR、相对 clean 的下降、clean 与 -6 dB 逐类表现、参数量/推理时间权衡、每个模型 train/valid loss、noisy 与 denoised 对比。坐标和图例使用可读英文显示名，不能直接暴露内部变量名。

最终 ZIP 位于当前运行根目录：

```text
<output_dir>/original_models_benchmark_report.zip
```

ZIP 包含 `final_report/`、`metrics/`、`predictions/`、`training_logs/` 和 `config/`。checkpoint 与大型 Wavelet 缓存保留在运行根目录，不放入 ZIP，也不得提交到 Git。

## 添加或修改模型

1. 明确实际训练入口、输入 shape、类别顺序、loss 和输出语义，不按文件名猜测架构。
2. 提供 `evaluation/model_registry.py` 可用的 factory 或 adapter，验证 CPU single-input、feature-only 或 late-fusion 调用以及严格 checkpoint 加载。
3. 对随机小张量执行不加载数据、不训练的 dummy forward；验证输出类别数和有限值。
4. 新模型、可训练配置变体、factory/registry 注册项、attention、feature branch、fusion 或 classifier 的实质变更，必须同步更新 `model_architecture_summary.md` 和 `model_architecture_summary.json`。
5. 架构摘要必须记录定义/训练文件、输入输出 shape、主干、attention、特征分支、fusion、分类头、训练配置、参数量或实例化失败原因和静态风险。

## 环境

| 文件 | 用途 |
|---|---|
| `environments/ecg-training.yml` | Python 3.10、PyTorch CUDA、Wavelet 和 TensorFlow/Keras 的当前训练环境 |
| `environments/legacy/ecg-fastai-py38-no-torchvision.yml` | Fork 历史 FastAI v1 环境快照 |
| `environments/legacy/ecg-fastai-py38-original.yml` | 含 torchvision 的原仓库 FastAI v1 环境快照 |

创建当前环境：

```bash
conda env create -f environments/ecg-training.yml
conda activate ecg-training
```

legacy 文件用于复现，不是当前开发环境，不应把其中 Python 3.8、PyTorch 1.4 或 FastAI v1 约束传播到新代码。

## 测试

从仓库根目录运行：

```bash
python -m pytest -q tests
```

显式测试 `tests/` 可避免收集旧运行脚本 `code/test_evaluate_exp0.py`。`code/__init__.py` 暴露标准库同名模块的交互式 API，避免 FastAI/PDB 导入时发生名称冲突。CI 位于 `.github/workflows/python-tests.yml`，监听 `master` 和 `main` 的 push/PR，使用 Python 3.10 与 CPU PyTorch。测试包含工具函数、数据契约、模型 dummy forward、benchmark/report 和统一评估，不运行完整数据训练。

修改训练/评估代码后至少运行相关测试；修改共享模型 factory、数据契约或报告 schema 时运行完整测试。需要真实 PTB-XL、GPU 或大型 checkpoint 的验证必须明确记录为本地/Colab 验证，不能伪装成 CI 覆盖。

## 代码与 Git 协作

- 新代码以 Python 3.10 为基线；修改旧 FastAI 文件时保持其兼容范围和局部风格。
- 优先使用 `pathlib.Path`。旧模块不强制类型注解，`evaluation/` 等现代模块沿用现有类型风格。
- 不提交 `data/`、`output/`、`results/`、`__pycache__/`、`.ipynb_checkpoints/`、模型权重或运行时 taskmanager 状态/日志。
- 不直接推送 `master`；使用分支和 PR。提交消息采用简洁中文 `类型: 描述` 格式。
- 不修改与当前任务无关的用户改动，不用生成物覆盖手工配置，不在测试集上调参或选阈值。
