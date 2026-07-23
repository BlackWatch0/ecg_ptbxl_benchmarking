# ECG PTB-XL Benchmarking

本仓库基于 [helme/ecg_ptbxl_benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking)，用于 PTB-XL 多标签心电分类的训练、推理、标准评估与噪声鲁棒性实验。上游原始说明保存在 [README_original.md](README_original.md)。

## 当前状态

- 数据划分遵循 PTB-XL folds 1-8 训练、fold 9 验证、fold 10 测试。
- `code/` 保留训练、推理、数据准备和报告入口，`evaluation/` 提供统一评估管线。
- 原论文七模型基准覆盖 xResNet1D-101、ResNet1D-Wang、LSTM、BiLSTM、FCN-Wang、Inception1D 和 Wavelet+NN。
- 原有 9 个 Shell 已移至 [`scripts/legacy/`](scripts/legacy/README.md)，仅供追溯，不是当前入口，也不会由 taskmanager 调用。
- taskmanager 通过 `taskmanager.py` 编排原作者七模型的数据准备、训练、评估、报告和可移植 ZIP；配置与限制见 [`docs/TASK_MANAGER.md`](docs/TASK_MANAGER.md)。
- EMD late-fusion **当前 blocked**：`configs/datasets.json` 中没有可用 EMD 归档，依赖 EMD 的训练、消融和评估不得视为可运行工作流。

## 文档

| 文档 | 内容 |
|---|---|
| [任务管理](docs/TASK_MANAGER.md) | taskmanager 命令、配置、状态与执行边界 |
| [YAML 参数参考](docs/TASK_MANAGER_CONFIG.md) | 所有可选字段、默认值、优先级与 task 参数 |
| [依赖与环境](docs/DEPENDENCIES.md) | AWS 训练依赖、版本固定、GPU/CPU 隔离和环境检查 |
| [AWS 部署](docs/AWS_SETUP.md) | EC2、加密 gp3 EBS、SSM、环境和人工 S3 同步 |
| [数据资产](docs/DATA_ASSETS.md) | 数据来源、目录契约、对齐要求和 blocked 资产 |
| [工作流](docs/WORKFLOWS.md) | 当前 Python 入口、smoke、resume、日志和输出 |
| [下载链接复核](docs/DOWNLOAD_LINKS_REVIEW.md) | 配置中已登记与待补充的外部资产 |
| [统一评估](evaluation/README.md) | 标准化模型评估配置与结果 schema |
| [旧指南](docs/archive/) | 已归档的 Colab、EMD 和消融说明，不代表当前支持状态 |

## 本地环境

当前训练环境可按版本约束文件创建：

```bash
conda env create -f environments/ecg-training.yml
conda activate ecg-training
python -m pip check
python code/check_training_environment.py --require-cuda --check-compute
```

该环境固定 Python 3.10.16、PyTorch 2.5.1/CUDA 12.1，并使用 CPU-only TensorFlow 2.15.1 运行 Wavelet+NN。完整版本和验证方式见 [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md)。旧 fastai v1 环境保存在 `environments/legacy/`，只用于对应历史入口，不要与当前训练环境混用。

## Taskmanager

```bash
python taskmanager.py list-models
python taskmanager.py validate --config configs/taskmanager/original_models_benchmark.yaml
python taskmanager.py run --config configs/taskmanager/original_models_benchmark.yaml --dry-run
python taskmanager.py run --config configs/taskmanager/original_models_benchmark.yaml
```

先提交并复核 YAML，再执行 `--dry-run`。taskmanager 只接受本地路径，不调用 legacy Shell，也不负责 S3 同步。

独立于训练的统一评估从仓库根目录运行：

```bash
python evaluation/evaluate.py --config configs/evaluation/default.yaml
```

配置中的数据、checkpoint 和输出路径必须先改为本机实际路径。训练和完整基准入口见 [`docs/WORKFLOWS.md`](docs/WORKFLOWS.md)。

## 数据与产物

数据与运行产物默认位于 `data/`、`output/`、`results/`，均不应提交 Git。外部资产登记以 [`configs/datasets.json`](configs/datasets.json) 为准；“已登记”不等于已经下载、校验或可直接用于某个模型。

AWS 运行采用本地 EBS 工作目录。S3 仅通过操作者显式执行 `aws s3 sync` 传入数据和传出结果，代码与 taskmanager 都不提供原生 S3 数据层。

## 测试

```bash
python -m pytest -q tests
```

## 引用

- Strodthoff et al., *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL*, IEEE JBHI, 2021. <https://doi.org/10.1109/JBHI.2020.3022989>
- Wagner et al., *PTB-XL, a large publicly available electrocardiography dataset*, Scientific Data, 2020. <https://doi.org/10.1038/s41597-020-0495-6>
