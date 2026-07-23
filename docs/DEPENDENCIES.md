# 依赖与环境

## 支持范围

当前 AWS 训练环境服务以下现代工作流：

- 根目录 `taskmanager.py`；
- 七个原论文模型的训练、三域评估、报告和打包；
- `evaluation/` 统一评估；
- CBAM/SE/Wavelet 直接 PyTorch 消融；
- 仓库 `tests/`。

FastAI v1、PyTorch 1.4 和历史推理不属于当前 AWS GPU 环境。对应快照保存在 `environments/legacy/`，不得安装到 `ecg-training`。

## Canonical 环境

AWS 唯一推荐的环境文件是 `environments/ecg-training.yml`。

| 组件 | 版本 | 用途 |
|---|---|---|
| Python | `3.10.16` | 当前训练和评估基线 |
| PyTorch | `2.5.1` | 六个 raw-waveform 模型和 attention/fusion 模型 |
| PyTorch CUDA runtime | `12.1` | EC2 NVIDIA GPU；由 Conda 环境提供 |
| NumPy | `1.26.4` | 与 TensorFlow 2.15 兼容，禁止升级到 NumPy 2 |
| Pandas | `2.2.3` | metadata、日志、指标和报告 |
| SciPy | `1.13.1` | Wavelet 特征统计 |
| scikit-learn | `1.5.2` | 标准化、标签和指标 |
| Matplotlib | `3.9.2` | headless 报告和图表 |
| PyYAML | `6.0.2` | taskmanager/evaluation YAML |
| PyWavelets | `1.7.0` | db6 level-5 特征提取 |
| WFDB | `4.1.2` | PTB-XL waveform 读取 |
| tqdm | `4.66.5` | 数据读取进度 |
| TensorFlow CPU | `2.15.1` | Wavelet+NN Keras 分类器 |
| pytest | `8.3.5` | AWS 上线前测试 |

TensorFlow 必须使用 `tensorflow-cpu`。不要在同一环境安装 `tensorflow`、`tensorflow[and-cuda]`、`tensorflow-gpu` 或额外 CUDA/cuDNN wheel。

## GPU 与 TensorFlow 隔离

PyTorch 使用 A10G GPU 和 Conda 提供的 CUDA 12.1 用户态运行库。Wavelet+NN 的特征提取和小型 Keras MLP 使用 CPU。

taskmanager 对 `wavelet_nn` 子进程自动设置：

```text
--device cpu
CUDA_VISIBLE_DEVICES=-1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

这样可以避免 TensorFlow 与 PyTorch 竞争显存、避免 TensorFlow 2.15 加载另一套 CUDA/cuDNN，并防止多进程 Wavelet 提取与 BLAS 线程相乘造成 CPU 过度订阅。提取器默认使用 `min(18, os.cpu_count())` 个进程；六个 PyTorch 模型仍使用 YAML 中的 `device: cuda`。

## 安装

在 AWS DLAMI 或已安装 NVIDIA driver 的 Ubuntu 22.04 上执行：

```bash
cd /mnt/ecg/workspace/ecg_ptbxl_benchmarking

conda env create --solver libmamba -f environments/ecg-training.yml
conda activate ecg-training

python -m pip check
python code/check_training_environment.py \
  --require-cuda \
  --check-compute \
  --output /mnt/ecg/runs/environment-check.json
```

不需要安装系统 CUDA Toolkit。`pytorch-cuda=12.1` 已提供用户态 CUDA runtime；主机只需要版本足够新的 NVIDIA driver。`nvidia-smi` 显示的是驱动支持的最高 CUDA 版本，不要求文本恰好显示 12.1。

## Pip requirements

`environments/requirements/` 是按用途拆分的精确版本清单：

| 文件 | 内容 |
|---|---|
| `core.txt` | 不含 PyTorch 的科学计算、数据、报告和 YAML 依赖 |
| `wavelet.txt` | `tensorflow-cpu` |
| `test.txt` | pytest |

这些文件主要供 CI、故障恢复和依赖审计使用。AWS GPU 正式环境应优先从 Conda YAML 创建，因为 PyTorch CUDA 不能只靠通用 requirements 文件表达。

若在已有 DLAMI Conda 环境中手工安装，先用官方 Conda channel 安装 PyTorch/CUDA，再安装其余 requirements：

```bash
conda create -n ecg-training -c pytorch -c nvidia -c conda-forge \
  python=3.10.16 pytorch=2.5.1 pytorch-cuda=12.1 pip
conda activate ecg-training

python -m pip install -r environments/requirements/core.txt
python -m pip install -r environments/requirements/wavelet.txt
python -m pip install -r environments/requirements/test.txt
python -m pip check
```

不要再执行 `pip install torch`，否则可能用 PyPI wheel 覆盖 Conda CUDA 构建。

## 环境验证

主机层：

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

Python 层：

```bash
python code/check_training_environment.py --require-cuda --check-compute
```

检查器验证 Python 和关键包版本、PyTorch CUDA 12.1、GPU compute、`tensorflow-cpu` 分发包、TensorFlow 不可见 GPU，以及一次 Keras train step。

随后运行：

```bash
python -m pytest -q tests
python taskmanager.py validate \
  --config configs/taskmanager/aws_original_models.example.yaml
python taskmanager.py run \
  --config configs/taskmanager/aws_original_models.example.yaml \
  --dry-run
```

## 环境留档

正式训练前保存实际解析结果，不要只保存输入 YAML：

```bash
mkdir -p /mnt/ecg/runs/environment
conda list --explicit > /mnt/ecg/runs/environment/conda-explicit.txt
conda env export --no-builds > /mnt/ecg/runs/environment/conda-environment.yml
python -m pip freeze > /mnt/ecg/runs/environment/pip-freeze.txt
python -m pip check > /mnt/ecg/runs/environment/pip-check.txt
nvidia-smi > /mnt/ecg/runs/environment/nvidia-smi.txt
git rev-parse HEAD > /mnt/ecg/runs/environment/git-revision.txt
```

将该目录与训练结果一起同步到 S3。

## Legacy 环境

`environments/legacy/` 固定 Python 3.8、PyTorch 1.4、CUDA 10 和 FastAI v1，仅用于历史复现。A10G 是 Ampere GPU，不能依赖这些旧构建提供兼容 kernel。

确实需要 legacy 时应使用独立环境并隐藏 GPU：

```bash
CUDA_VISIBLE_DEVICES="" conda run \
  -n ecg-fastai-py38-no-torchvision \
  python -c "import torch, fastai; print(torch.__version__, fastai.__version__)"
```

不要为解决 legacy 导入问题而降级或修改 `ecg-training`。

## 升级规则

- 不单独升级 NumPy 到 2.x；
- 不把 TensorFlow升级到 2.16+/Keras 3，除非重新验证 `.keras` checkpoint；
- 不改变 PyTorch/CUDA minor 后直接恢复旧 optimizer/scheduler checkpoint；
- 任何版本变更都必须运行环境检查器、完整测试和至少一个真实数据 smoke；
- 更新 `ecg-training.yml` 时同步更新三个 requirements 文件和本页版本表。
