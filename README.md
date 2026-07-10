# ECG PTB-XL Benchmarking（Fork）

基于 [helme/ecg_ptbxl_benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking) 的修改版本，原论文见 [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://doi.org/10.1109/jbhi.2020.3022989)。

本 Fork 新增了推理管线，支持使用预训练权重直接推理（无需训练），兼容 PyTorch Lightning 检查点，并针对清洗后（去噪）的 PTB-XL 数据集。

原始 README 留存在 [README_original.md](README_original.md)。

## 与上游的变更

| 变更项 | 说明 |
|---|---|
| **推理管线** | `run_inference.py` — 以纯推理模式运行 fastai 模型（`skip_training=True`） |
| **Lightning 支持** | `run_lightning_inference.py` — 直接加载 PyTorch Lightning 检查点，无需 fastai |
| **自定义数据集** | `load_dataset()` 新增 `database_filename` 和 `dataset_type` 参数，支持使用清洗后的 CSV |
| **预训练权重加载** | `_predict_with_pretrained()` 将已有的 `.pth` 权重加载到 fastai 模型中 |
| **PyTorch ≥2.6 兼容** | Monkey-patch `Learner.load`，使用 `weights_only=False` |
| **环境配置** | 移除了 `torchvision` 依赖；原始 `ecg_env.yml` 保存为 `ecg_env_original.yml` |
| **Git 管理** | `output/`、检查点文件和 Python 缓存目录已加入 `.gitignore` |

## 环境配置

### 1. 安装依赖

```bash
conda env create -f ecg_env.yml
conda activate ecg_env
```

### 2. 获取数据

```bash
./get_datasets.sh
```

预期数据目录结构：
```
data/
  ptbxl_clean_no_noise/
    ptbxl_database_clean_no_noise.csv
    records100/  (WFDB 信号文件)
```

### 3. 预训练模型

将预训练权重放置到 `output/exp0/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pth`。

Lightning 模型检查点放置到 `output/<模型名>/checkpoints/best_model.ckpt`。

### 4. 运行推理

**Fastai 模型推理：**

```bash
cd code
python run_inference.py
```

**Lightning 模型推理：**

```bash
cd code
python run_lightning_inference.py
```

支持测试全部架构（xresnet、inception、resnet、lstm），涵盖 `all`（71 类）和 `superdiagnostic`（5 类）两种任务。

### 5. 已有结果快速评估

```bash
cd code
python test_evaluate_exp0.py
```

## 核心脚本一览

| 脚本 | 用途 |
|---|---|
| `code/run_inference.py` | Fastai xresnet1d101 推理，兼容 PyTorch 2.6 |
| `code/run_lightning_inference.py` | Lightning 检查点推理（4 种架构 × 2 种任务） |
| `code/test_evaluate_exp0.py` | 不重新推理，直接评估已有预测结果 |
| `code/reproduce_results.py` | 完整复现流程（已适配清洗数据集） |

## 参考文献

```bibtex
@article{Strodthoff:2020Deep,
  doi = {10.1109/jbhi.2020.3022989},
  year = {2021},
  volume={25}, number={5}, pages={1519-1528},
  author = {Nils Strodthoff and Patrick Wagner and Tobias Schaeffter and Wojciech Samek},
  title = {Deep Learning for {ECG} Analysis: Benchmarks and Insights from {PTB}-{XL}},
  journal = {{IEEE} Journal of Biomedical and Health Informatics}
}

@article{Wagner:2020PTBXL,
  doi = {10.1038/s41597-020-0495-6},
  year = {2020},
  volume = {7}, number = {1}, pages = {154},
  author = {Patrick Wagner and Nils Strodthoff and Ralf-Dieter Bousseljot and Dieter Kreiseler
            and Fatima I. Lunze and Wojciech Samek and Tobias Schaeffter},
  title = {{PTB}-{XL}, a large publicly available electrocardiography dataset},
  journal = {Scientific Data}
}
```
