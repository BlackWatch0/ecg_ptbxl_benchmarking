# ECG PTB-XL Benchmarking（Fork）

基于 [helme/ecg_ptbxl_benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking) 的修改版本，原论文见 [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://doi.org/10.1109/jbhi.2020.3022989)。

本 Fork 新增了推理管线，支持使用预训练权重直接推理（无需训练），兼容 PyTorch Lightning 检查点，并针对清洗后（去噪）的 PTB-XL 数据集。

原始 README 留存在 [README_original.md](README_original.md)。

## 与上游的变更

| 变更项 | 说明 |
|---|---|
| **推理管线** | `run_inference.py` — 以纯推理模式运行 fastai 模型（`skip_training=True`） |
| **Lightning 支持** | `run_lightning_inference.py` 提供 XResNet 推理；`evaluate_noisy_mixed_lightning.py` 严格加载全部现有架构 |
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

当前实现仅支持 XResNet Lightning checkpoint。Lightning checkpoint 需要使用 PyTorch 2.x；项目默认的 `ecg_env`（PyTorch 1.4）只能用于 fastai `.pth` 推理。

**混合噪声 Lightning checkpoint 评估：**

```bash
cd code
python evaluate_noisy_mixed_lightning.py
```

该脚本支持 `lenet`、`lstm`、`resnet`、`inception`、`xresnet` 的 `all` 和 `superdiagnostic` checkpoint，并严格校验每个 checkpoint 的全部参数键和张量形状。运行前需使用 PyTorch 2.4 或更高版本。

### 5. 已有结果快速评估

```bash
cd code
python test_evaluate_exp0.py
```

## 核心脚本一览

| 脚本 | 用途 |
|---|---|
| `code/run_inference.py` | Fastai xresnet1d101 推理，兼容 PyTorch 2.6 |
| `code/run_lightning_inference.py` | XResNet Lightning checkpoint 推理 |
| `code/test_evaluate_exp0.py` | 不重新推理，直接评估已有预测结果 |
| `code/reproduce_results.py` | 完整复现流程（已适配清洗数据集） |
| `code/evaluate_noisy_mixed_fastai.py` | 混合噪声数据集的 fastai xresnet1d101 SNR 评估（v0.1.0） |
| `code/evaluate_noisy_mixed_lightning.py` | 混合噪声数据集的全部 Lightning checkpoint SNR 评估（v0.1.0） |
| `code/models/lightning_checkpoint_models.py` | Lightning checkpoint 严格加载模型定义（v0.1.0） |
| `code/generate_noisy_superclass_reports.py` | 生成 5 类 superclass 预测明细、汇总与逐类统计（v0.1.0） |
| `code/evaluate_cbam_emd_snr.py` | 使用匹配 waveform/EMD 的 CBAM 各 SNR 测试 |
| `docs/emd_features.md` | EMD 特征文件、公共列、排序和标签对齐说明 |
| `docs/cbam_emd_late_fusion.md` | CBAM-xResNet1D EMD late-fusion 训练说明 |
| `docs/colab.md` | Colab 数据下载、校验与训练入口 |

## 混合噪声 SNR 评估（v0.1.0）

`evaluate_noisy_mixed_fastai.py` 使用 `data/ptbxl_noisy_mixed_shared/` 中的第 10 折记录，复用 `exp0` 的标签编码和训练集标准化器，对 `fastai_xresnet1d101.pth` 在每个 SNR 下评估。

```bash
cd code
python evaluate_noisy_mixed_fastai.py
```

预测和指标写入 `output/noisy_mixed_shared/fastai_xresnet1d101/`；已有的 SNR 预测会自动复用。结果同时包含 71 类 SCP 指标，以及从 SCP 预测聚合得到的 5 类诊断 superclass 指标（macro-AUC、标签准确率、完全匹配准确率、macro-F1、macro recall、CD/HYP/MI/NORM/STTC 各类 recall）。v0.1.0 已完成的测试结果如下：

| SNR | Macro-AUC | 标签准确率 | 完全匹配准确率 |
|---:|---:|---:|---:|
| 24 dB | 0.9287 | 0.9790 | 0.3695 |
| 12 dB | 0.9269 | 0.9782 | 0.3563 |
| 6 dB | 0.9191 | 0.9768 | 0.3323 |
| 0 dB | 0.9010 | 0.9741 | 0.2497 |
| -6 dB | 0.8474 | 0.9685 | 0.1659 |

## Superclass 报告（v0.1.0）

```bash
cd code
python generate_noisy_superclass_reports.py
```

该脚本以固定阈值 0.5 聚合 71 类 SCP 预测为 CD、HYP、MI、NORM、STTC 五类诊断 superclass，并使用 manifest 中的 `snr_realized_db` 分组。输出位于 `output/noisy_mixed_shared/fastai_xresnet1d101/`：

| 文件 | 内容 |
|---|---|
| `sample_predictions.csv` | 每条 ECG 的真实标签、概率、二值预测及实际 SNR |
| `overall_metrics.csv` | 每个实际 SNR 区间的整体指标与 1000 次 bootstrap 95% CI |
| `per_class_metrics.csv` | 每个实际 SNR 区间、每个 superclass 的混淆矩阵和指标 |

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
