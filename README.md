# ECG PTB-XL Benchmarking（Fork）

基于 [helme/ecg_ptbxl_benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking) 的修改版本，原论文见 [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://doi.org/10.1109/jbhi.2020.3022989)。

本 Fork 新增了推理管线，支持使用预训练权重直接推理（无需训练），兼容 PyTorch Lightning 检查点，并针对清洗后（去噪）的 PTB-XL 数据集。

原始 README 留存在 [README_original.md](README_original.md)。

## 与上游的变更

| 变更项 | 说明 |
|---|---|
| **推理管线** | `run_inference.py` — 以纯推理模式运行 fastai 模型（`skip_training=True`） |
| **Lightning 支持** | `run_lightning_inference.py` 提供 XResNet 推理；`evaluate_noisy_mixed_lightning.py` 严格加载全部现有架构 |
| **CBAM + EMD** | `run_cbam_emd_experiment.py` 训练 5 类诊断 superclass 的 CBAM-xResNet1D + EMD late fusion 模型 |
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
| `code/run_cbam_emd_experiment.py` | CBAM-xResNet1D + EMD late fusion 5 类训练入口 |
| `code/recover_cbam_emd_predictions.py` | 训练中断后恢复预测（不重训） |
| `code/diagnose_cbam_emd.py` | CBAM 验证集诊断（BCE、AUC、F1、每类指标） |
| `code/evaluate_cbam_emd_snr.py` | CBAM 各 SNR 测试（24/12/6/0/-6 dB） |
| `code/colab_data_setup.py` | Colab 数据下载、解压与目录归位 |
| `code/utils/emd_features.py` | EMD 特征加载、排序、对齐与标准化 |
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

## CBAM-xResNet1D + EMD Late Fusion（5 类诊断）

训练 10 秒、1000 点 CBAM-xResNet1D 主干 + EMD 特征 MLP encoder 的 late fusion 模型，输出 5 个诊断 superclass（NORM、MI、STTC、CD、HYP）。

```bash
cd code
python run_cbam_emd_experiment.py          # 训练 + 预测
python diagnose_cbam_emd.py               # 验证集诊断
python evaluate_cbam_emd_snr.py           # 各 SNR 测试
```

训练中断时使用恢复脚本：

```bash
python recover_cbam_emd_predictions.py
```

详细说明见 `docs/cbam_emd_late_fusion.md` 和 `docs/colab.md`。

## 一键消融实验（CBAM + EMD）

`run_ablation_colab.sh` 会依次训练并评估四个固定的五类 `superdiagnostic` 多标签实验。四组实验使用相同的 fold 划分、随机种子、batch size、Adam + OneCycle 学习率策略和 `BCEWithLogitsLoss`，完整训练 50 轮并使用 validation loss 最优 checkpoint 评估。

SE-xResNet1D 消融使用 `run_se_ablation_colab.sh`。标准 1D SE 模块在每个 residual block 的主分支卷积完成后、与 shortcut 相加前执行时间维全局平均池化和通道重标定，与 CBAM 的插入位置一致。脚本只训练 `se_xresnet1d101` 和 `se_xresnet1d101_emd_late_fusion`，不会覆盖已有 baseline/CBAM 结果；它会将六模型报告写入 Google Drive 的 `ECG/se_ablation_results`。

| 实验名 | CBAM | SE | EMD | 输入 |
|---|---:|---:|---:|---|
| `xresnet1d101_baseline` | 否 | 否 | 否 | ECG |
| `cbam_xresnet1d101` | 是 | 否 | 否 | ECG |
| `se_xresnet1d101` | 否 | 是 | 否 | ECG |
| `xresnet1d101_emd_late_fusion` | 否 | 否 | 是 | ECG + EMD concat |
| `cbam_xresnet1d101_emd_late_fusion` | 是 | 否 | 是 | ECG + EMD concat |
| `se_xresnet1d101_emd_late_fusion` | 否 | 是 | 是 | ECG + EMD concat |

### SE 消融当前进度（2026-07-12）

已完成：

- 实现标准 1D Squeeze-and-Excitation block，默认 `reduction=16`，并保护小通道的 hidden size 不小于 1；
- 支持模型工厂参数 `use_se` / `use_cbam`，两者同时启用时直接报错；
- 注册 ECG-only 和 ECG + EMD late-fusion 两个 SE 实验；
- 扩展真实数据 smoke test，覆盖输出 shape、有限 logits/loss、backward、SE scale `[B,C,1]`、参数量、split ID 和 EMD 对齐；
- 实现六模型汇总、贡献分析、鲁棒性指标、复杂度表、PNG/PDF 图表和英文结果简报；
- 新增真正的 epoch 级断点恢复，保存 model、optimizer、OneCycleLR scheduler、AMP scaler、最佳指标和累计训练时间；
- 本地模型测试 6 项和报告器测试 2 项通过；Colab Tesla T4 上六模型真实数据 smoke test 全部通过；
- baseline/CBAM 四模型 50-epoch 结果已保存在 `/content/drive/MyDrive/ECG/ablation_results_full_ptbxl_50_epochs`。

当前阻塞：

- 首次 `se_xresnet1d101` 训练运行至第 39/50 轮后 Colab GPU 配额耗尽；当时最佳 validation loss 为 `0.335922`（epoch 39）；
- 该次运行开始于 epoch 级恢复功能加入之前，因此 Drive 中保留了 best checkpoint 和 39 轮 history，但没有 optimizer/scheduler 的 `last_checkpoint.pth`；恢复 GPU 后该模型需要从 epoch 1 重新训练，避免用不连续的 OneCycleLR 产生不公平结果；
- `se_xresnet1d101_emd_late_fusion` 尚未开始正式训练，六模型最终报告和数值结论尚未生成。

Colab GPU 配额恢复后，在已挂载 Google Drive、保留现有 `data/` 的仓库中同步最新 `master`，执行唯一入口：

```bash
!bash run_se_ablation_colab.sh
```

SE checkpoint、history、预测、指标和最终报告写入：

```text
/content/drive/MyDrive/ECG/se_ablation_results/
```

最终压缩包将写入：

```text
/content/drive/MyDrive/ECG/se_ablation_results/se_ablation_summary_figures_metrics.zip
```

相关提交：`5b40064`（SE 消融管线）、`5895a99`（epoch 级断点续跑）。

### 原作者模型三测试域基准

所有数据下载地址集中在 [`configs/datasets.json`](configs/datasets.json)，当前配置为：

| 配置键 | 用途 | 下载地址 |
|---|---|---|
| `ptbxl_original` | clean 训练/验证/测试 | [Google Drive](https://drive.google.com/file/d/1SvI2suvuKf4KJ7bikHuGp0PVNAjRJ6Ge/view) |
| `ptbxl_noisy` | mixed-noise 测试 | [Google Drive](https://drive.google.com/file/d/1aCC9jzUUqXJjgrXoRTfRlroOMMSa505u/view) |
| `ptbxl_denoised` | denoised 测试 | [Google Drive](https://drive.google.com/file/d/1gjnomlJreB8ttsuRoOiD8DM8IXaa7ciD/view) |

后续特征提取归档统一添加到配置中的 `feature_archives` 数组，不再向运行脚本写入下载常量。预留格式如下：

```json
{
  "name": "unique_feature_name",
  "scenario": "clean_or_noisy_or_denoised",
  "role": "model_feature_input",
  "url": "https://drive.google.com/file/d/FILE_ID/view",
  "drive_id": "FILE_ID",
  "archive_name": "features.tar",
  "format": "tar"
}
```

`run_original_models_benchmark_colab.sh` 在 clean PTB-XL folds 1-8 上训练、fold 9 上选择最佳 validation-loss checkpoint，并在 fold 10 的三个测试域上评估：

- clean PTB-XL；
- [mixed-noise PTB-XL](https://drive.google.com/file/d/1aCC9jzUUqXJjgrXoRTfRlroOMMSa505u/view)，包含 24、12、6、0、-6 dB；
- [denoised PTB-XL](https://drive.google.com/file/d/1gjnomlJreB8ttsuRoOiD8DM8IXaa7ciD/view)，包含与 noisy 数据相同的记录和 SNR。

基准覆盖原论文复现入口中的七个模型：xResNet1D-101、ResNet1D-Wang、LSTM、BiLSTM、FCN-Wang、Inception1D 和 Wavelet+NN。六个波形网络沿用 250 点随机训练 crop、验证/测试 stride 125 crop 和 max probability 聚合；Wavelet+NN 沿用 db6 level-5 特征、train-only StandardScaler、128 单元全连接层、0.25 dropout 和 Adamax。

在 Colab 挂载 Google Drive、同步最新 `master` 后执行：

```bash
!bash run_original_models_benchmark_colab.sh
```

脚本会自动下载并安全解压两个测试归档，按 `ecg_id` 验证 fold-10 完整覆盖，运行 smoke test、断点训练、三阈值评估、逐类指标、图表、Markdown 报告和 ZIP 打包。结果写入：

```text
/content/drive/MyDrive/ECG/original_models_benchmark/
```

测试结果固定由以下部分组成，不生成简化版：

```text
original_models_benchmark/
├── config/                 # 配置、数据完整性、标准化器、smoke test
├── checkpoints/            # best/last checkpoint、阈值和模型信息
├── training_logs/          # 每模型逐 epoch loss 和 learning rate
├── features/wavelet_nn/    # 可恢复的 Wavelet 特征缓存
├── predictions/            # 三种阈值策略的逐记录概率与预测
├── metrics/                # 整体、逐类、复杂度和对齐指标
├── errors/
├── completed_models.json
├── final_report/
│   ├── benchmark_summary.csv
│   ├── clean_comparison.csv
│   ├── noisy_snr_comparison.csv
│   ├── denoised_snr_comparison.csv
│   ├── denoising_contributions.csv
│   ├── robustness_metrics.csv
│   ├── mean_domain_metrics.csv
│   ├── per_class_metrics.csv
│   ├── model_complexity.csv
│   ├── best_model_summary.json
│   ├── ORIGINAL_MODELS_BENCHMARK_RESULTS.md
│   └── figures/            # 每张图同时保存 PNG 和 PDF
└── original_models_benchmark_report.zip
```

三种阈值策略为 `threshold_0.5`、`best_global_threshold` 和 `per_class_thresholds`，均只使用 clean validation fold 9 选择；最终主表使用 `per_class_thresholds`。

整体指标包含 Macro/Micro ROC-AUC、Macro/Micro PR-AUC、Macro/Micro/Samples F1、label accuracy、exact-match accuracy、预测阳性率、平均预测标签数和全零预测率。逐类指标对 `NORM, MI, STTC, CD, HYP` 保存 ROC-AUC、PR-AUC、precision、recall、specificity、F1、正负样本支持数和预测阳性数。

复杂度表保存参数量、可训练参数量、训练时间、最佳 epoch、最佳 validation loss、单样本推理时间和实际 batch size。图表覆盖 noisy/denoised 的 AUC/F1 vs SNR、相对 clean 的下降、clean/-6 dB 逐类结果、参数与推理时间权衡、训练曲线和去噪前后对比。

压缩包为：

```text
/content/drive/MyDrive/ECG/original_models_benchmark/original_models_benchmark_report.zip
```

在 Colab 新建一个代码单元，完整执行以下代码。它会保留已有的 `data/` 目录；即使之前清理工作区时删除了 `.git` 或脚本，也会自动重建仓库并放回数据集。

```python
from google.colab import drive
from pathlib import Path
import shutil
import subprocess

drive.mount('/content/drive')

repo = Path('/content/ecg_ptbxl_benchmarking')
saved_data = Path('/content/ecg_ptbxl_preserved_data')

if repo.exists() and not (repo / '.git').exists():
    if saved_data.exists():
        raise RuntimeError('Temporary data path already exists: {}'.format(saved_data))
    if (repo / 'data').exists():
        shutil.move(str(repo / 'data'), str(saved_data))
    shutil.rmtree(repo)

if not repo.exists():
    subprocess.run([
        'git', 'clone', 'https://github.com/BlackWatch0/ecg_ptbxl_benchmarking.git', str(repo)
    ], check=True)

if saved_data.exists():
    if (repo / 'data').exists():
        raise RuntimeError('Refusing to overwrite cloned data path')
    shutil.move(str(saved_data), str(repo / 'data'))

subprocess.run(['git', '-C', str(repo), 'fetch', 'origin'], check=True)
subprocess.run(['git', '-C', str(repo), 'checkout', 'master'], check=True)
subprocess.run(['git', '-C', str(repo), 'reset', '--hard', 'origin/master'], check=True)
subprocess.run(['bash', 'run_ablation_colab.sh'], cwd=str(repo), check=True)
```

脚本会检查数据；缺失时调用既有下载准备流程。它以 `--resume` 运行，因此已存在的 best checkpoint、训练 history、全部 SNR 指标和预测文件会被复用。结果默认保存到：

```text
/content/drive/MyDrive/ECG/ablation_results/
```

运行器在 validation set 上搜索 global threshold 和五个 per-class threshold，并对 clean、24、12、6、0、-6 dB 测试集保存三种阈值策略的指标。最终 `final_report/` 包含模型比较、SNR 比较、消融贡献、鲁棒性、最佳模型和 PNG/PDF 图表。

每个 SNR 的 EMD 通过 `RecordNumber` 对齐到 ECG `ecg_id`。若存在对应 noisy EMD 文件，结果标记为 `emd_source=matched_snrXX`；若文件缺失，才使用 clean EMD upper bound，并明确标记 `emd_source=clean_original`、`feature_scenario=clean`，不会混作 matched EMD。完整说明见 [COLAB_ABLATION_GUIDE.md](COLAB_ABLATION_GUIDE.md)。

### 合并剩余 PTB-XL 记录后从零训练

`run_full_ablation_colab.sh` 下载并合并 `ptbxl_original_noisy_remaining.tar` 与 `ptbxl_original_noisy_remaining_plus_mixed_noise.tar`。它会合并 metadata 和 noisy manifest、复制缺失 WFDB 文件、删除旧 raw cache，并验证 full clean metadata、五个 SNR 和 original EMD 的 ID 覆盖关系。结果写入独立目录，不会混入旧的 16,789 条记录实验：

```text
/content/drive/MyDrive/ECG/ablation_results_full_ptbxl/
```

在 Colab 挂载 Drive 并同步仓库后执行：

```bash
!bash run_full_ablation_colab.sh
```

合并明细会保存到 `data/full_data_merge_report.json`。清理 clean `raw100.npy` 是预期行为，下一次训练会从合并后的 `records100/` 自动重建缓存。

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
