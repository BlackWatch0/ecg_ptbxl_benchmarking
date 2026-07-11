# Colab 运行指南

## 一键运行

在新的 Colab notebook 中依次运行以下单元。GPU runtime 可加快 CBAM-xResNet1D 训练。

```bash
!git clone https://github.com/BlackWatch0/ecg_ptbxl_benchmarking.git
%cd ecg_ptbxl_benchmarking
!bash colab_run.sh --train
```

`colab_run.sh` 使用以下 Google Drive 文件：

| 资源 | 目标目录 |
|---|---|
| clean PTB-XL | `data/ptbxl_clean_no_noise/` |
| mixed-noise PTB-XL | `data/ptbxl_noisy_mixed_shared/` |
| EMD features | `data/emd_features/` |

脚本支持 zip、tar、7z、RAR 和直接 CSV，并通过标志文件定位压缩包内部目录，不依赖压缩包顶层目录名称：

- clean：`ptbxl_database_clean_no_noise.csv`
- noisy：`ptbxl_noisy_mixed_shared_manifest.csv`
- EMD：`original/PTBXL_Batch_Original_EMD_reduced_features.csv`

默认完整流程会下载、解压、校验后运行 `code/run_cbam_emd_experiment.py`。该实验使用 clean waveform 与 original EMD features，并把结果写入 `output/exp_emd_late_fusion/`。

## 分步运行

仅下载、解压和归位数据：

```bash
!bash colab_run.sh --prepare
```

仅验证目录与标志文件：

```bash
!bash colab_run.sh --validate
```

数据准备完成后手动训练：

```bash
%cd /content/ecg_ptbxl_benchmarking/code
!python run_cbam_emd_experiment.py
```

## 场景一致性

当前默认配置是 `original`。若训练某个噪声场景，必须在 `code/configs/cbam_configs.py` 中同时把 `emd_scenario` 和 `waveform_scenario` 设置为同一个值，并确保 `datafolder` 指向该场景对应的 waveform/metadata。代码会拒绝两个场景名称不一致的配置。

## 依赖说明

脚本安装 `gdown`、`wfdb` 与 `fastai==1.0.61`。项目原训练流程基于 Fastai v1；若 Colab 的预装 PyTorch 与 Fastai v1 不兼容，应使用项目 `ecg_env.yml` 对应的兼容环境，或先在 Colab 固定可兼容的 PyTorch/Fastai 组合。数据下载、解压、目录校验与 EMD loader 不依赖 Fastai。
