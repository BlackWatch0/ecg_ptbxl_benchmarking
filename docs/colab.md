# Colab 运行指南

## 全新运行（推荐）

在一个新的 Colab notebook 单元格中执行：

```bash
!git clone https://github.com/BlackWatch0/ecg_ptbxl_benchmarking.git
%cd ecg_ptbxl_benchmarking
!git checkout feat/noisy-snr-evaluation
!bash colab_run.sh --train
```

这会：

1. 下载 clean PTB-XL、noisy PTB-XL、EMD features 三个数据包。
2. 将数据归位到 `data/`。
3. 执行 `run_cbam_emd_experiment.py`，训练 5 类 CBAM-xResNet1D + EMD late fusion 模型。
4. 生成验证集预测和评估结果。

建议使用 GPU runtime（Runtime → Change runtime type → T4 GPU）。

## 仅准备数据

```bash
!bash colab_run.sh --prepare
```

仅校验数据：

```bash
!bash colab_run.sh --validate
```

## 训练后操作

训练完成后，在同目录依次执行：

```bash
%cd /content/ecg_ptbxl_benchmarking/code

# 验证集诊断
!python diagnose_cbam_emd.py

# 各 SNR 测试
!python evaluate_cbam_emd_snr.py
```

## 训练中断恢复

若训练在结束后 crash（checkpoint 已保存），不需要重训：

```bash
%cd /content/ecg_ptbxl_benchmarking/code
!python recover_cbam_emd_predictions.py
!python diagnose_cbam_emd.py
!python evaluate_cbam_emd_snr.py
```

## 更新已有仓库

```bash
%cd /content/ecg_ptbxl_benchmarking
!git fetch origin feat/noisy-snr-evaluation
!git checkout feat/noisy-snr-evaluation
!git reset --hard origin/feat/noisy-snr-evaluation
```
