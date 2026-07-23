# CBAM-xResNet1D + EMD Late Fusion（5 类诊断 Superclass）

## 任务与输出

```text
task = superdiagnostic
num_classes = 5
label_order = [NORM, MI, STTC, CD, HYP]
loss = binary_cross_entropy_with_logits（多标签，不是 CrossEntropy）
```

训练输入：

```text
ECG:  [B, 12, 1000]  (10 秒, 100 Hz)
EMD:  [B, 12, 11]    (11 个公共特征)
Label: [B, 5]         (multi-hot)
```

## 一键运行

```bash
%cd /content/ecg_ptbxl_benchmarking
!git pull origin feat/noisy-snr-evaluation
!bash colab_run.sh --train
```

完整流程：数据下载 → 校验 → `run_cbam_emd_experiment.py`（训练 + 预测 + 评估）。

## 分步运行

```bash
# 只下载数据
!bash colab_run.sh --prepare

# 校验数据
!bash colab_run.sh --validate

# 训练（需要先准备好数据）
%cd /content/ecg_ptbxl_benchmarking/code
!python run_cbam_emd_experiment.py
```

## 训练后诊断

```bash
%cd /content/ecg_ptbxl_benchmarking/code
!python diagnose_cbam_emd.py
```

输出保存至：

```text
output/exp_emd_late_fusion_superdiagnostic/models/cbam_xresnet1d101_late_fusion_superdiagnostic/
  validation_diagnosis.json
  validation_per_class_threshold_0.5.csv
  validation_per_class_threshold_0.3.csv
```

## 各 SNR 测试

训练完成后：

```bash
%cd /content/ecg_ptbxl_benchmarking/code
!python evaluate_cbam_emd_snr.py
```

对 24、12、6、0、-6 dB 分别使用匹配的 noisy waveform 和同 SNR EMD CSV，输出：

```text
output/exp_emd_late_fusion_superdiagnostic/models/cbam_xresnet1d101_late_fusion_superdiagnostic/
  snr_24/   y_test_prob.npy  y_test_logits.npy
  snr_12/   ...
  snr_0/    ...
  snr_-6/   ...
  snr_test_results.csv
```

## 训练中断恢复

若训练完成后报错（如 PyTorch 2.6 `weights_only` 或 `purge` 参数），`best_valid_loss.pth` 已保存，不需要重新训练：

```bash
%cd /content/ecg_ptbxl_benchmarking/code
!python recover_cbam_emd_predictions.py
!python diagnose_cbam_emd.py
!python evaluate_cbam_emd_snr.py
```

## 消融实验

三种输入模式各有对应配置：

| 配置 | 模式 |
|---|---|
| `conf_cbam_xresnet1d101_late_fusion_superdiagnostic` | ECG + EMD late fusion |
| `conf_cbam_xresnet1d101_ecg_only_superdiagnostic` | ECG only |
| `conf_cbam_xresnet1d101_feature_only_superdiagnostic` | EMD feature only |

修改 `run_cbam_emd_experiment.py` 的导入和 `SCP_Experiment` 配置即可切换。

## 配置参数

```python
input_mode='late_fusion'  # ecg_only | feature_only | late_fusion
use_cbam=True
fusion_type='concat'      # concat | gated
cbam_reduction=16
cbam_kernel_size=7
feature_hidden_dim=256
feature_embedding_dim=128
feature_dropout=0.3
fusion_hidden_dim=256
fusion_dropout=0.4
input_size=10.0           # 秒
epochs=50
lr=1e-2
bs=128
```

## 结果文件

```text
output/exp_emd_late_fusion_superdiagnostic/
  data/
    mlb.pkl
    standard_scaler.pkl
    emd_scaler.npz
    emd_config.json
    y_train.npy / y_val.npy / y_test.npy
  models/cbam_xresnet1d101_late_fusion_superdiagnostic/
    models/
      cbam_xresnet1d101_late_fusion_superdiagnostic.pth
      best_valid_loss.pth
    results/te_results.csv
    y_train_pred.npy / y_val_pred.npy / y_test_pred.npy
    y_val_logits.npy
    training_history.csv
    validation_diagnosis.json
    snr_*/  y_test_prob.npy  y_test_logits.npy
    snr_test_results.csv
```

## 实际标签

```text
总记录: 16,476
train:   13,213
val:     1,618
test:    1,645
多标签记录: 3,955

类别阳性数 (train/val/test):
  NORM: 5999 / 733 / 738
  MI:   3257 / 395 / 408
  STTC: 3195 / 392 / 410
  CD:   3021 / 373 / 375
  HYP:  1701 / 210 / 207
```
