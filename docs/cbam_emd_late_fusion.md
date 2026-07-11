# CBAM-xResNet1D EMD Late Fusion

## 配置

配置位于 `code/configs/cbam_configs.py`，使用现有 Python dict 配置风格，不引入 YAML。默认 late-fusion 配置为 `conf_cbam_xresnet1d101_late_fusion`：

```python
input_mode='late_fusion'
use_cbam=True
fusion_type='concat'
cbam_reduction=16
cbam_kernel_size=7
feature_hidden_dim=256
feature_embedding_dim=128
feature_dropout=0.3
fusion_hidden_dim=256
fusion_dropout=0.4
missing_record_policy='drop'
feature_log_transform=False
```

`emd_feature_paths` 是场景到 CSV 的映射。`emd_scenario` 与 `waveform_scenario` 必须相同，避免将不同噪声场景的波形与 EMD 特征混合。默认配置使用 clean waveform 和 `original` EMD 特征。

支持的 `input_mode`：

| 模式 | 输入 | 分类路径 |
|---|---|---|
| `ecg_only` | `[B, 12, T]` | CBAM-xResNet1D ECG embedding 到分类器 |
| `feature_only` | `[B, 12, F]` | EMD MLP encoder 到分类器 |
| `late_fusion` | 两者 | ECG embedding 与 EMD embedding 拼接或 gated 后经 fusion MLP |

模型输出为 raw logits；训练损失保持 `binary_cross_entropy_with_logits`，sigmoid 仅在预测阶段应用。

## 数据对齐

EMD loader 位于 `code/utils/emd_features.py`。它：

1. 用 `RecordNumber == metadata.ecg_id` 对齐，而不使用 CSV 行号或 `RecordFile`。
2. 显式按 `RecordNumber, LeadIndex` 排序。
3. 要求每条记录完整包含 12 个固定顺序导联。
4. 对 `drop` 策略同步删除不完整的 ECG、EMD、标签和 fold；`error` 策略会停止并报告记录 ID。
5. 使用训练折 1-8 的 `[lead, feature]` 均值和标准差标准化 EMD；验证折 9 和测试折 10 不参与拟合。

跨配置中全部 feature CSV 的交集决定最终列，且保持 `CANDIDATE_EMD_FEATURES` 的顺序。当前实际检测到 11 列：

```text
RetainedEnergy, ERV, ERS, IF_Median, IF_Variance, IF_Slope,
IB2_Variance, IB2_Slope, IE12_Mean, IE12_Median, IE12_Slope
```

因此默认 EMD tensor shape 是 `[N, 12, 11]`，不是文档早期误称的 12 个特征。

## 运行

在 `code/` 下执行：

```bash
python run_cbam_emd_experiment.py
```

这会训练 `exp_emd_late_fusion` 的 `all` 任务。已有 baseline 的完整训练命令保持不变：

```bash
python reproduce_results.py
```

结果目录会保存 `emd_scaler.npz`、`emd_config.json`、标签、预测和 fastai checkpoint。`emd_config.json` 保存所用文件、特征列和被删除记录；`emd_scaler.npz` 保存训练集 scaler 和固定导联顺序。

## 训练诊断

CBAM 配置使用 `input_size=10.0`，因此模型输入是 100 Hz 下完整的 1,000 点、10 秒 ECG。原项目 fastai baseline 的默认 `input_size=2.5` 保持不变，训练时输入 250 点裁剪。

训练后执行：

```bash
python diagnose_cbam_emd.py --experiment exp_emd_late_fusion --model cbam_xresnet1d101_late_fusion
```

诊断会保存 validation BCE、全阴性和 class-prior BCE 基线、标签稀疏度、0.5/0.3 阈值完整指标及逐类 ROC-AUC、PR-AUC、F1 到模型结果目录。训练过程会保存 `best_valid_loss.pth`、最终重载后的模型权重和 `training_history.csv`。

## 已完成训练的恢复

若 Colab 在训练结束后的 checkpoint 加载阶段因 PyTorch 2.6 `weights_only=True` 报错，训练得到的 `.pth` 仍可恢复，不需要重新训练。同步修复后执行：

```bash
python recover_cbam_emd_predictions.py
python diagnose_cbam_emd.py --experiment exp_emd_late_fusion --model cbam_xresnet1d101_late_fusion
```

恢复脚本固定使用原训练时的 250 点、2.5 秒验证滑窗，仅生成 train/validation/test 预测与评估结果，不调用 `fit()`。后续新训练仍使用配置中的完整 1,000 点、10 秒输入。
