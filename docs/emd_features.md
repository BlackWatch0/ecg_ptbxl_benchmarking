# EMD 特征数据说明

## 版本与范围

本说明基于当前工作区的 `data/emd_features/`。EMD 文件本身不包含诊断标签，标签必须通过 `RecordNumber == ecg_id` 关联 `data/ptbxl_clean_no_noise/ptbxl_database_clean_no_noise.csv`。

清洗后的 PTB-XL 元数据包含 16,789 条记录，`ecg_id` 范围为 2 至 21,836。噪声 EMD 文件均完整覆盖这 16,789 条记录。`original` 文件包含更大的原始集合，不应直接与清洗数据的标签按行位置关联。

## 文件清单

| 场景 | 文件 | 记录数 | 行数 | 特征列数 | 备注 |
|---|---|---:|---:|---:|---|
| 原始 | `original/PTBXL_Batch_Original_EMD_reduced_features.csv` | 21,799 | 261,588 | 13 | 包含非清洗集合记录；record 12722 的 V5 EMD 失败 |
| 24 dB | `mixed_snr24/mixed_snr24_MAT_Batch_EMD_reduced_features.csv` | 16,789 | 201,468 | 13 | 与清洗标签完整对齐 |
| 12 dB | `mixed_snr12/mixed_snr12_MAT_Batch_EMD_reduced_features.csv` | 16,789 | 201,468 | 14 | 额外包含 `IF_RMS` |
| 6 dB | `mixed_snr6/mixed_snr6_DenoisedCSV_EMD_reduced_features.csv` | 16,789 | 201,468 | 14 | 含 `DenoisingCondition` 与 `IE12_RMS` |
| 0 dB | `mixed_snr0/mixed_snr0_DenoisedCSV_EMD_reduced_features.csv` | 16,789 | 201,468 | 13 | 含 `DenoisingCondition` 与 `IE12_RMS` |
| -6 dB | `mixed_snrm6/mixed_snrm6_MAT_Batch_EMD_reduced_features.csv` | 16,789 | 201,468 | 12 | 缺少 `IB2_Median` |
| -6 dB 全量 | `mixed_snrm6/mixed_snrm6_MAT_Batch_EMD_all_features.csv` | 16,789 | 201,468 | 18 | 包含均值和 RMS 等扩展特征 |

特征列数不含记录标识、导联标识、采样信息、处理状态和 IMF 计数等元数据列。

## 行排序与张量格式

每一行对应一个 `(RecordNumber, LeadIndex)`，而不是一条完整 ECG。所有噪声文件均满足：

- 每条记录恰好 12 行。
- `LeadIndex` 是 1 到 12。
- 导联顺序固定为 `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`。
- `(RecordNumber, LeadIndex)` 没有重复。
- 采样率为 100 Hz，长度为 1,000 点，时长为 10 秒。

后续处理必须显式排序，不能依赖 CSV 当前行顺序：

```python
import numpy as np
import pandas as pd

feature_columns = [
    'RetainedEnergy', 'ERV', 'ERS',
    'IF_Median', 'IF_Variance', 'IF_Slope',
    'IB2_Variance', 'IB2_Slope',
    'IE12_Mean', 'IE12_Median', 'IE12_Slope',
]

rows = pd.read_csv(feature_path, low_memory=False)
rows = rows[
    (rows['ProcessingStatus'] == 'Success') &
    rows['RecordNumber'].isin(metadata['ecg_id'])
].sort_values(['RecordNumber', 'LeadIndex'])

assert rows.groupby('RecordNumber').size().eq(12).all()
record_ids = rows['RecordNumber'].drop_duplicates().to_numpy()
X = np.stack([
    group[feature_columns].to_numpy(dtype=np.float32)
    for _, group in rows.groupby('RecordNumber', sort=True)
])
assert X.shape == (len(record_ids), 12, len(feature_columns))
```

`X` 的轴顺序固定为 `[记录, 导联, EMD 特征]`。如模型需要展平导联特征，可使用 `X.reshape(len(X), -1)`；如模型需要序列输入，EMD 特征不含时间轴，不能误当作 `[时间, 导联]` 信号。

## 公共特征集

不同文件的 reduced 特征列不一致。跨原始、全部五个 SNR 和 -6 dB 全量文件共同存在的稳定特征仅有以下 11 个：

```text
RetainedEnergy, ERV, ERS,
IF_Median, IF_Variance, IF_Slope,
IB2_Variance, IB2_Slope,
IE12_Mean, IE12_Median, IE12_Slope
```

要进行跨 SNR 对比，应只使用这 11 列。`IF_RMS`、`IE12_RMS`、`IE12_Variance`、`IB2_Median` 等列只在部分场景出现，不能直接拼接为统一训练集。`mixed_snrm6_MAT_Batch_EMD_all_features.csv` 是 -6 dB 的扩展版本，不能与 reduced 文件混用，除非先显式选择共同列。

## 标签对齐

标签来源是清洗后的 PTB-XL CSV：

```python
metadata = pd.read_csv(
    'data/ptbxl_clean_no_noise/ptbxl_database_clean_no_noise.csv'
).set_index('ecg_id')
labels = metadata.loc[record_ids]
```

关键规则：

- 连接键是 `RecordNumber` 与 `ecg_id`，不是 `RecordFile`。噪声文件的 `RecordFile` 有时带有 `_mixed_snr*` 后缀，有时没有。
- 标签字段是 `scp_codes`，存储为 Python 字典字符串；加载后需要 `ast.literal_eval`。
- 训练/验证/测试划分使用 `strat_fold`：1-8 为训练，9 为验证，10 为测试。
- `task='all'` 对应 71 个 SCP 标签；诊断 superclass 对应 `CD`、`HYP`、`MI`、`NORM`、`STTC` 五类。
- 同一个 `RecordNumber` 在不同 SNR 中共享完全相同的标签和 fold，只替换 EMD 特征来源。

推荐在切分前按 `RecordNumber` 排序并对齐 metadata，再根据 `strat_fold` 建立索引。禁止分别按两个 CSV 的行号切分或连接。

## 原始文件的额外处理

`original/PTBXL_Batch_Original_EMD_reduced_features.csv` 不等同于清洗数据：

- 共 21,799 条记录，其中只有 16,789 条出现在清洗后的 metadata 中。
- `RecordNumber=12722`、`Lead=V5` 的 `ProcessingStatus` 为 `Failed`，错误信息表明信号能量无效，不能执行 EMD。
- 如需与噪声数据对照，应先筛选清洗 metadata 中的 `ecg_id`，再仅保留 `ProcessingStatus == 'Success'`，并确认每条记录仍有 12 个导联；否则应剔除整条记录或采用明确的缺失导联策略。
