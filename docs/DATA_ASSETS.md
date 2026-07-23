# 数据资产

## 配置来源

外部数据登记的唯一仓库内来源是 [`configs/datasets.json`](../configs/datasets.json)。配置记录下载元数据，不代表文件已经下载、内容已经校验，或已满足特定工作流的 shape、列名和 ID 对齐要求。

## 已登记数据集

| 配置键 | 角色 | 预期用途 | 本地建议目录 |
|---|---|---|---|
| `ptbxl_original` | `clean_train_validation_test` | clean 训练、验证和测试 | `data/ptbxl/` 或 `data/ptbxl_clean_no_noise/` |
| `ptbxl_noisy` | `noisy_test` | 24、12、6、0、-6 dB mixed-noise 测试 | 数据准备器生成的 noisy 目录 |
| `ptbxl_denoised` | `denoised_test` | 与 noisy ID/SNR 对齐的去噪测试 | 数据准备器生成的 denoised 目录 |

clean PTB-XL 至少需要 metadata、`scp_statements.csv` 和 `records100/`。仓库代码同时兼容官方 `ptbxl_database.csv` 与部分流程使用的 `ptbxl_database_clean_no_noise.csv`，实际文件名必须与入口配置一致。

## 特征归档

| 名称 | 登记状态 | 当前约束 |
|---|---|---|
| `wavelet_feature_extraction` | 已登记 | 原始七模型基准当前从 ECG 直接提取 db6 特征；在完成 ID、场景和 feature-column 对齐验证前，不得用预计算归档替代 |
| `time_domain_feature_extraction` | 已登记 | 是 Lead II beat-level 长表，主键为 `RecordNumber, LeadIndex, BeatIndex`；不能只按 `RecordNumber` 当作单行记录特征 |
| `emd_features` | `source_required` | **blocked**；没有活动 URL/Drive ID，依赖 11 个公共 EMD 特征每导联的工作流不可运行 |

Wavelet、time-domain 和 EMD 是不同 schema，不能相互替代。资产名称相似、记录数相同或同属 PTB-XL 均不足以证明可互换。

## ID 与划分

- 训练使用 folds 1-8，验证使用 fold 9，测试使用 fold 10。
- clean、noisy 和 denoised 测试必须按 `ecg_id` 对齐，不能依赖压缩包顺序、文件系统顺序或 DataFrame 当前行号。
- 同一测试记录在不同 SNR 和去噪场景中必须共享标签。
- 标准化器只能在训练 split 上拟合，并复用于验证和全部测试场景。
- 阈值只能由验证集确定，不得在测试集重新优化。

## 本地目录

建议在本地或 EBS 使用明确分层：

```text
/mnt/ecg/
├── workspace/ecg_ptbxl_benchmarking/
├── downloads/                 # 原始归档，可删除重建
├── data/                      # 解压并校验后的运行输入
├── cache/                     # 可重建 waveform/feature cache
└── runs/                      # checkpoint、日志、预测、指标和报告
```

通过配置或命令参数把入口指向这些绝对路径，不要在仓库中创建指向 S3 的透明挂载。Python 工作流只消费本地普通文件。

## 入库检查

每次新增或替换资产至少记录并校验：

- 来源 URL、归档文件名、格式、字节数和 SHA-256；
- 解压目标与路径遍历防护；
- metadata 行数、唯一 `ecg_id`、fold 分布和标签列；
- WFDB `.hea`/`.dat` 配对与 metadata 覆盖率；
- noisy/denoised 的场景、SNR、重复 ID、缺失 ID和额外 ID；
- 特征列、dtype、非有限值、预期 shape 和主键唯一性；
- 训练、验证、测试 ID 无交集。

下载链接状态及待补资产见 [`DOWNLOAD_LINKS_REVIEW.md`](DOWNLOAD_LINKS_REVIEW.md)。旧 EMD 统计和 Colab 路径仅保存在 [`archive/`](archive/) 供追溯，不能作为当前资产证明。
