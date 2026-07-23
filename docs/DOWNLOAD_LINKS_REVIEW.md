# 下载链接复核

本页记录 [`configs/datasets.json`](../configs/datasets.json) 中的外部资产。登记状态只说明仓库保存了元数据，不代表本次提交重新下载、校验了内容，也不代表归档能直接输入任意工作流。

## 已登记资产

| 配置项 | Drive ID | 仓库内状态 | 使用限制 |
|---|---|---|---|
| `ptbxl_original` | `1SvI2suvuKf4KJ7bikHuGp0PVNAjRJ6Ge` | 已登记 | 下载后必须校验官方 metadata、WFDB 文件和 fold 覆盖 |
| `ptbxl_noisy` | `1aCC9jzUUqXjgrXoRTfRlroOMMSa505u` | 已登记 | 必须按 `ecg_id` 和 SNR 校验测试记录 |
| `ptbxl_denoised` | `1gjnomlJreB8ttsuRoOiD8DM8IXaa7ciD` | 已登记 | 必须与 noisy 的 ID、SNR 和标签对齐 |
| `wavelet_feature_extraction` | `1mGZRk_SJ20miD8DNvK_BjGtQhoJsA60O` | feature archive 已登记 | 当前原作者 Wavelet+NN 基准从 ECG 直接提取 db6 特征；完成 ID/列对齐验证前不能替代直接提取 |
| `time_domain_feature_extraction` | `1wD8Mb216Xd0pjCJhCUrr2nsDhaBaEi-Q` | feature archive 已登记 | Zstandard 压缩长表；按 `RecordNumber, LeadIndex, BeatIndex` 对齐，不能按记录行号直接拼接 |

9 个历史 Shell 已移入 [`scripts/legacy/`](../scripts/legacy/README.md)，不再有“活动 Shell 自动消费这些链接”的承诺。当前 Python 工作流要求操作者先把资产准备到本地文件系统。

## 待补资产

| 资产 | 配置状态 | 影响 | 所需动作 |
|---|---|---|---|
| EMD late-fusion features | `source_required` | EMD 训练、消融和 SNR 评估 blocked | 提供稳定归档，登记 URL/ID、文件名、格式、大小、SHA-256 和 schema，并完成 ID/导联/特征列对齐验证 |

在 EMD 资产完成登记与验证前：

- 不得在 README 或运行指南中宣称 EMD 可运行；
- 不得把旧文档中的本地 `data/emd_features/` 统计当作可获取资产；
- 不得以 Wavelet、time-domain 或 clean-only EMD 代替匹配场景特征；
- 不得仅依赖环境变量 `EMD_ARCHIVE_PATH` 绕过资产登记和完整性检查。

## 复核要求

外部链接变化时，先更新 `configs/datasets.json`，再更新本页。任何“可用”结论应附带实际下载日期、字节数、SHA-256、解压检查和最小数据 smoke 结果；这些证据未进入仓库前，统一描述为“已登记，运行前需校验”。
