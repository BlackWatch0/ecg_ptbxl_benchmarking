# Legacy Shell Archive

此目录仅归档历史 Shell，保留文件内容用于审计和复现实验背景，不属于当前受支持的运行入口。

## 约束

- taskmanager 不发现、不导入、也不调用本目录中的任何脚本。
- 不从 CI、当前工作流或文档示例执行这些脚本。
- 脚本迁移后仍保留原始相对路径假设，因此多数脚本从当前位置直接运行会解析到错误的仓库根目录。
- 需要恢复历史实验时，应先阅读脚本并在独立分支显式适配；不得把归档脚本当作当前数据或环境契约。
- 新工作流应使用 Python 入口和结构化配置，不向此目录增加新的生产入口。

## 归档清单

根目录旧脚本：

- `colab_run.sh`
- `get_datasets.sh`
- `run_ablation_colab.sh`
- `run_full_ablation_colab.sh`
- `run_full_original_baseline_colab.sh`
- `run_original_models_benchmark_colab.sh`
- `run_se_ablation_colab.sh`
- `run_wavelet_ablation_colab.sh`

原 `evaluation/` 层级：

- `evaluation/run_colab_saved_checkpoints.sh`

对应的历史 Colab 与消融指南保存在 [`docs/archive/`](../../docs/archive/)。

## 配置替代

| 历史脚本组 | 当前结构化配置/入口 |
|---|---|
| 原论文七模型训练与三域评估 | `configs/taskmanager/original_models_benchmark.yaml` |
| 已有 checkpoint 的七模型评估 | `configs/taskmanager/original_models_evaluate.yaml` |
| clean/noisy/denoised 数据准备 | `configs/taskmanager/prepare_original_data.yaml` |
| AWS EBS 路径示例 | `configs/taskmanager/aws_original_models.example.yaml` |
| CBAM/SE/EMD 消融 | `configs/ablation_cbam_emd.yaml`、`configs/ablation_se.yaml` 与直接 Python runner；EMD 资产当前 blocked |
| Wavelet late-fusion 消融 | `configs/ablation_cbam_wavelet.yaml` 与直接 Python runner |
| 独立 checkpoint 标准评估 | `configs/evaluation/default.yaml` 与 `evaluation/evaluate.py` |

表中的非 taskmanager 工作流仍是直接 Python 入口，不代表已纳入统一调度。
