# Taskmanager

完整 YAML 字段、默认值、参数优先级和每种 task 的可选参数见
[`TASK_MANAGER_CONFIG.md`](TASK_MANAGER_CONFIG.md)。

## 范围

仓库根入口为 `taskmanager.py`，实现位于 `code/task_manager/`。它只编排原作者七模型基准的五类 Python 任务：

| task type | Python 入口 | 行为 |
|---|---|---|
| `prepare_data` | `code/prepare_original_models_benchmark_data.py` | 从本地归档/目录生成标准数据配置 |
| `train` | `code/run_original_models_benchmark.py` | 按模型训练，并传入 `--skip-test-evaluation` |
| `evaluate` | `code/run_original_models_benchmark.py` | 按模型评估，并传入 `--evaluate-only` |
| `report` | `code/build_original_models_benchmark_report.py` | 汇总已完成产物 |
| `package` | `code/package_original_models_benchmark.py` | 按白名单生成可移植报告 ZIP |

taskmanager 不调用 [`scripts/legacy/`](../scripts/legacy/README.md) 中的 Shell，不编排 fastai/Lightning 独立推理、`evaluation/evaluate.py`、CBAM/SE/EMD 消融，也不提供原生 S3。

七模型共享现代环境，但 `wavelet_nn` 子进程会隐藏 CUDA 并使用 CPU-only TensorFlow；依赖安装见 [`DEPENDENCIES.md`](DEPENDENCIES.md)。

## 命令

当前 CLI 只有以下命令：

```bash
python taskmanager.py list-models
python taskmanager.py validate --config configs/taskmanager/original_models_benchmark.yaml
python taskmanager.py run --config configs/taskmanager/original_models_benchmark.yaml --dry-run
python taskmanager.py run --config configs/taskmanager/original_models_benchmark.yaml
python taskmanager.py run --config configs/taskmanager/original_models_benchmark.yaml --task train evaluate
```

`--task` 会自动包含所选任务的依赖。`--dry-run` 解析配置、生成命令、resolved config、`task_plan.json` 和 `task_plan_logs/`，但不启动训练，也不覆盖真实运行的状态与日志。

当前没有独立的 `smoke`、`status`、`resume` 或 `logs` 子命令：

- smoke 使用底层 Python 入口的 `--smoke-test`，见 [`WORKFLOWS.md`](WORKFLOWS.md)。
- 状态直接读取 `<output_dir>/task_status.json`。
- 日志位于 `<output_dir>/task_logs/<task>/<model-or-type>.log`。
- `global.resume: true` 会加载同一 resolved config 的前次状态，并向底层 runner 传递 `--resume`；模型/任务 override 只能控制对应底层 runner 的 `--resume`。

## 模型

`list-models` 返回固定七模型：`xresnet1d101`、`resnet1d_wang`、`lstm`、`lstm_bidir`、`fcn_wang`、`inception1d`、`wavelet_nn`。别名会在配置加载时规范化；未知模型、重复模型和循环 group 会直接报错。

EMD 不在 taskmanager 模型集合中。EMD 资产当前 `source_required`，相关工作流 blocked。

## 配置示例

可复现配置应放在 `configs/taskmanager/`。以下示例假定数据已经在本地准备完成；所有相对路径都相对于 YAML 所在目录解析：

仓库提供四份可跟踪模板：

| 配置 | 用途 |
|---|---|
| `prepare_original_data.yaml` | 从本地归档生成三域 manifest 配置 |
| `original_models_benchmark.yaml` | 七模型训练、评估和报告 |
| `original_models_evaluate.yaml` | 从已有 checkpoint 仅评估和报告 |
| `aws_original_models.example.yaml` | 使用 `/mnt/ecg` EBS 路径的 AWS 示例 |

```yaml
version: 1
output_dir: /mnt/ecg/runs/original-models-seed42

global:
  data_config: /mnt/ecg/workspace/normalized/original_models_benchmark_data.json
  seed: 42
  device: cuda
  resume: true
  fail_fast: true

models:
  - xresnet1d101
  - resnet1d_wang
  - lstm
  - lstm_bidir
  - fcn_wang
  - inception1d
  - wavelet_nn

model_groups:
  all_original:
    - xresnet1d101
    - resnet1d_wang
    - lstm
    - lstm_bidir
    - fcn_wang
    - inception1d
    - wavelet_nn

tasks:
  - name: train
    type: train
    models: "@all_original"
  - name: evaluate
    type: evaluate
    depends_on: [train]
    models: "@all_original"
  - name: report
    type: report
    depends_on: [evaluate]
  - name: package
    type: package
    depends_on: [report]
```

数据准备可作为单独 task 配置：

```yaml
version: 1
output_dir: /mnt/ecg/runs/data-prepare
tasks:
  - name: prepare
    type: prepare_data
    options:
      archive:
        - /mnt/ecg/downloads/ptbxl_original_database_plus_mixed_WFDB.tar
        - /mnt/ecg/downloads/denoised_WFDB.tar
      search_root:
        - /mnt/ecg/data
      workspace: /mnt/ecg/workspace/prepared
      output_dir: /mnt/ecg/workspace/normalized
```

准备任务的输出不会自动注入另一次运行；训练配置仍应显式填写生成的 `data_config`。

## 校验与执行

推荐顺序：

```bash
python taskmanager.py validate --config configs/taskmanager/original_models_benchmark.yaml
python taskmanager.py run --config configs/taskmanager/original_models_benchmark.yaml --dry-run
python code/run_original_models_benchmark.py \
  --data-root /mnt/ecg/data/ptbxl \
  --data-config /mnt/ecg/workspace/normalized/original_models_benchmark_data.json \
  --output-dir /mnt/ecg/runs/original-models-smoke \
  --smoke-test
python taskmanager.py run --config configs/taskmanager/original_models_benchmark.yaml
```

`validate` 检查严格 YAML schema、类型、路径解析、模型/group、依赖和拓扑顺序，但不会确认文件存在、CUDA 可用、数据完整或磁盘空间充足。正式执行前必须单独完成这些 preflight。

## 状态、日志与恢复

taskmanager 在 `output_dir` 中原子写入：

```text
<output_dir>/
├── config/resolved_config.yaml
├── task_status.json
├── task_plan.json              # 仅在 dry-run 后存在
├── task_plan_logs/             # dry-run 命令记录
└── task_logs/
    ├── train/<model>.log
    ├── evaluate/<model>.log
    ├── report/report.log
    └── package/package.log
```

`package` 只收录 `final_report/`、`metrics/`、`predictions/`、`training_logs/` 和 `config/`。checkpoint 与 `features/` 缓存保留在运行根目录，不进入 ZIP。

状态包含 config digest、命令参数、时间、return code 和每个 run 的状态。`resume: true` 时，只有 config digest 相同且前次 run 已标记 `completed` 才会跳过；配置变化会重新执行。底层 benchmark 同时收到 `--resume` 并负责 checkpoint 级恢复。

真正重新执行的 run 会以写模式更新对应日志；同配置 resume 跳过已完成 run 时保留原日志。需要长期审计时仍应为不同实验使用唯一 `output_dir`。taskmanager 不替代 checkpoint/metrics 完整性检查。

## 本地与 AWS 边界

- 配置只填写本地普通路径；不要传 `s3://` URI。
- AWS 上使用挂载的加密 EBS，例如 `/mnt/ecg/data` 和 `/mnt/ecg/runs`。
- 操作者在任务前后手动执行 `aws s3 sync`，详见 [`AWS_SETUP.md`](AWS_SETUP.md)。
- 不把 access key 写入 YAML、日志或环境文件；EC2 使用 instance profile。
- `scripts/legacy/` 只归档，不作为 fallback。
