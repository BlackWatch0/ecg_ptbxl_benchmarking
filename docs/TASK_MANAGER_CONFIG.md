# Taskmanager YAML 配置参考

本文是 `taskmanager.py` 的完整配置字段参考。入门流程、状态文件和运行命令见
[`TASK_MANAGER.md`](TASK_MANAGER.md)，可直接修改的模板位于
[`configs/taskmanager/`](../configs/taskmanager/)。

## 配置与执行顺序

推荐从七模型模板开始：

```bash
cp configs/taskmanager/original_models_benchmark.yaml configs/taskmanager/my_benchmark.yaml
```

填写本地数据路径并选择模型后，依次执行：

```bash
python taskmanager.py validate --config configs/taskmanager/my_benchmark.yaml
python taskmanager.py run --config configs/taskmanager/my_benchmark.yaml --dry-run
python taskmanager.py run --config configs/taskmanager/my_benchmark.yaml
```

所有相对路径均相对于 YAML 文件所在目录解析，不相对于当前终端目录解析。配置仅接受本地文件系统路径；AWS 上使用 EBS 路径，不填写 `s3://` URI。

`validate` 可以在 Windows 检查包含 `/mnt/ecg/...` 的 AWS YAML，但 `run`/`--dry-run` 必须在与路径格式匹配的操作系统执行，避免把 Linux 绝对路径误写到 Windows 当前盘符。

## 完整结构示例

```yaml
version: 1
output_dir: ../../results/original_models_benchmark

global:
  data_root: ../../data/ptbxl
  data_config: ../../data/normalized/original_models_benchmark_data.json
  seed: 42
  epochs: 50
  batch_size: 128
  wavelet_epochs: 30
  wavelet_batch_size: 128
  crop_length: 250
  num_workers: 0
  device: cuda
  official_raw_data: false
  cache_dir: ../../data/cache
  resume: true
  no_mixed_precision: false
  fail_fast: false

models:
  xresnet1d101: {}
  lstm:
    learning_rate: 0.001
  wavelet_nn:
    wavelet_epochs: 30

model_groups:
  selected:
    - xresnet1d101
    - lstm
    - wavelet_nn

tasks:
  - name: train
    type: train
    models: "@selected"
    options:
      epochs: 50

  - name: evaluate
    type: evaluate
    depends_on: [train]
    models: "@selected"

  - name: report
    type: report
    depends_on: [evaluate]

  - name: package
    type: package
    depends_on: [report]
```

`null` 不能用于需要数字、布尔值或路径的字段。要使用底层默认值，应直接省略该字段。

## 根字段

| 字段 | 必需 | 类型 | 默认值 | 说明 |
|---|---:|---|---|---|
| `version` | 否 | integer | `1` | 当前只接受 `1` |
| `output_dir` | 是 | path | 无 | taskmanager 状态、日志及原模型产物的统一运行根目录 |
| `global` | 否 | mapping | `{}` | train/evaluate 的公共参数及 taskmanager 执行策略 |
| `models` | 否 | list/mapping | 全部七模型 | 可用模型及模型级参数覆盖 |
| `model_groups` | 否 | mapping | `{}` | 可复用模型组 |
| `tasks` | 是 | non-empty list | 无 | 需要执行的任务定义 |

根字段和所有子层级都执行严格校验。字段拼写错误、重复 YAML key、未知模型、循环依赖和错误类型都会在 `validate` 阶段失败。

## 参数优先级

train/evaluate 参数按以下顺序覆盖，右侧优先级最高：

```text
底层 runner 默认值 < global < task.options < models.<model>
```

示例：

```yaml
global:
  batch_size: 128

models:
  xresnet1d101: {}
  lstm:
    batch_size: 32

tasks:
  - name: train
    type: train
    options:
      batch_size: 64
```

最终 `xresnet1d101` 使用 batch 64，`lstm` 使用 batch 32。

`seed` 和 `seeds` 在同一层不能同时出现。高优先级层声明其中一个时，会替换低优先级层的另一个。

## Global 全部参数

下表字段可出现在 `global`。除 `fail_fast` 外，也可出现在 train/evaluate 的 `options` 或模型级 overrides 中。

| 字段 | 类型 | 省略时行为 | 底层 CLI | 说明 |
|---|---|---|---|---|
| `data_root` | path | 仓库 `data/` | `--data-root` | clean PTB-XL 根目录，应包含 metadata 和 WFDB records |
| `data_config` | path | 不加载标准 manifest 配置 | `--data-config` | 数据准备器生成的 `original_models_benchmark_data.json` |
| `seed` | non-negative integer | seed 42 | 转为 `--seeds` | 单个随机种子 |
| `seeds` | non-empty integer list | seed 42 | `--seeds` | 多个随机种子；taskmanager 按模型和 seed 启动独立进程 |
| `epochs` | positive integer | `50` | `--epochs` | 六个 PyTorch raw-waveform 模型的 epoch 数 |
| `batch_size` | positive integer | `128` | `--batch-size` | raw-waveform 模型 batch；CUDA OOM 时底层 runner 可降低实际 batch |
| `wavelet_epochs` | positive integer | `30` | `--wavelet-epochs` | Wavelet+NN 的 epoch 数 |
| `wavelet_batch_size` | positive integer | `128` | `--wavelet-batch-size` | Wavelet+NN 训练与推理 batch |
| `learning_rate` | positive number | LSTM/BiLSTM `0.001`，其他 raw 模型 `0.01` | `--learning-rate` | 仅覆盖 PyTorch raw-waveform 模型；Wavelet+NN 使用 Adamax 默认设置 |
| `crop_length` | positive integer | `250` | `--crop-length` | raw-waveform 训练和推理 crop 长度 |
| `num_workers` | non-negative integer | `0` | `--num-workers` | PyTorch DataLoader worker 数 |
| `device` | non-empty string | CUDA 可用时 `cuda`，否则 `cpu` | `--device` | raw-waveform 模型使用该设备；taskmanager 强制 Wavelet+NN 使用 CPU |
| `official_raw_data` | boolean | `false` | `--official-raw-data` | 使用官方 PTB-XL metadata/records 路径而非 fork 的 clean 布局 |
| `cache_dir` | path | 由底层数据入口决定 | `--cache-dir` | official raw data 的可重建缓存目录 |
| `resume` | boolean | `false` | `--resume` | 恢复底层 checkpoint；在 `global` 中还启用 task 状态恢复 |
| `no_mixed_precision` | boolean | `false` | `--no-mixed-precision` | `true` 时禁用 PyTorch AMP |
| `noisy_manifest` | path | 可由 `data_config` 填充 | `--noisy-manifest` | noisy fold-10 manifest |
| `noisy_root` | path | 可由 `data_config` 填充 | `--noisy-root` | noisy WFDB 根目录 |
| `denoised_manifest` | path | 可由 `data_config` 填充 | `--denoised-manifest` | denoised fold-10 manifest |
| `denoised_root` | path | 可由 `data_config` 填充 | `--denoised-root` | denoised WFDB 根目录 |
| `fail_fast` | boolean | `true` | taskmanager only | 任一 run 失败后是否停止后续独立 run；只能出现在 `global` |

通常只需要填写 `data_root` 和 `data_config`。如果使用 `data_config`，不要再填写与其冲突的 noisy/denoised 路径。若不使用 `data_config`，noisy/denoised 的四个字段必须同时提供，否则底层 runner 会拒绝执行。

## Models 配置

### 支持的模型

| canonical name | 常用别名 |
|---|---|
| `xresnet1d101` | `fastai_xresnet1d101` |
| `resnet1d_wang` | `fastai_resnet1d_wang` |
| `lstm` | `fastai_lstm` |
| `lstm_bidir` | `bidir_lstm`、`bidirectional_lstm`、`fastai_lstm_bidir` |
| `fcn_wang` | `fastai_fcn_wang` |
| `inception1d` | `fastai_inception1d` |
| `wavelet_nn` | `wavelet`、`wavelet+nn` |

运行 `python taskmanager.py list-models` 可查看 canonical names。taskmanager 当前只编排这七个原论文模型，不编排 EMD、CBAM/SE 消融和独立 Lightning/FastAI 推理。

### 简单列表

```yaml
models:
  - xresnet1d101
  - lstm
```

### Mapping 与模型级覆盖

```yaml
models:
  xresnet1d101:
    batch_size: 128
  lstm:
    learning_rate: 0.001
    batch_size: 32
```

### 显式对象列表

```yaml
models:
  - name: xresnet1d101
    overrides:
      batch_size: 128
  - name: lstm
    overrides:
      learning_rate: 0.001
```

模型级 overrides 接受 Global 表中除 `fail_fast` 外的全部 benchmark 参数。模型级不能修改统一 `output_dir`，避免 checkpoint、指标和报告分散到不同运行根目录。

## Model Groups

```yaml
model_groups:
  cnn:
    - xresnet1d101
    - resnet1d_wang
    - fcn_wang
    - inception1d
  selected:
    - cnn
    - lstm
```

group 可以引用模型或其他 group。任务中推荐使用 `"@selected"` 明确表示 group：

```yaml
tasks:
  - name: train
    type: train
    models: "@selected"
```

循环 group、未知成员和同一组展开后的重复模型会在加载时处理：循环/未知值报错，重复模型只保留第一次出现的位置。

## Task 公共字段

| 字段 | 必需 | 类型 | 默认值 | 说明 |
|---|---:|---|---|---|
| `name` | 与 `id` 二选一 | string | 无 | task 唯一名称 |
| `id` | 与 `name` 二选一 | string | 无 | `name` 的等价写法；不能同时填写 |
| `type` | 是 | enum | 无 | `prepare_data`、`train`、`evaluate`、`report`、`package` |
| `depends_on` | 否 | string list | `[]` | 依赖 task 名；支持任意声明顺序并执行拓扑排序 |
| `models` | train/evaluate 可选 | string/list | 根 `models` | 模型名、group 名或列表；其他 task 禁止填写 |
| `options` | 否 | mapping | `{}` | task 类型对应的参数覆盖 |

`python taskmanager.py run ... --task evaluate` 会自动加入 `evaluate.depends_on` 指定的依赖。如果要从现有 checkpoint 仅执行评估，请使用不依赖 train 的 `original_models_evaluate.yaml`。

## prepare_data 参数

prepare task 调用 `code/prepare_original_models_benchmark_data.py`。

| `options` 字段 | 类型 | 默认值 | CLI | 说明 |
|---|---|---|---|---|
| `archive` | path list | `[]` | 重复 `--archive` | 本地 clean/noisy/denoised 归档 |
| `search_root` | path list | `[]` | 重复 `--search-root` | 搜索已解压数据的本地目录 |
| `workspace` | path | `<output_dir>/data_workspace` | `--workspace` | 安全解压和发现数据的工作目录 |
| `output_dir` | path | `<output_dir>/config/prepared_data` | `--output-dir` | 生成 normalized manifests 和数据配置的目录 |

`archive` 与 `search_root` 至少有一个非空。taskmanager 不下载 Google Drive 文件，也不执行 S3 同步。

```yaml
tasks:
  - name: prepare
    type: prepare_data
    options:
      archive:
        - ../../data/archives/ptb-xl-1.0.3.zip
        - ../../data/archives/ptbxl_original_database_plus_mixed_WFDB.tar
        - ../../data/archives/denoised_WFDB.tar
      search_root: [../../data]
      workspace: ../../data/prepared
      output_dir: ../../data/normalized
```

prepare 输出不会自动注入另一个独立 YAML。训练配置仍应显式将 `global.data_config` 指向生成的 JSON，并将 `global.data_root` 指向 clean PTB-XL 根目录。

## train 参数

train task 调用 `code/run_original_models_benchmark.py`，自动添加 `--skip-test-evaluation`。因此它只训练模型、评估 validation fold 9 并选择阈值，不评估 fold 10。

train 的 `options` 接受 Global 参数表中除 `fail_fast` 外的全部 benchmark 字段。例如：

```yaml
tasks:
  - name: train
    type: train
    models: [xresnet1d101, lstm]
    options:
      epochs: 20
      batch_size: 64
      resume: true
```

## evaluate 参数

evaluate task 调用同一 benchmark runner，自动添加 `--evaluate-only`。它要求目标运行根目录中已经存在对应模型和 seed 的 checkpoint。

evaluate 的 `options` 接受 Global 参数表中除 `fail_fast` 外的全部 benchmark 字段。类别顺序固定为 `NORM, MI, STTC, CD, HYP`，测试域固定为 clean、五个 noisy SNR 和五个 denoised SNR，阈值固定来自 validation fold 9。

为避免 TensorFlow 与 PyTorch CUDA runtime/显存冲突，`wavelet_nn` 子进程总是使用 `--device cpu` 和 `CUDA_VISIBLE_DEVICES=-1`，同时把 OMP/MKL/OpenBLAS/NumExpr 线程设为 1，避免与 Wavelet 多进程提取叠加；其他模型使用配置的 `device`。

```yaml
tasks:
  - name: evaluate
    type: evaluate
    models: [xresnet1d101, wavelet_nn]
    options:
      batch_size: 64
      device: cuda:0
```

## report 参数

report task 调用 `code/build_original_models_benchmark_report.py`。

| `options` 字段 | 类型 | 默认值 | CLI | 说明 |
|---|---|---|---|---|
| `input_root` | path | 根 `output_dir` | `--input-root` | 包含 metrics、predictions 和 training logs 的 benchmark 根目录 |
| `output_dir` | path | `<input_root>/final_report` | `--output-dir` | 汇总表、图和 Markdown 报告目录 |
| `expected_seeds` | non-empty integer list | `global.seeds`、`global.seed` 或 `[42]` | `--expected-seeds` | 报告完整性检查要求的 seed |
| `excluded_wavelet_status` | non-empty string | 不设置 | `--excluded-wavelet-status` | 仅用于明确记录 Wavelet 排除原因；正式七模型基准不应设置 |

```yaml
tasks:
  - name: report
    type: report
    depends_on: [evaluate]
    options:
      expected_seeds: [42, 123]
```

## package 参数

package task 调用 `code/package_original_models_benchmark.py`。

| `options` 字段 | 类型 | 默认值 | CLI | 说明 |
|---|---|---|---|---|
| `input_root` | path | 根 `output_dir` | `--input-root` | benchmark 运行根目录 |
| `output_file` | path | `<input_root>/original_models_benchmark_report.zip` | `--output-file` | 最终 ZIP 文件 |

ZIP 只包含 `final_report/`、`metrics/`、`predictions/`、`training_logs/` 和 `config/`。`checkpoints/` 与 `features/` 不进入 ZIP。缺少任一必需目录时 package 会失败。

## Resume、失败和状态

| 配置 | 行为 |
|---|---|
| `global.resume: false` | 不读取前次 task 状态；底层 runner 不接收 `--resume` |
| `global.resume: true` | config digest 相同时跳过已完成的模型/seed，并向底层 runner 传递 `--resume` |
| task/model `resume: true` | 只控制对应底层 runner 的 checkpoint 恢复，不启用全局 task 状态跳过 |
| `global.fail_fast: true` | 首次失败后停止后续独立 run，并跳过依赖失败 task |
| `global.fail_fast: false` | 继续执行其他独立模型/seed；依赖失败 task 仍会跳过 |

运行产物：

```text
<output_dir>/
├── config/resolved_config.yaml
├── task_status.json
├── task_logs/<task>/<model>_seed_<seed>.log
├── task_plan.json                 # dry-run
└── task_plan_logs/                # dry-run
```

dry-run 不覆盖真实 `task_status.json` 和 `task_logs/`。不同实验应使用不同 `output_dir`；不要通过修改 resolved config 或复制不完整 checkpoint 强行恢复。

## 常见配置错误

| 错误 | 修正方法 |
|---|---|
| YAML 中同时填写 `seed` 和 `seeds` | 每一层只保留其中一个 |
| train/evaluate 找不到 clean metadata | 修正 `data_root`，它不是归档目录或 taskmanager 输出目录 |
| noisy/denoised 参数不完整 | 使用准备器生成的 `data_config`，或同时提供四个 manifest/root 字段 |
| `--task evaluate` 又执行了 train | 当前 YAML 中 evaluate 依赖 train；改用 `original_models_evaluate.yaml` |
| 修改模型列表后 resume 不跳过 | resolved config digest 已变化，这是预期行为 |
| EMD/CBAM/SE 模型提示 unknown model | 它们尚未纳入 taskmanager，应使用对应直接 Python runner；EMD 数据工作流当前 blocked |
| AWS YAML 在本地指向 `/mnt/ecg` | AWS 模板只适用于已挂载该 EBS 路径的 Linux 实例 |
