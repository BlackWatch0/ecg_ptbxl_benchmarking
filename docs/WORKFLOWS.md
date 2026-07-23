# 工作流

## 原则

当前受支持入口是 Python 模块、taskmanager 和结构化配置。[`scripts/legacy/`](../scripts/legacy/README.md) 仅归档，不应执行。所有入口都应先使用本地数据和独立 smoke 输出目录验证，再启动长任务。

## 环境确认

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

当前训练环境使用 `environments/ecg-training.yml`。旧 fastai 推理使用 `environments/legacy/` 中的 Python 3.8 环境；不能仅凭环境名判断 checkpoint 与运行时兼容。

AWS 正式训练前运行 `python code/check_training_environment.py --require-cuda --check-compute`，并确认 TensorFlow GPU 列表为空。完整依赖边界见 [`DEPENDENCIES.md`](DEPENDENCIES.md)。

## Taskmanager

数据准备完成并通过 smoke 后，可使用 taskmanager 编排七模型训练、评估和报告：

```bash
python taskmanager.py validate --config configs/taskmanager/original_models_benchmark.yaml
python taskmanager.py run --config configs/taskmanager/original_models_benchmark.yaml --dry-run
python taskmanager.py run --config configs/taskmanager/original_models_benchmark.yaml
```

它不会执行 smoke，也不覆盖统一评估和独立推理。配置 schema、状态、日志和恢复语义见 [`TASK_MANAGER.md`](TASK_MANAGER.md)。

## 原作者七模型基准

数据准备和训练是两个 Python 阶段。先把 clean、noisy、denoised 归档放在本地磁盘，再运行准备器生成数据配置：

```bash
python code/prepare_original_models_benchmark_data.py \
  --archive /mnt/ecg/downloads/ptbxl_original_database_plus_mixed_WFDB.tar \
  --archive /mnt/ecg/downloads/denoised_WFDB.tar \
  --search-root /mnt/ecg/data \
  --workspace /mnt/ecg/workspace/prepared \
  --output-dir /mnt/ecg/workspace/normalized
```

先执行 smoke：

```bash
python code/run_original_models_benchmark.py \
  --data-root /mnt/ecg/data/ptbxl \
  --data-config /mnt/ecg/workspace/normalized/original_models_benchmark_data.json \
  --output-dir /mnt/ecg/runs/original-models-smoke \
  --smoke-test
```

正式运行和恢复使用同一输出目录，并由 taskmanager 分模型记录状态：

```bash
python taskmanager.py run \
  --config configs/taskmanager/aws_original_models.example.yaml
```

`--resume` 不代替产物检查。重启前应确认 resolved config、checkpoint、threshold、训练日志和数据配置仍来自同一次运行。

## 统一评估

复制并修改 `configs/evaluation/default.yaml`，确保 scenario NPZ、checkpoint、类别顺序和输出目录均为实际绝对路径，然后运行：

```bash
python evaluation/evaluate.py --config /mnt/ecg/workspace/evaluation.yaml
```

所有 scenario 必须具有一致的 sample ID 顺序和标签。`strict_checkpoint` 应保持启用；测试阈值应从验证集文件加载或使用预先声明的固定值。

## 推理

fastai 与 Lightning 使用不同运行时：

```bash
cd code
python run_inference.py
python run_lightning_inference.py
```

这些历史入口包含固定或相对路径假设。运行前检查脚本中的数据、scaler、checkpoint 和输出位置，不要让两种环境共用不兼容的 PyTorch 安装。

## EMD 工作流

`code/run_cbam_emd_experiment.py`、`code/run_ablation_study.py` 中的 EMD 变体及 EMD SNR 评估当前 **blocked**。原因是 `configs/datasets.json` 仅有 `emd_features: source_required`，没有活动归档；不应执行训练命令，也不应使用 Wavelet 归档、time-domain 归档或缺失场景的 clean EMD 代替。

解除 blocked 前必须完成：

- 登记稳定来源、文件大小和 SHA-256；
- 验证 clean 及所需 SNR 的 11 个公共 feature columns；
- 按 `RecordNumber == ecg_id` 和导联顺序验证覆盖与唯一性；
- 使用 train-only scaler；
- 通过 feature-only 与 late-fusion smoke test。

## 日志、恢复与同步

- smoke 使用单独目录，禁止覆盖正式 checkpoint。
- 长任务使用 `python -u` 和 `tee -a` 保存完整 stdout/stderr。
- 定期检查 EBS 剩余空间、GPU 状态、最后 checkpoint 和错误目录。
- EC2 断连后通过 SSM 重新连接；任务进程应运行在 `tmux`/`systemd-run` 等独立会话中。
- S3 只做人工作业边界同步，具体命令见 [`AWS_SETUP.md`](AWS_SETUP.md)。同步结果前先确保写入已结束，并在同步后比较对象数量或校验和。

## Taskmanager 边界

当前 taskmanager 把 `prepare_data` 以及 `train -> evaluate -> report -> package` 映射到原作者七模型 Python 入口。smoke、统一评估和 fastai/Lightning 推理仍需直接调用对应 Python；S3 同步始终由操作者在任务生命周期之外执行。
