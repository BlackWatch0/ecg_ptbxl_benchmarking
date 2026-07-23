# AWS 运行环境

## 架构边界

本项目在单台 EC2 GPU 实例上使用本地加密 EBS 运行：代码、数据、缓存、checkpoint、日志和结果都以普通本地路径提供给 Python。S3 只作为人工控制的传入/传出存储，操作者在任务前后显式运行 `aws s3 sync`。

当前代码和 taskmanager 均不实现原生 S3：受支持配置只使用本地路径，不自动拉取资产，不自动上传 checkpoint，也不把 S3 当作可随机访问文件系统。不要使用 s3fs 等挂载替代 EBS。

## EC2 示例

推荐起点为 `g5.4xlarge`：x86_64、16 vCPU、64 GiB 内存、1 张 NVIDIA A10G 24 GiB GPU。实例可用性、GPU 配额和 Spot 容量因 Region/AZ 而异，启动前检查 `Running On-Demand G and VT instances` vCPU quota；长训练若不能容忍中断，优先 On-Demand。

AMI 有两种选择：

| 选择 | 适用场景 | 注意事项 |
|---|---|---|
| 当前 Region 的 AWS Deep Learning AMI GPU PyTorch | 希望复用已安装的 NVIDIA driver、CUDA 和 PyTorch | AMI 名称和 ID 随 Region/日期变化，启动时从 AWS 官方 DLAMI 列表选择，不在文档中固定 AMI ID |
| Ubuntu Server 22.04 LTS x86_64 | 需要完全控制依赖 | 需自行安装匹配 A10G 的 NVIDIA driver/CUDA/PyTorch；先验证 `nvidia-smi` 和 CUDA smoke |

`environments/ecg-training.yml` 是 Python 3.10、PyTorch 2.5、CUDA 12.1 的当前训练环境。`environments/legacy/` 中的 Python 3.8、PyTorch 1.4 和 fastai v1 环境不应作为 A10G 当前训练环境的默认方案，只用于确有需要的旧 fastai 路径。

建议实例配置：

- 使用无公网入站规则的安全组，通过 Systems Manager Session Manager 登录，不开放 SSH 22。
- 使用 instance profile 提供 SSM 和限定前缀的 S3 权限，不在实例保存 access key。
- 要求 IMDSv2：`HttpTokens=required`、`HttpEndpoint=enabled`。本项目不需要容器，hop limit 可保持 1。
- 根卷建议至少 100 GiB；另挂 500 GiB 或按数据规模调整的加密 gp3 数据卷。
- gp3 可从 3,000 IOPS、125 MiB/s 起步，依据 waveform 解码、缓存和 checkpoint 观测结果再调优。
- EBS 使用默认加密或客户管理 KMS key，并启用终止保护；注意实例终止时卷的 `DeleteOnTermination` 设置。

以下 AWS CLI 仅展示关键启动参数，需替换 AMI、子网、安全组和 instance profile：

```bash
aws ec2 run-instances \
  --region us-east-1 \
  --image-id ami-REPLACE_WITH_CURRENT_DLAMI_OR_UBUNTU \
  --instance-type g5.4xlarge \
  --subnet-id subnet-REPLACE \
  --security-group-ids sg-REPLACE \
  --iam-instance-profile Name=ecg-benchmark-instance-profile \
  --metadata-options HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1 \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","Encrypted":true,"DeleteOnTermination":true}},{"DeviceName":"/dev/sdf","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Iops":3000,"Throughput":125,"Encrypted":true,"DeleteOnTermination":false}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ecg-ptbxl-benchmark},{Key=Project,Value=ecg-ptbxl}]'
```

## Instance Profile

EC2 role 的 trust principal 为 `ec2.amazonaws.com`。附加 AWS 托管策略 `AmazonSSMManagedInstanceCore`，再添加只允许项目 bucket 前缀的内联/客户管理策略。不要附加 `AmazonS3FullAccess`。

最小 S3 示例需要替换 bucket 名：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListProjectPrefixes",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::REPLACE_BUCKET",
      "Condition": {
        "StringLike": {
          "s3:prefix": ["ecg/assets", "ecg/assets/*", "ecg/runs", "ecg/runs/*"]
        }
      }
    },
    {
      "Sid": "ReadAssets",
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::REPLACE_BUCKET/ecg/assets/*"
    },
    {
      "Sid": "ReadWriteRuns",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
      "Resource": "arn:aws:s3:::REPLACE_BUCKET/ecg/runs/*"
    }
  ]
}
```

不授予 `s3:DeleteObject`，同步命令也不使用 `--delete`。若 bucket 使用客户管理 KMS key，还需给 role 增加针对该 key 的最小 `kms:Decrypt`、`kms:Encrypt`、`kms:GenerateDataKey` 权限，并同步修改 key policy。

启动后确认身份与 SSM：

```bash
aws sts get-caller-identity
sudo systemctl status amazon-ssm-agent
nvidia-smi
```

DLAMI/Ubuntu 通常提供或支持 SSM Agent，但必须实际检查 managed node 状态。私有子网还需要 NAT，或为 SSM、SSM Messages、EC2 Messages 和 S3 配置相应 VPC endpoint；安装系统/Conda/PyPI 包仍需可达的软件源。

## 挂载加密 gp3

先用 `lsblk -f` 识别新卷。Nitro 实例上的 `/dev/sdf` 通常显示为 `/dev/nvme1n1`，但不要假定设备名。

只对确认是全新且无文件系统的数据卷执行一次格式化：

```bash
lsblk -f
sudo mkfs.xfs /dev/nvme1n1
sudo mkdir -p /mnt/ecg
sudo mount /dev/nvme1n1 /mnt/ecg
sudo chown ubuntu:ubuntu /mnt/ecg
sudo blkid /dev/nvme1n1
```

把 `blkid` 返回的 UUID 写入 `/etc/fstab`，使用 `defaults,nofail`，然后执行 `sudo mount -a` 和 `findmnt /mnt/ecg` 验证。已有文件系统或恢复卷绝对不能重新 `mkfs`。

创建本地层级：

```bash
mkdir -p /mnt/ecg/workspace /mnt/ecg/downloads /mnt/ecg/data /mnt/ecg/cache /mnt/ecg/runs
```

## 代码与环境

在 EBS 上 clone 固定 commit，不直接在临时根盘运行长任务：

```bash
git clone https://github.com/BlackWatch0/ecg_ptbxl_benchmarking.git /mnt/ecg/workspace/ecg_ptbxl_benchmarking
cd /mnt/ecg/workspace/ecg_ptbxl_benchmarking
git rev-parse HEAD
```

DLAMI 可创建仓库当前 Conda 环境；Ubuntu 应先安装 Miniconda/Conda，再创建同一环境。PyTorch CUDA 包仍要求主机 NVIDIA driver 足够新。创建并记录环境：

```bash
cd /mnt/ecg/workspace/ecg_ptbxl_benchmarking
conda env create -f environments/ecg-training.yml
conda activate ecg-training
```

无论采用哪种 AMI，都记录环境：

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip freeze > /mnt/ecg/runs/environment-pip-freeze.txt
nvidia-smi > /mnt/ecg/runs/environment-nvidia-smi.txt
```

不要为了兼容旧 fastai 降级 `ecg-training` 环境；需要旧入口时创建独立 legacy 环境。

## 人工同步数据

先检查 instance profile，再从 S3 同步到 EBS。命令由操作者执行，不由 Python 或 taskmanager 触发：

```bash
aws s3 sync s3://REPLACE_BUCKET/ecg/assets/ /mnt/ecg/downloads/ --no-progress
```

对照资产清单检查文件大小和 SHA-256，再解压/准备到 `/mnt/ecg/data`。不要把未解压归档、运行数据和缓存混在同一目录。资产与 ID 校验要求见 [`DATA_ASSETS.md`](DATA_ASSETS.md)。

## Smoke

从仓库根目录确认 GPU 和导入，再运行目标入口的 smoke。七模型基准示例：

```bash
cd /mnt/ecg/workspace/ecg_ptbxl_benchmarking
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python code/run_original_models_benchmark.py \
  --data-root /mnt/ecg/data/ptbxl \
  --data-config /mnt/ecg/workspace/normalized/original_models_benchmark_data.json \
  --output-dir /mnt/ecg/runs/original-models-smoke \
  --smoke-test
```

只有 smoke 明确完成且输出目录、类别顺序、数据 split 和有限 logits/loss 均正确后，才启动正式训练。EMD smoke 当前也 blocked，不能因模型能做 dummy forward 就视为数据工作流可用。

## 长任务、日志与恢复

复制 AWS 配置模板并填写实际 EBS 路径：

```bash
cp configs/taskmanager/aws_original_models.example.yaml /mnt/ecg/workspace/original_models.yaml
python taskmanager.py validate --config /mnt/ecg/workspace/original_models.yaml
python taskmanager.py run --config /mnt/ecg/workspace/original_models.yaml --dry-run
```

使用 `tmux` 保持会话，日志和 checkpoint 写 EBS：

```bash
tmux new -s ecg-benchmark
cd /mnt/ecg/workspace/ecg_ptbxl_benchmarking
python taskmanager.py run --config /mnt/ecg/workspace/original_models.yaml
```

SSM 断连后重新进入实例并执行 `tmux attach -t ecg-benchmark`。实例重启后，先确认 `/mnt/ecg` 已挂载、GPU 正常、数据配置未变化，再对同一输出目录使用 `--resume`。不要通过复制不完整 checkpoint 或修改 seed/config 强行恢复。

taskmanager 只操作本地 EBS，不接管 S3 同步；其当前命令和恢复语义见 [`TASK_MANAGER.md`](TASK_MANAGER.md)。

## 人工同步日志与结果

任务停止写入后，将单个 run 目录同步到唯一 S3 前缀：

```bash
aws s3 sync /mnt/ecg/runs/original-models-seed42/ s3://REPLACE_BUCKET/ecg/runs/original-models-seed42/ --no-progress
```

同步后执行只读核对：

```bash
aws s3 ls s3://REPLACE_BUCKET/ecg/runs/original-models-seed42/ --recursive --summarize
```

checkpoint 可在训练过程中人工增量同步，但应先确认文件已原子落盘；不要同步临时 `.part` 文件。终止实例前确认最终报告、metrics、predictions、training logs、resolved config 和必要 checkpoint 均已上传，并保留 EBS 卷直到 S3 核对完成。

## 成本与停机

- GPU 实例空闲时先 stop；确认数据卷和终止策略后再 terminate。
- EBS、快照、S3 和停止实例保留的 EBS 仍会计费。
- Spot 只适合 checkpoint/resume 已实际验证的工作流。
- 使用 AWS Budget/成本标签监控 `Project=ecg-ptbxl`，避免遗留 g5 实例和大容量卷。
