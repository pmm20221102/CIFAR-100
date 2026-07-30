# Pretrained ViT for CIFAR-100 (Local RTX 3060)

# 预训练 ViT 微调 CIFAR-100（本地 RTX 3060）

本目录在现有 CIFAR-100 项目基础上，新增**预训练 Vision Transformer 微调**实验，针对本地 **RTX 3060 6GB** 优化。

---

## 1. 安装依赖 / Install

```bash
pip install timm torch torchvision pyyaml tensorboard
```

---

## 2. 目录结构 / Directory Layout

```
pretrained_vit_local/
├── pretrained_vit.py                        # 模型构建、冻结/解冻工具
├── train_pretrained_vit.py                  # 训练脚本（支持 resume、TensorBoard、early stopping）
├── load_cifar100_pretrained_vit.py          # CIFAR-100 数据加载（224×224、ImageNet normalization）
├── evaluate_pretrained_vit.py               # 独立评测脚本（Top-1/Top-5、显存、耗时）
├── config_pretrained_vit_linear_probe.yaml  # DeiT-Tiny Linear Probe 配置
├── config_partial_finetune.yaml             # DeiT-Tiny Partial Finetune 配置
├── config_full_finetune.yaml                # DeiT-Tiny Full Finetune 配置
├── config_small_linear_probe.yaml           # DeiT-Small Linear Probe 配置
├── config_small_partial_finetune.yaml       # DeiT-Small Partial Finetune 配置
├── config_small_full_finetune.yaml          # DeiT-Small Full Finetune 配置
├── README_pretrained_vit.md                 # 本文档
├── models/                                  # checkpoint 保存目录（按模式分子目录）
├── logs/                                    # TensorBoard 日志
└── results/                                 # 训练摘要 JSON
```

---

## 3. 三种微调模式 / Three Fine-tuning Modes

每种模式有独立的配置文件和独立的输出目录，互不干扰。

Each mode has its own config file and output directory — no overlap.

### 模式 A：Linear Probing（线性探测）

冻结整个 backbone，只训练分类头。

Freeze the entire backbone, train only the classifier head.

- 配置文件 / Config: `config_pretrained_vit_linear_probe.yaml`
- 可训练参数 / Trainable: ~19K (0.35%)
- 显存 / VRAM: ~334 MB peak
- 每 epoch / Per epoch: ~3 分钟

### 模式 B：Partial Finetuning（部分解冻）

冻结前面的 transformer blocks，解冻最后 2 个 block + norm + 分类头。

Freeze early transformer blocks, unfreeze last 2 blocks + norm + head.

- 配置文件 / Config: `config_partial_finetune.yaml`
- 可训练参数 / Trainable: ~890K (16%)
- 显存 / VRAM: ~1-2 GB peak
- 每 epoch / Per epoch: ~3-4 分钟
- **推荐作为默认配置** / **Recommended as default**

### 模式 C：Full Finetuning（全量微调）

解冻全部参数，使用较小学习率。

Unfreeze all parameters, use smaller learning rate.

- 配置文件 / Config: `config_full_finetune.yaml`
- 可训练参数 / Trainable: 5.54M (100%)
- 显存 / VRAM: ~3-4 GB peak
- 每 epoch / Per epoch: ~8-10 分钟

---

## 4. 运行命令 / Commands

所有命令在本目录（`pretrained_vit_local/`）下执行。

All commands run from this directory (`pretrained_vit_local/`).

### DeiT-Tiny / 微型

```bash
# Linear Probe / 线性探测
python train_pretrained_vit.py --config config_pretrained_vit_linear_probe.yaml

# Partial Finetuning / 部分解冻
python train_pretrained_vit.py --config config_partial_finetune.yaml

# Full Finetuning / 全量微调
python train_pretrained_vit.py --config config_full_finetune.yaml
```

### DeiT-Small / 小型

```bash
# Linear Probe / 线性探测
python train_pretrained_vit.py --config config_small_linear_probe.yaml

# Partial Finetuning / 部分解冻
python train_pretrained_vit.py --config config_small_partial_finetune.yaml

# Full Finetuning / 全量微调
python train_pretrained_vit.py --config config_small_full_finetune.yaml
```

### 断点续训 / Resume from checkpoint

```bash
python train_pretrained_vit.py --config <config_file> --resume models/<output_dir>/last.pth
```

注意：resume 时必须使用与训练时相同的配置文件（相同的 mode）。

Note: the config file must match the one used during training (same mode).

### 评测 / Evaluate

```bash
python evaluate_pretrained_vit.py --config <config_file> --checkpoint models/<output_dir>/best.pth
```

---

## 5. Tiny vs Small 参数对比 / Tiny vs Small Comparison

### DeiT-Tiny（224×224，256 维，6 层，8 头）

| 项目 / Item | Linear Probe | Partial Finetune | Full Finetune |
|---|---|---|---|
| 配置文件 / Config | `config_pretrained_vit_linear_probe.yaml` | `config_partial_finetune.yaml` | `config_full_finetune.yaml` |
| 模式 / mode | `linear_probe` | `partial_finetune` | `full_finetune` |
| 总参数 / Total | 5.54M | 5.54M | 5.54M |
| 可训练参数 / Trainable | ~19K (0.35%) | ~890K (16%) | 5.54M (100%) |
| backbone lr | 无 / N/A | 5e-5 | 2e-5 |
| head lr | 1e-3 | 2e-4 | 1e-4 |
| epochs | 20 | 40 | 50 |
| warmup | 2 | 3 | 5 |
| early stop patience | 8 | 12 | 15 |
| train batch size | 64 | 128 | 64 |
| test batch size | 128 | 256 | 128 |
| gradient accumulation | 1 | 1 | 2 |

### DeiT-Small（224×224，384 维，12 层，6 头）

| 项目 / Item | Linear Probe | Partial Finetune | Full Finetune |
|---|---|---|---|
| 配置文件 / Config | `config_small_linear_probe.yaml` | `config_small_partial_finetune.yaml` | `config_small_full_finetune.yaml` |
| 模式 / mode | `linear_probe` | `partial_finetune` | `full_finetune` |
| 总参数 / Total | 22.1M | 22.1M | 22.1M |
| 可训练参数 / Trainable | ~38K (0.17%) | ~3.6M (16%) | 22.1M (100%) |
| backbone lr | 无 / N/A | 3e-5 | 1e-5 |
| head lr | 1e-3 | 1.5e-4 | 5e-5 |
| epochs | 20 | 40 | 50 |
| warmup | 2 | 3 | 5 |
| early stop patience | 8 | 12 | 15 |
| train batch size | 64 | 32 | 16 |
| test batch size | 128 | 64 | 32 |
| gradient accumulation | 1 | 2 | 4 |

---

## 6. 配置参数说明 / Config Reference

### 数据 / Data

```yaml
data:
  dataset_root: D:\Study\cifar100    # CIFAR-100 数据集根目录
  image_size: 224                      # 输入分辨率（224 兼容预训练权重）
  train_batch_size: 128                # 训练 batch（根据显存调整）
  test_batch_size: 256                 # 测试 batch
  num_workers: 4                       # 数据加载线程数
```

### 模型 / Model

```yaml
model:
  library: timm
  name: deit_tiny_patch16_224          # 预训练模型名称
  pretrained: true                     # 是否加载 ImageNet 预训练权重
  num_classes: 100                     # CIFAR-100 类别数
  mode: partial_finetune               # linear_probe | partial_finetune | full_finetune
  unfreeze_last_blocks: 2              # partial_finetune 时解冻的最后 N 个 block
```

### 训练 / Training

```yaml
train:
  epochs: 40                           # 总训练轮次
  optimizer: adamw                      # 优化器（仅支持 adamw）
  backbone_lr: 0.00005                 # backbone 学习率
  head_lr: 0.0002                      # 分类头学习率
  weight_decay: 0.05
  warmup_epochs: 3                     # warmup 轮次
  label_smoothing: 0.1                 # 标签平滑
  gradient_clip_norm: 1.0              # 梯度裁剪
  gradient_accumulation_steps: 1       # 梯度累积步数
  early_stop_patience: 12              # 早停耐心值
  early_stop_min_delta: 0.001          # 最小提升阈值
  seed: 42                             # 随机种子
```

### 输出 / Output

```yaml
output:
  model_dir: models/pretrained_vit_partial     # checkpoint 保存目录（各模式不同）
  log_dir: logs/pretrained_vit_partial         # TensorBoard 日志目录
  result_dir: results/pretrained_vit_partial   # 训练摘要保存目录
  save_every_epochs: 10                        # 每 N 个 epoch 保存一次周期 checkpoint
```

三种模式的输出目录分别命名为：

The output directories for each mode:

| 模式 | model_dir | log_dir | result_dir |
|---|---|---|---|
| Linear Probe | `models/pretrained_vit_linear_probe/` | `logs/pretrained_vit_linear_probe/` | `results/pretrained_vit_linear_probe/` |
| Partial Finetune | `models/pretrained_vit_partial/` | `logs/pretrained_vit_partial/` | `results/pretrained_vit_partial/` |
| Full Finetune | `models/pretrained_vit_full/` | `logs/pretrained_vit_full/` | `results/pretrained_vit_full/` |

---

## 7. 输出产物 / Outputs

每种模式各自生成：

Each mode generates independently:

| 文件 / File | 说明 / Description |
|---|---|
| `models/<output_dir>/best.pth` | 最佳模型 checkpoint / Best model checkpoint |
| `models/<output_dir>/last.pth` | 最新模型 checkpoint / Latest model checkpoint |
| `logs/<output_dir>/` | TensorBoard 日志 / TensorBoard logs |
| `results/<output_dir>/pretrained_vit_summary.json` | 训练摘要 / Training summary |
| `results/<output_dir>/eval_pretrained_vit.json` | 评测结果 / Evaluation results |

---

## 8. 评测结果 / Evaluation Results

### DeiT-Tiny vs DeiT-Small 对比 / Tiny vs Small Comparison

| 模型 / Model | 模式 / Mode | 总参数 / Total | 可训练 / Trainable | Top-1 | Top-5 | Test Loss | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| DeiT-Tiny | Linear Probe | 5.54M | 19K (0.35%) | **69.32%** | 92.23% | 1.1307 | 347 MB |
| DeiT-Tiny | Partial Finetune | 5.54M | 890K (16%) | **75.84%** | 95.07% | 0.9070 | 649 MB |
| DeiT-Tiny | Full Finetune | 5.54M | 5.54M (100%) | **84.96%** | 97.42% | 0.6262 | 389 MB |
| **DeiT-Small** | **Linear Probe** | **21.7M** | **38K (0.17%)** | **76.78%** | **94.81%** | **0.8104** | **692 MB** |
| **DeiT-Small** | **Partial Finetune** | **21.7M** | **3.59M (16%)** | **81.15%** | **96.79%** | **0.7291** | **720 MB** |
| **DeiT-Small** | **Full Finetune** | **21.7M** | **21.7M (100%)** | **88.66%** | **98.18%** | **0.5360** | **471 MB** |

### 与从零训练模型对比 / vs Training from Scratch

| 模型 / Model | 初始化 / Init | 输入 / Input | 参数量 / Params | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|
| MobileNetV1 | 随机 / Random | 32 | 1.95M | 73.09% | - |
| MobileViT | 随机 / Random | 32 | 1.39M | 70.99% | - |
| WideResNet-16-6 | 随机 / Random | 32 | 10.95M | 74.10% | - |
| ConvNeXt-Tiny | 随机 / Random | 32 | 3.70M | 76.26% | - |
| ViT-Tiny (从零) | 随机 / Random | 32 | 4.79M | 68.24% | - |
| DeiT-Tiny | ImageNet | 224 | 5.54M | 84.96% | 97.42% |
| **DeiT-Small** | **ImageNet** | **224** | **21.7M** | **88.66%** | **98.18%** |

### 关键结论 / Key Findings

1. **DeiT-Small Full Finetune 88.66% 是全场最高**，比之前最好的 ConvNeXt (76.26%) 高 12.4 个百分点
2. **DeiT-Small Linear Probe 76.78% 只用 38K 参数就超过 ConvNeXt (76.26%)**
3. **DeiT-Tiny Full Finetune 84.96% 也显著超过所有从零训练模型**
4. 预训练权重 + 224 输入 + ImageNet normalization 的迁移学习效果非常显著
5. Small 比 Tiny 的提升：Linear +7.4%、Partial +5.3%、Full +3.7%

---

## 9. RTX 3060 6GB 显存建议 / VRAM Tips

- 必须启用 AMP（默认已开启）/ AMP must be enabled (on by default)
- 224×224 输入比 32×32 占用更多显存 / 224×224 input uses more VRAM than 32×32
- 如果 OOM：先减小 batch size，再增加 gradient accumulation / If OOM: reduce batch size first, then increase grad accum
- 验证阶段使用 `torch.inference_mode()` 节省显存 / Eval uses `inference_mode` to save VRAM
- peak 显存目标 < 5.5GB / Peak VRAM target: < 5.5GB

---

## 9. 注意事项 / Notes

- 每次训练都从 timm 重新加载原始预训练权重，三种模式互不影响 / Each run loads fresh pretrained weights from timm — modes don't affect each other
- 使用 timm 的 ImageNet normalization，不使用 CIFAR-100 的 mean/std / Uses ImageNet normalization, not CIFAR-100's
- 第一版不使用 Mixup/CutMix/EMA，确保归因清晰 / First version skips Mixup/CutMix/EMA for clean attribution
- 不修改原有项目文件（train.py 等）/ Does not modify existing project files
- 输入 32×32 会 Resize 到 224×224 以兼容预训练权重 / 32×32 input is resized to 224×224 for pretrained weight compatibility
