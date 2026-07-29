# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

CIFAR-100 图像分类项目，使用 PyTorch 训练多种轻量级 CNN/ViT 模型。当前处于实验对比阶段，已训练模型包括 MobileNet、MobileViT、WideResNet、ConvNeXt、ViT。

## Training Commands

```bash
# 通用训练（通过 config.yaml 中 active_model 切换模型）
python train.py

# 使用自定义配置
python train.py --config config_colab_vit.yaml

# 断点续训
python train.py --config config_colab_vit.yaml --resume models/vit/cifar100_last.pth --drive-sync /content/drive/MyDrive/cifar100_runs/vit
```

在 [config.yaml](config.yaml) 中修改 `active_model` 字段切换模型：`mobilenet`、`mobilevit`、`convnext`、`wide_resnet`、`vit`。

## Architecture

- **[config.yaml](config.yaml)** — 所有训练/数据/模型超参数的唯一配置源，通过 `active_model` 选择当前模型
- **[train.py](train.py)** — 通用训练脚本，支持所有模型。特性：AdamW/Adam 优化器、AMP 混合精度、Mixup/CutMix、EMA、梯度裁剪、梯度累积、断点续训、Drive 同步、TensorBoard 日志、训练摘要 JSON
- **[load_cifar100.py](load_cifar100.py)** — 数据加载，训练集增强包含 RandomCrop、RandomHorizontalFlip、RandAugment、RandomErasing；测试集仅 Normalize
- **Model files**：每个模型一个文件，导出工厂函数：
  - [mobilenetv1.py](mobilenetv1.py) → `mobilenetv1_small()`
  - [mobilevit.py](mobilevit.py) → `mobilevit_small()`
  - [wide_resnet.py](wide_resnet.py) → `wide_resnet_cifar100()`
  - [convnext_cifar.py](convnext_cifar.py) → `convnext_tiny_cifar100()`
  - [vit_cifar.py](vit_cifar.py) → `vit_tiny_cifar100()`

## ViT 模型说明

`vit_cifar.py` 实现了纯 Vision Transformer（非 CNN+Transformer 混合），包含：
- Patch Embedding（Conv2d 等价实现）
- 可学习 CLS token + 位置编码
- Multi-Head Self-Attention（自实现 QKV 投影）
- Pre-Norm Transformer Encoder Block
- MLP/FFN + GELU
- DropPath（Stochastic Depth）
- 分类头（取 CLS token 输出）

默认配置：embed_dim=256, depth=6, num_heads=8, 约 4.79M 参数。

ViT 与 MobileViT 的区别：MobileViT 是 CNN+Transformer 混合架构，ViT 是纯 Transformer Encoder。

## 训练配置说明

`train.py` 通过 config.yaml 的 `train` 字段控制训练行为：

```yaml
train:
  optimizer: adamw    # adam | adamw（ViT 推荐 adamw）
  amp: true           # 混合精度（需要 CUDA）
  use_ema: true       # 指数移动平均
  grad_clip_norm: 1.0 # 梯度裁剪（0 表示禁用）
  accumulation_steps: 1  # 梯度累积步数
  mixup_alpha: 0.2    # Mixup 强度
  cutmix_prob: 0.5    # CutMix 概率
```

## Colab 训练

- 配置文件：`config_colab_vit.yaml`
- 支持 `--resume` 断点续训和 `--drive-sync` Google Drive 同步
- 默认 batch 256 + AMP，可根据显存调整

## Data & Model Layout

- 数据集路径：`D:\Study\cifar100\cifar-100-python\`（本地）或 `/content/cifar100_data`（Colab）
- 模型 checkpoint 保存在 `models/<model_name>/`
- TensorBoard 日志保存在 `logs/<model_name>/`
- 训练摘要保存在 `results/<model_name>_summary.json`

## Key Conventions

- 所有模型输出 100 类（CIFAR-100）
- 输入统一为 32×32 RGB 图像
- 模型超参数通过 config.yaml 的 `models.<name>.params` 传入工厂函数，使用 `**kwargs` 解包
- 训练日志和 checkpoint 按模型名分子目录管理
- ViT 使用 AdamW + AMP + EMA，其他模型默认 Adam
- 保存的 checkpoint 包含：model、optimizer、scheduler、scaler（AMP）、ema（如启用）、epoch、best accuracy、patience、global_step
