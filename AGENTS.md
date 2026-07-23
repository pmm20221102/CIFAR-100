# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

CIFAR-100 图像分类项目，使用 PyTorch 训练多种轻量级 CNN/ViT 模型。当前处于实验对比阶段，已训练模型包括 MobileNet、MobileViT、WideResNet、ConvNeXt。

## Training Commands

```bash
# 通用训练（通过 config.yaml 中 active_model 切换模型）
python train.py

# ConvNeXt 专用训练脚本（支持 Mixup/CutMix/EMA）
python train_convnext.py
```

在 [config.yaml](config.yaml) 中修改 `active_model` 字段切换模型：`mobilenet`、`mobilevit`、`convnext`、`wide_resnet`。

## Architecture

- **[config.yaml](config.yaml)** — 所有训练/数据/模型超参数的唯一配置源，通过 `active_model` 选择当前模型
- **[train.py](train.py)** — 通用训练脚本，从 config.yaml 读取配置，通过 `build_model()` 工厂函数实例化模型。支持 warmup + cosine annealing、early stopping、label smoothing。Mixup/CutMix/EMA 代码已实现但当前禁用
- **[train_convnext.py](train_convnext.py)** — ConvNeXt 专用训练脚本，与 train.py 功能相同但启用了 Mixup/CutMix/EMA，使用 AdamW 优化器和梯度裁剪
- **[load_cifar100.py](load_cifar100.py)** — 数据加载，训练集增强包含 RandomCrop、RandomHorizontalFlip、RandAugment、RandomErasing；测试集仅 Normalize
- **Model files**：每个模型一个文件，导出工厂函数：
  - [mobilenetv1.py](mobilenetv1.py) → `mobilenetv1_small()`
  - [mobilevit.py](mobilevit.py) → `mobilevit_small()`
  - [wide_resnet.py](wide_resnet.py) → `wide_resnet_cifar100()`
  - [convnext_cifar.py](convnext_cifar.py) → `convnext_tiny_cifar100()`

## Data & Model Layout

- 数据集路径：`D:\Study\cifar100\cifar-100-python\`（需要手动下载 CIFAR-100 原始数据）
- 模型 checkpoint 保存在 `models/<model_name>/`
- TensorBoard 日志保存在 `logs/<model_name>/`

## Key Conventions

- 所有模型输出 100 类（CIFAR-100）
- 输入统一为 32×32 RGB 图像
- 模型超参数通过 config.yaml 的 `models.<name>.params` 传入工厂函数，使用 `**kwargs` 解包
- 训练日志和 checkpoint 按模型名分子目录管理
- 两个训练脚本有大量重复代码（format_seconds、mixup/cutmix 函数等），修改训练逻辑时需注意同步
