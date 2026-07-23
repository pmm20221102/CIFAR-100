# CIFAR-100 Image Classification

# CIFAR-100 图像分类

A PyTorch project for benchmarking lightweight CNN and Vision Transformer models on the CIFAR-100 dataset.

一个基于 PyTorch 的项目，用于在 CIFAR-100 数据集上对比多种轻量级 CNN 和 Vision Transformer 模型。

---

## Test Results / 测试结果

| Model / 模型 | Params / 参数量 | Best Test Acc / 最佳测试准确率 | Best Epoch / 最佳轮次 | Train Acc / 训练准确率 | Best Test Loss / 最佳测试损失 |
|---|---|---|---|---|---|
| MobileNetV1 | 1.95M | **73.09%** | 184 | 93.83% | 1.4513 |
| MobileViT | 1.39M | **70.99%** | 193 | 73.12% | 1.4090 |
| WideResNet-16-6 | 10.95M | **74.10%** | 184 | 84.54% | 1.3573 |
| ConvNeXt-Tiny | 3.70M | **76.26%** | 243 | 62.87% | 1.2775 |

> ConvNeXt achieves the best accuracy (76.26%) with EMA, Mixup, CutMix, and a larger model configuration trained on Google Colab T4 GPU.
>
> ConvNeXt 通过 EMA、Mixup、CutMix 和更大的模型配置（在 Google Colab T4 GPU 上训练）达到了最高准确率（76.26%）。

---

## Project Structure / 项目结构

```
cifar100/
├── config.yaml              # Training hyperparameters (唯一配置源)
├── train.py                 # General training script (通用训练脚本)
├── train_convnext.py        # ConvNeXt training with Mixup/CutMix/EMA
├── load_cifar100.py         # Data loading with augmentation
├── mobilenetv1.py           # MobileNetV1 → mobilenetv1_small()
├── mobilevit.py             # MobileViT → mobilevit_small()
├── convnext_cifar.py        # ConvNeXt → convnext_tiny_cifar100()
├── wide_resnet.py           # WideResNet → wide_resnet_cifar100()
├── models/                  # Checkpoints / 模型检查点
│   ├── mobilenet/
│   ├── mobilevit/
│   ├── wide_resnet/
│   └── convnext/
├── logs/                    # TensorBoard logs / 日志
│   ├── mobilenet/
│   ├── mobilevit/
│   ├── wide_resnet/
│   └── convnext/
├── train_colab.ipynb        # Google Colab training notebook
├── config_colab.yaml        # Colab config (larger batch + AMP)
└── cifar-100-python/        # Dataset (not in repo)
```

---

## Models / 模型

### MobileNetV1
Depthwise separable convolution network. Width multiplier 1.25, dropout 0.2.

使用深度可分离卷积的网络，宽度乘数 1.25，dropout 0.2。

### MobileViT
Hybrid CNN + Transformer architecture combining MobileNet blocks with attention layers.

混合 CNN + Transformer 架构，结合 MobileNet 块与注意力层。

### WideResNet-16-6
Wide residual network (depth=16, widen_factor=6, dropout=0.3). Classic CNN baseline.

宽残差网络（深度 16，扩展因子 6，dropout 0.3），经典 CNN 基线模型。

### ConvNeXt-Tiny
Modern pure-ConvNet with large-kernel depthwise convolution, layer scale, and drop path. Configured with depths=[2,2,6,2] and dims=[32,64,160,320].

现代纯卷积网络，使用大核深度卷积、layer scale 和 drop path。配置为 depths=[2,2,6,2]，dims=[32,64,160,320]。

---

## Training

### Switch model (切换模型)

Edit `active_model` in `config.yaml`:

修改 `config.yaml` 中的 `active_model`：

```yaml
active_model: convnext  # mobilenet | mobilevit | convnext | wide_resnet
```

### Run training (运行训练)

```bash
# General training (for all models except ConvNeXt)
python train.py

# ConvNeXt training (with Mixup/CutMix/EMA)
python train_convnext.py
```

### Google Colab

See [README_colab.md](README_colab.md) for T4 GPU training with checkpoint syncing to Google Drive.

参见 [README_colab.md](README_colab.md) 了解 T4 GPU 训练及 Google Drive 断点同步。

---

## Key Features / 核心特性

- **Data Augmentation / 数据增强**: RandomCrop, RandomHorizontalFlip, RandAugment, RandomErasing
- **Training Strategy / 训练策略**: Warmup + Cosine Annealing, Early Stopping, Label Smoothing
- **ConvNeXt Extras / ConvNeXt 附加**: Mixup, CutMix, EMA, AMP (mixed precision), Gradient Clipping
- **Single Config / 统一配置**: All hyperparameters managed in `config.yaml`
- **Colab Support / Colab 支持**: Resume training + Google Drive sync via notebook
