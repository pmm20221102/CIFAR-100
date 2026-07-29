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
| ViT-Tiny | 4.79M | **68.24%** | 219 | 62.56% | 1.8079 |

> ConvNeXt achieves the best accuracy (76.26%) with EMA, Mixup, CutMix, and a larger model configuration trained on Google Colab T4 GPU.
>
> ConvNeXt 通过 EMA、Mixup、CutMix 和更大的模型配置（在 Google Colab T4 GPU 上训练）达到了最高准确率（76.26%）。

---

## Project Structure / 项目结构

```
cifar100/
├── config.yaml              # Training hyperparameters (唯一配置源)
├── config_colab_vit.yaml    # Colab ViT training config
├── train.py                 # General training script (通用训练脚本)
├── load_cifar100.py         # Data loading with augmentation
├── mobilenetv1.py           # MobileNetV1 → mobilenetv1_small()
├── mobilevit.py             # MobileViT → mobilevit_small()
├── convnext_cifar.py        # ConvNeXt → convnext_tiny_cifar100()
├── wide_resnet.py           # WideResNet → wide_resnet_cifar100()
├── vit_cifar.py             # ViT → vit_tiny_cifar100()
├── models/                  # Checkpoints / 模型检查点
├── logs/                    # TensorBoard logs / 日志
├── train_colab_vit.ipynb     # ViT Colab training notebook (T4 实测)
├── train_local_vit.ipynb     # ViT local debug notebook
├── train_colab.ipynb        # Legacy Colab training notebook
├── config_colab.yaml        # Colab config (legacy)
├── cifar100_code.zip        # Colab deploy archive (not in repo)
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

### ViT-Tiny (Pure Vision Transformer)
Pure Transformer Encoder for image classification. Patch Embedding (4x4) + learnable CLS token + positional encoding + Multi-Head Self-Attention + MLP. 6 layers, 8 heads, embed_dim=256, ~4.79M params.

纯 Transformer Encoder 图像分类模型。Patch Embedding (4x4) + 可学习 CLS token + 位置编码 + 多头自注意力 + MLP。6 层，8 头，embed_dim=256，约 4.79M 参数。

> **Note / 注意**: ViT is a pure Transformer (patch tokens + attention), while MobileViT is a CNN+Transformer hybrid.
>
> ViT 是纯 Transformer（patch token + 注意力），MobileViT 是 CNN+Transformer 混合架构。

---

## Training

### Switch model (切换模型)

Edit `active_model` in `config.yaml`:

修改 `config.yaml` 中的 `active_model`：

```yaml
active_model: vit  # mobilenet | mobilevit | convnext | wide_resnet | vit
```

### Run training (运行训练)

```bash
# General training (supports all models)
python train.py

# Custom config
python train.py --config config_colab_vit.yaml

# Resume from checkpoint
python train.py --resume models/vit/cifar100_last.pth

# Resume + Drive sync (Colab)
python train.py --config config_colab_vit.yaml \
  --resume /content/drive/MyDrive/cifar100_runs/vit/cifar100_last.pth \
  --drive-sync /content/drive/MyDrive/cifar100_runs/vit
```

### Google Colab

ViT Colab config: `config_colab_vit.yaml` (AdamW, AMP, EMA, batch 1024). Trained on Tesla T4, early-stopped at epoch 269/300.

See [README_colab.md](README_colab.md) for general Colab training instructions.

---

## Key Features / 核心特性

- **Data Augmentation / 数据增强**: RandomCrop, RandomHorizontalFlip, RandAugment, RandomErasing
- **Training Strategy / 训练策略**: Warmup + Cosine Annealing, Early Stopping, Label Smoothing
- **Advanced Training / 高级训练**: AdamW optimizer, AMP (mixed precision), Mixup, CutMix, EMA, Gradient Clipping, Gradient Accumulation
- **Single Config / 统一配置**: All hyperparameters managed in `config.yaml`
- **Checkpoint / 断点续训**: Full state save/restore (model, optimizer, scheduler, scaler, EMA, epoch, best accuracy)
- **Colab Support / Colab 支持**: Resume training + Google Drive sync via `--drive-sync`
