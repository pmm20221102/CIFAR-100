# CIFAR-100 纯 ViT 实现与训练完善任务书

## 1. 任务目标

在当前 CIFAR-100 项目中新增一个**纯 Vision Transformer（ViT）图像分类模型**，用于和已有的 MobileNetV1、MobileViT、WideResNet、ConvNeXt 进行公平对比。

本次不是调用 `torchvision.models.vit_*` 或 `timm` 的现成模型，而是基于 PyTorch 自己实现 ViT 的关键结构，以便完整展示：

- Patch Embedding
- 可学习的 `[CLS]` token
- 可学习的位置编码
- Multi-Head Self-Attention
- Pre-Norm Transformer Encoder Block
- MLP/FFN
- Residual Connection
- Stochastic Depth / DropPath（推荐实现）
- 分类头

目标运行环境：

- 主要训练平台：Google Colab GPU Runtime
- 优先适配：NVIDIA T4 16 GB；同时兼容 Colab 可能分配的 L4、A100 等 GPU
- 本地 RTX 3060 6 GB 仅用于代码调试和小规模验证，不作为正式长时间训练平台
- 输入：CIFAR-100，`32×32×3`
- 类别数：100
- 框架：PyTorch
- Colab 临时工作目录：`/content/cifar100`
- Google Drive 持久化目录建议：`/content/drive/MyDrive/cifar100_runs`

注意：Transformer 的序列位置可以并行计算，但这不代表 ViT 一定比 CNN 训练更快。ViT 的 Self-Attention 仍需计算 token-token 关系，且从零训练通常需要更多 epoch。CIFAR-100 中序列长度只有 65，因此 Attention 开销不大；在 Colab T4 上的主要优势是 16 GB 显存、AMP 和更大的 batch，而不是“Transformer 天生必然更快”。

必须保证新增 ViT 后，原有 CNN/MobileViT/ConvNeXt/WideResNet 功能不被破坏。

---

## 2. 当前代码现状

当前工程已有：

- `config.yaml`：统一配置源
- `train.py`：通用训练脚本
- `train_convnext.py`：带 AMP、Mixup、CutMix、EMA、梯度裁剪、断点续训的强化训练脚本
- `load_cifar100.py`：数据加载及增强
- `mobilenetv1.py`
- `mobilevit.py`
- `convnext_cifar.py`
- `wide_resnet.py`

当前数据增强已经包含：

- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`
- `RandAugment(num_ops=2, magnitude=9)`
- CIFAR-100 Normalize
- `RandomErasing`

当前基础配置：

```yaml
train:
  epoch_num: 300
  lr: 0.002
  warmup_epochs: 5
  early_stop_patience: 40
  label_smoothing: 0.05
  weight_decay: 0.0005
  mixup_alpha: 0.1
  cutmix_alpha: 1.0
  cutmix_prob: 0.5
  ema_decay: 0.999
```

注意：

- `train.py` 中虽然定义了 Mixup/CutMix/EMA，但当前并未真正启用。
- `train_convnext.py` 已实现更完整的训练流程，但与 `train.py` 有较多重复代码。
- ViT 从零训练比 CNN 更依赖 AdamW、较大 weight decay、warmup、强数据增强和 AMP。

---

## 3. ViT 模型设计要求

新增文件：

```text
vit_cifar.py
```

并导出工厂函数：

```python
vit_tiny_cifar100(...)
```

### 3.1 推荐默认结构

针对 CIFAR-100 从零训练，并兼顾 Colab T4 与本地 RTX 3060 调试，默认使用轻量级 ViT-Tiny 配置：

```yaml
image_size: 32
patch_size: 4
in_channels: 3
num_classes: 100
embed_dim: 256
depth: 6
num_heads: 8
mlp_ratio: 4.0
dropout: 0.1
attention_dropout: 0.0
drop_path_rate: 0.1
```

对应维度：

- 每张图像：`32×32`
- Patch：`4×4`
- Patch 网格：`8×8`
- Patch token 数：64
- 加上 `[CLS]` 后序列长度：65
- 每个 head 维度：`256 / 8 = 32`
- FFN hidden dim：`256 × 4 = 1024`

默认模型参数量建议控制在约 **4M–7M**，避免在 CIFAR-100 上模型过大；该规模在 Colab T4 上可使用更大 batch，在本地 6 GB 显卡上也能完成调试。

### 3.2 Patch Embedding

使用以下等价方式之一：

```python
nn.Conv2d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=patch_size,
    stride=patch_size,
)
```

输出转换：

```text
[B, 3, 32, 32]
→ [B, D, 8, 8]
→ [B, 64, D]
```

必须检查：

- `image_size % patch_size == 0`
- forward 输入高度和宽度与配置一致，或者给出明确错误信息

### 3.3 CLS Token 与位置编码

实现可学习参数：

```python
cls_token: [1, 1, D]
pos_embed: [1, 65, D]
```

forward 中扩展 CLS token：

```text
[B, 64, D] → [B, 65, D]
```

位置编码使用可学习参数即可，不需要为了本任务引入复杂插值逻辑。

### 3.4 Multi-Head Self-Attention

优先自己实现 QKV 投影，便于学习和后续分析：

```python
qkv = nn.Linear(embed_dim, 3 * embed_dim)
```

张量形状必须清楚：

```text
输入            [B, N, D]
Q/K/V           [B, H, N, Dh]
Attention score [B, H, N, N]
输出            [B, N, D]
```

计算：

```text
softmax(QK^T / sqrt(Dh))V
```

本任务是 Encoder-only 图像分类：

- 不使用 causal mask
- 所有 patch token 可相互关注

如果使用 PyTorch 2.x，可在保持代码可读性的前提下使用 `torch.nn.functional.scaled_dot_product_attention` 作为可选加速路径，但不要让实现完全变成黑盒；至少保留明确的 Q/K/V reshape 逻辑。

### 3.5 Transformer Encoder Block

使用 Pre-Norm：

```python
x = x + drop_path(attn(norm1(x)))
x = x + drop_path(mlp(norm2(x)))
```

必须包含：

- `LayerNorm`
- Multi-Head Self-Attention
- 两层 MLP
- GELU
- Dropout
- Residual Connection

推荐加入 `DropPath`，并让各层 drop path rate 从 0 线性递增到 `drop_path_rate`。

若不引入 `timm`，请自行实现一个简洁可靠的 `DropPath`。

### 3.6 分类头

最后：

```python
x = final_norm(x)
cls_feature = x[:, 0]
logits = head(cls_feature)
```

输出：

```text
[B, 100]
```

模型末尾不要手动加 Softmax，因为训练使用 `CrossEntropyLoss`。

### 3.7 参数初始化

建议：

- Linear weight：`trunc_normal_(std=0.02)`
- Linear bias：0
- CLS token：`trunc_normal_(std=0.02)`
- Position embedding：`trunc_normal_(std=0.02)`
- LayerNorm weight：1
- LayerNorm bias：0
- Patch projection Conv2d：合理初始化，可使用 trunc normal 或 Kaiming

---

## 4. 配置文件修改

在 `config.yaml` 中新增：

```yaml
models:
  vit:
    params:
      image_size: 32
      patch_size: 4
      in_channels: 3
      num_classes: 100
      embed_dim: 256
      depth: 6
      num_heads: 8
      mlp_ratio: 4.0
      dropout: 0.1
      attention_dropout: 0.0
      drop_path_rate: 0.1
```

允许：

```yaml
active_model: vit
```

不要删除或改坏原有模型配置。

### 4.1 Google Colab 推荐训练配置

正式训练优先按 Colab T4 16 GB 设计。不要把本地 3060 6 GB 的保守 batch size 直接作为 Colab 默认值。

第一轮稳定配置：

```yaml
data:
  dataset_root: /content/cifar100_data
  train_batch_size: 256
  test_batch_size: 512
  num_workers: 2
  model_dir: /content/cifar100/models
  log_dir: /content/cifar100/logs

train:
  epoch_num: 300
  lr: 0.0005
  warmup_epochs: 10
  early_stop_patience: 50
  early_stop_min_delta: 0.001
  label_smoothing: 0.1
  weight_decay: 0.05
  mixup_alpha: 0.2
  cutmix_alpha: 1.0
  cutmix_prob: 0.5
  ema_decay: 0.999
  use_ema: true
  amp: true
  grad_clip_norm: 1.0
  accumulation_steps: 1
```

Colab T4 上先尝试：

```yaml
train_batch_size: 256
test_batch_size: 512
```

若峰值显存低于约 13 GB，可继续试：

```yaml
train_batch_size: 384
```

甚至：

```yaml
train_batch_size: 512
```

如果出现 OOM，则依次回退到 192、128。必须通过实测记录峰值显存，不能仅凭估计。

对于 Colab，建议有效 batch size 控制在 256～512。若物理 batch 较小，可用梯度累积：

```yaml
train_batch_size: 128
accumulation_steps: 2
```

得到有效 batch 256。

### 4.2 Colab 专用配置文件

建议新增：

```text
config_colab_vit.yaml
```

不要覆盖本地 `config.yaml`，以免 Windows 路径和 Colab 路径互相污染。`config_colab_vit.yaml` 应设置：

```yaml
active_model: vit

data:
  dataset_root: /content/cifar100_data
  train_batch_size: 256
  test_batch_size: 512
  num_workers: 2
  model_dir: /content/cifar100/models
  log_dir: /content/cifar100/logs
```

其余 ViT 和训练参数沿用本任务书推荐值。

### 4.3 Colab GPU 自适应

训练启动时打印：

- `torch.cuda.get_device_name(0)`
- CUDA 是否可用
- GPU 总显存
- PyTorch 与 CUDA 版本
- AMP 是否启用

不要硬编码“T4”。如果 Colab 分配到 L4/A100，可继续使用同一脚本，并允许通过配置增大 batch size。

---

## 5. 训练脚本修改要求

优先方案：完善 `train.py`，让它可以统一支持 ViT，同时不破坏已有模型。

### 5.1 模型注册

在 `train.py` 中导入：

```python
from vit_cifar import vit_tiny_cifar100
```

在 `build_model()` 中加入：

```python
if model_name == 'vit':
    return vit_tiny_cifar100(**model_params)
```

### 5.2 优化器

ViT 默认使用 AdamW，而不是当前 `train.py` 中的 Adam：

```python
optimizer = torch.optim.AdamW(
    net.parameters(),
    lr=lr,
    weight_decay=weight_decay,
    betas=(0.9, 0.999),
)
```

为了避免影响已有实验，可按模型选择：

- `vit`、`convnext`：AdamW
- 其他模型：保持原行为，或让 optimizer 类型进入配置

更推荐在配置中新增：

```yaml
train:
  optimizer: adamw
```

但修改时必须保持向后兼容。

### 5.3 AMP 混合精度

Colab GPU 正式训练必须支持 AMP；本地 RTX 3060 6 GB 调试也应启用 AMP。

当前 `train_convnext.py` 使用旧式：

```python
from torch.cuda.amp import autocast, GradScaler
```

可保持兼容，或者按当前 PyTorch 版本使用新版 `torch.amp` API。

要求：

- AMP 可由 `config.yaml -> train.amp` 控制
- CPU 环境自动禁用 AMP
- 保存并恢复 scaler state
- 验证阶段也可使用 autocast

### 5.4 Mixup/CutMix

当前 `train.py` 虽然有函数，但训练循环没有启用。

ViT 训练应真正启用：

- 按 `cutmix_prob` 随机选择 CutMix 或 Mixup
- loss 使用双标签加权
- 训练准确率使用加权统计，或明确标注为 mixed accuracy

注意 `CrossEntropyLoss(label_smoothing=...)` 与 Mixup/CutMix 可以共存，但需要保持逻辑清楚。

### 5.5 EMA

建议支持 EMA，但做成配置项：

```yaml
train:
  use_ema: true
```

验证时使用 EMA 模型。

如果为了减少显存，EMA 模型可保存在 CPU 上，但这会增加每步拷贝开销；对于当前小型 ViT，在 Colab T4 16 GB 上将 EMA 模型放在 GPU 通常可接受，但仍需实测。

如果显存紧张：

1. 优先减小 batch size；
2. 其次禁用 EMA；
3. 不要先大幅削弱模型结构。

### 5.6 梯度裁剪

使用：

```python
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
```

AMP 下必须先：

```python
scaler.unscale_(optimizer)
```

再裁剪。

### 5.7 Warmup + Cosine

当前 warmup 是按 epoch 手动修改 LR，随后调用 `CosineAnnealingLR`，基本可用，但请检查：

- warmup 结束后 LR 连续，不应突然跳变
- resume 后 scheduler 状态正确
- TensorBoard 记录真实 LR

可以保留现有方式，也可以实现组合 scheduler，但不要引入不必要依赖。

### 5.8 梯度累积

加入可选的 `accumulation_steps`：

```python
loss = loss / accumulation_steps
```

只有达到 accumulation step 或最后一个 batch 时才执行：

- `optimizer.step()` / `scaler.step()`
- `optimizer.zero_grad()`
- EMA 更新

统计日志时要使用未除以前的真实 loss。

### 5.9 断点续训

让通用训练脚本支持：

```bash
python train.py --config config_colab_vit.yaml --resume /content/drive/MyDrive/cifar100_runs/vit/cifar100_last.pth --drive-sync /content/drive/MyDrive/cifar100_runs/vit
```

checkpoint 至少保存：

- epoch
- model state
- EMA state（如果启用）
- optimizer state
- scheduler state
- GradScaler state
- best accuracy
- best epoch
- patience counter
- global step
- 当前配置或配置快照

同时保存：

- `cifar100_best.pth`
- `cifar100_last.pth`
- 每 10 或 20 epoch 的周期 checkpoint

### 5.10 显存观测

每个 epoch 或首次 forward 后打印：

- 模型参数量
- trainable 参数量
- `torch.cuda.max_memory_allocated()`
- `torch.cuda.max_memory_reserved()`

并在每个 epoch 开始前可调用：

```python
torch.cuda.reset_peak_memory_stats()
```

用于比较 Colab GPU 和本地 RTX 3060 的实际显存占用，并选择合适 batch size。

---

## 5.11 Google Colab Notebook 与持久化要求

建议新增或完善：

```text
train_colab_vit.ipynb
```

Notebook 必须包含以下单元，并保证重启 Runtime 后可以按顺序重新执行：

1. 检查 GPU：

```bash
!nvidia-smi
```

2. 挂载 Google Drive：

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. 获取项目代码：支持从 Git 仓库 clone，或从 Drive 复制到 `/content/cifar100`。训练期间代码与数据优先放在 `/content`，避免直接在 Drive 上高频读写。

4. 安装依赖：只安装项目缺失的最小依赖，不要无条件重装 PyTorch。

5. 自动下载 CIFAR-100：修改 `load_cifar100.py`，允许：

```python
CIFAR100(root=dataset_root, train=True, download=True, ...)
```

或者增加配置项：

```yaml
data:
  download: true
```

Colab 不应要求用户手动上传 `cifar-100-python`。

6. 启动训练并同步 checkpoint。

### Drive 同步策略

Colab Runtime 可能中断，因此必须把以下文件同步到 Drive：

- `cifar100_best.pth`
- `cifar100_last.pth`
- 配置快照
- 最终 metrics JSON/CSV

不要每个 iteration 写 Drive。建议：

- 每个 epoch 先保存到 `/content/cifar100/models/vit`
- 每个 epoch 只同步 `last`
- 出现新最佳时同步 `best`
- 周期 checkpoint 每 20 epoch 同步一次

### Colab 断点续训

Notebook 启动时检查 Drive 中是否存在：

```text
/content/drive/MyDrive/cifar100_runs/vit/cifar100_last.pth
```

若存在，允许用户选择自动 resume；恢复内容必须包含模型、优化器、scheduler、GradScaler、EMA、epoch 和 best accuracy。

### Runtime 防丢失

训练日志中每个 epoch 输出：

- epoch / total epochs
- train/test loss
- train/test accuracy
- best accuracy
- learning rate
- epoch time 与 ETA
- peak GPU memory

并将最终摘要写入：

```text
results/vit_summary.json
```

---

## 6. 数据增强要求

现有增强总体适合 ViT，可以保留。

但是需要做以下检查：

1. `RandAugment` 是否与当前 torchvision 版本兼容；
2. `RandomErasing` 必须位于 `ToTensor()` 和 Normalize 之后，当前顺序可用；
3. 不要对 test dataset 使用随机增强；
4. 可通过配置控制强增强，避免所有模型被强制绑定同一增强策略。

推荐后续增加可配置项，但本次不要求大规模重写数据加载器。

---

## 7. 必须完成的验证

### 7.1 Shape 单元测试

新增一个简单测试文件或在 `vit_cifar.py` 的 `__main__` 中验证：

```python
model = vit_tiny_cifar100()
x = torch.randn(8, 3, 32, 32)
y = model(x)
assert y.shape == (8, 100)
```

建议额外检查：

```text
Patch Embedding  [8, 64, 256]
含 CLS           [8, 65, 256]
最终 logits      [8, 100]
```

### 7.2 反向传播测试

使用随机输入执行一次：

```python
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()
```

确认所有关键参数有梯度且无 NaN/Inf。

### 7.3 小数据过拟合测试

在 128 或 256 张训练样本上临时训练，关闭过强增强，验证模型能把训练准确率提升到接近 100%。

如果无法过拟合小样本，优先排查：

- QKV reshape
- residual connection
- LayerNorm 位置
- loss 和标签
- optimizer step
- Mixup/CutMix 是否误开
- 位置编码和 CLS token

### 7.4 Colab 显存与吞吐测试

在 Colab 实际分配的 GPU 上验证：

- 先记录 GPU 型号和总显存
- T4 上测试 batch 256 + AMP
- 若显存充足，再测试 batch 384 / 512
- 若 OOM，依次回退到 192 / 128
- 记录每个 batch size 的峰值 allocated/reserved memory、images/s 和 epoch time
- 不允许仅凭估计声称可运行

本地 RTX 3060 6 GB 只需完成 batch 64～128 的 smoke test，不要求承担 300 epoch 正式训练。

### 7.5 训练趋势

至少进行 5–10 epoch smoke test，确认：

- loss 总体下降
- accuracy 总体上升
- learning rate 符合 warmup + cosine
- 无 NaN
- checkpoint 可保存和恢复
- TensorBoard 日志正常

---

## 8. 预期准确率与实验原则

纯 ViT 在 CIFAR-100 上从零训练，对数据增强和训练策略较敏感。即使 token 维度可并行，正式训练仍可能需要 200～300 epoch；不要把“Transformer 可并行”误解为“总训练时间一定短”。

不要为了快速得到高分而：

- 偷用测试集调参
- 加载未声明的 CIFAR-100 预训练权重
- 使用外部训练数据却不说明
- 将 MobileViT 冒充纯 ViT

合理目标：

- 首先保证实现正确、训练稳定；
- 基础配置争取达到约 65%–75% 测试准确率；
- 是否超过现有 ConvNeXt 76.26% 不是本次实现正确性的硬性要求。

需要在 README 中如实记录：

- 是否从零训练
- 是否使用 AMP
- 是否使用 Mixup/CutMix/EMA
- 最佳 epoch
- 参数量
- 峰值显存
- 最佳 Top-1 accuracy
- 训练时间

---

## 9. README 与工程文档更新

更新 `README.md`：

1. 在模型列表中增加 `ViT-Tiny`；
2. 在项目结构中增加 `vit_cifar.py`；
3. 模型切换说明增加 `vit`；
4. 写明纯 ViT 与 MobileViT 的区别：
   - `MobileViT` 是 CNN + Transformer 混合架构；
   - 新增 `ViT` 是基于 patch token 的纯 Transformer Encoder 分类模型；
5. 补充 Google Colab T4 推荐配置及本地 RTX 3060 调试配置；
6. 后续训练完成后再填真实准确率，不允许预填虚假结果。

更新 `AGENTS.md`：

- 注册新模型入口
- 写清训练命令
- 写清默认显存安全配置
- 写清 ViT 使用 AdamW + AMP

---

## 10. 建议的最终文件变化

```text
新增：
  vit_cifar.py

修改：
  config.yaml
  train.py
  README.md
  AGENTS.md

可选新增：
  tests/test_vit.py
  train_utils.py
  config_colab_vit.yaml
  train_colab_vit.ipynb
```

如果抽取 `train_utils.py`，只抽取明显重复且稳定的公共逻辑，例如：

- Mixup/CutMix
- EMA
- checkpoint save/load
- format_seconds

不要在本次任务中进行与 ViT 无关的大规模重构。

---

## 11. 验收标准

完成后必须给出一份简洁执行报告，包含：

1. 修改了哪些文件；
2. ViT 结构和参数量；
3. 默认 tensor shape 流程；
4. Colab 实际 GPU 型号、实测可用 batch size、峰值显存与吞吐；
5. AMP 开关是否正常；
6. smoke test 结果；
7. checkpoint 恢复测试结果；
8. 是否存在尚未解决的问题；
9. 给出正式训练命令。

正式 Colab 训练命令期望类似：

```bash
python train.py --config config_colab_vit.yaml --drive-sync /content/drive/MyDrive/cifar100_runs/vit
```

断点续训：

```bash
python train.py --config config_colab_vit.yaml --resume /content/drive/MyDrive/cifar100_runs/vit/cifar100_last.pth --drive-sync /content/drive/MyDrive/cifar100_runs/vit
```

---

## 12. 实施优先级

按以下顺序执行，不要一开始就追求最高准确率：

### Phase 1：模型正确性

- 实现 `vit_cifar.py`
- 完成 shape test
- 完成 backward test
- 注册到模型工厂

### Phase 2：训练稳定性

- AdamW
- AMP
- warmup + cosine
- gradient clipping
- checkpoint/resume

### Phase 3：ViT 训练增强

- Mixup
- CutMix
- EMA
- DropPath
- label smoothing

### Phase 4：Colab 显存与实验记录

- Colab GPU 型号检测
- T4 16 GB batch 256/384/512 实测
- 本地 RTX 3060 6 GB 仅做 smoke test
- 记录峰值显存
- 5–10 epoch smoke test
- 更新 README 和 AGENTS.md

### Phase 5：正式训练

- 300 epoch 或 early stopping
- 保存 best/last checkpoint
- 记录最终准确率、训练时间、显存和参数量

