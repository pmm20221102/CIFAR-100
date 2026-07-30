# CIFAR-100 预训练 ViT 本地微调实施计划（RTX 3060 6GB）

## 1. 任务目标

在现有 CIFAR-100 图像分类项目中，新增一条“预训练 Vision Transformer 微调”实验路线。

本任务与当前“从零训练 ViT”实验分开，目标是：

1. 使用公开预训练权重初始化 ViT；
2. 在 CIFAR-100 的 50,000 张训练图像上进行迁移学习；
3. 在 RTX 3060 6GB 本地显卡上稳定完成训练；
4. 对比以下实验：
   - 现有 CNN 模型；
   - MobileViT；
   - 从零训练的小型 ViT；
   - 预训练 ViT 线性探测；
   - 预训练 ViT 部分解冻；
   - 预训练 ViT 全量微调；
5. 记录准确率、训练时间、显存、参数量和收敛速度。

本任务优先复用现有项目的配置管理、日志、checkpoint、early stopping 和 CIFAR-100 数据加载逻辑，不破坏已有模型及实验结果。

---

## 2. 当前项目情况

现有关键文件：

```text
config.yaml
train.py
train_convnext.py
load_cifar100.py
mobilenetv1.py
mobilevit.py
convnext_cifar.py
wide_resnet.py
README.md
```

现有训练能力包括：

- CIFAR-100 本地数据集；
- RandomCrop；
- RandomHorizontalFlip；
- RandAugment；
- RandomErasing；
- Label Smoothing；
- Warmup；
- CosineAnnealing；
- Early Stopping；
- TensorBoard；
- checkpoint；
- 部分训练脚本包含 AMP、Mixup、CutMix、EMA 和 Gradient Clipping。

当前本地数据路径：

```yaml
dataset_root: D:\Study\cifar100
```

本地硬件：

```text
GPU: NVIDIA RTX 3060
VRAM: 6GB
OS: Windows
```

设计时必须优先保证 6GB 显存下稳定运行。

---

## 3. 模型来源建议

优先使用 `timm`，不要为了本任务把现有项目整体迁移到 Hugging Face Trainer。

推荐依赖：

```bash
pip install timm
```

优先模型：

```text
deit_tiny_patch16_224
vit_tiny_patch16_224
```

第一选择：

```python
timm.create_model(
    "deit_tiny_patch16_224",
    pretrained=True,
    num_classes=100,
)
```

原因：

- 模型较小；
- 适合 6GB 显存；
- DeiT 比原始 ViT 更强调数据效率；
- 容易和现有 PyTorch training loop 集成；
- 可直接替换为 CIFAR-100 的 100 类分类头。

不要优先使用：

```text
vit_base_patch16_224
vit_large_patch16_224
```

它们对 6GB 显存和本地训练速度都不友好。

可选第二实验：

```python
timm.create_model(
    "vit_tiny_patch16_224",
    pretrained=True,
    num_classes=100,
)
```

---

## 4. 新增文件

建议新增：

```text
pretrained_vit.py
train_pretrained_vit.py
config_pretrained_vit_local.yaml
load_cifar100_pretrained_vit.py
evaluate_pretrained_vit.py
README_pretrained_vit.md
```

也可以在现有统一框架中注册模型，但必须保证：

- 不破坏 `train.py`；
- 不改变已有模型默认训练结果；
- 预训练 ViT 使用独立配置；
- checkpoint 与日志存储到独立目录。

建议目录：

```text
models/pretrained_vit/
logs/pretrained_vit/
results/pretrained_vit/
```

---

## 5. 输入尺寸处理

公开 ImageNet 预训练 ViT 通常使用：

```text
224 × 224
```

而 CIFAR-100 原图为：

```text
32 × 32
```

本阶段优先采用：

```text
32 × 32 → Resize 到 224 × 224
```

这样可以直接兼容：

- patch embedding；
- positional embedding；
- 预训练权重；
- 模型默认配置。

不要在第一版中把 patch size 从 16 改为 4，因为这会导致 patch embedding 权重不兼容，增加额外插值或重初始化问题。

---

## 6. 数据预处理

预训练模型必须使用其对应的 ImageNet 预处理参数，不应继续直接使用 CIFAR-100 mean/std。

通过 `timm.data.resolve_model_data_config()` 和 `create_transform()` 获取模型对应配置，或明确使用 ImageNet normalization。

推荐训练增强：

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.20),
        ratio=(0.3, 3.3),
        value="random",
    ),
])
```

推荐测试增强：

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])
```

也可以直接使用 `timm.data.create_transform()`，但必须确认：

- train 和 test transform 分离；
- test 不使用随机增强；
- 输入最终为 224×224；
- normalization 与预训练权重一致。

---

## 7. 训练阶段设计

必须实现三种微调模式。

### 7.1 阶段 A：Linear Probing

冻结整个 backbone，只训练分类头。

目标：

- 验证预训练特征对 CIFAR-100 的可迁移性；
- 以最低显存和最低成本得到 baseline。

伪代码：

```python
for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in model.get_classifier().parameters():
    parameter.requires_grad = True
```

注意不同 timm 模型分类头访问方式可能不同，应优先使用：

```python
model.get_classifier()
```

或者检查模型结构后处理。

推荐配置：

```yaml
mode: linear_probe
epochs: 20
batch_size: 64
learning_rate: 0.001
weight_decay: 0.0001
optimizer: adamw
warmup_epochs: 2
early_stop_patience: 8
label_smoothing: 0.0
amp: true
```

只训练分类头时，学习率可以高于全量微调。

---

### 7.2 阶段 B：部分解冻

解冻分类头和最后 2 个 Transformer blocks。

目标：

- 在显存可控的情况下适应 CIFAR-100；
- 比 linear probing 获得更高上限；
- 降低全量微调造成 catastrophic forgetting 的风险。

推荐策略：

```text
冻结 patch embedding
冻结前面大部分 transformer blocks
解冻最后 2 blocks
解冻 final norm
解冻 classifier
```

Agent 必须根据 timm 模型实际模块名检查结构，例如：

```python
print(model)
```

常见结构可能包括：

```text
model.blocks
model.norm
model.head
```

推荐配置：

```yaml
mode: partial_finetune
unfreeze_last_blocks: 2
epochs: 40
batch_size: 32
learning_rate: 0.00005
head_learning_rate: 0.0002
weight_decay: 0.05
optimizer: adamw
warmup_epochs: 3
early_stop_patience: 12
label_smoothing: 0.1
amp: true
gradient_clip_norm: 1.0
```

建议使用参数组，为 head 设置更高学习率：

```python
optimizer = AdamW([
    {
        "params": backbone_parameters,
        "lr": 5e-5,
    },
    {
        "params": head_parameters,
        "lr": 2e-4,
    },
], weight_decay=0.05)
```

---

### 7.3 阶段 C：全量微调

解冻全部参数。

目标：

- 测试预训练 ViT 在 CIFAR-100 上的最高可达到性能；
- 与部分解冻结果对比。

推荐配置：

```yaml
mode: full_finetune
epochs: 50
batch_size: 16
gradient_accumulation_steps: 2
effective_batch_size: 32
learning_rate: 0.00002
head_learning_rate: 0.0001
weight_decay: 0.05
optimizer: adamw
warmup_epochs: 5
early_stop_patience: 15
label_smoothing: 0.1
amp: true
gradient_clip_norm: 1.0
```

全量微调必须采用小学习率，不能沿用从零训练时的 `0.002`。

推荐范围：

```text
backbone lr: 1e-5 ～ 5e-5
head lr: 5e-5 ～ 2e-4
```

---

## 8. RTX 3060 6GB 显存策略

必须启用 AMP：

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
    enabled=True,
):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

使用 `GradScaler`：

```python
scaler = torch.amp.GradScaler("cuda", enabled=True)
```

推荐从以下 batch size 开始：

```text
Linear probing: 64
部分解冻: 32
全量微调: 16
```

若 OOM：

1. 将 batch size 减半；
2. 增加 gradient accumulation；
3. 减小 test batch size；
4. 设置 `optimizer.zero_grad(set_to_none=True)`；
5. 避免在循环中保存不必要的 tensor；
6. 验证阶段使用 `torch.inference_mode()`；
7. 不要同时启用过多高显存增强操作；
8. 不要默认保存所有 epoch checkpoint。

目标显存：

```text
峰值不超过约 5.5GB
```

保留至少数百 MB 余量，避免 Windows 和 CUDA 显存碎片导致训练中途 OOM。

必须打印：

```text
GPU 名称
总显存
allocated memory
reserved memory
max memory allocated
```

日志格式可参考：

```text
Epoch [5/50] |
Train Loss: ... |
Train Acc: ... |
Test Loss: ... |
Test Acc: ... |
Best: ... |
GPU Mem: 5120/6144 MB |
Epoch: 00:03:20 |
ETA: ...
```

---

## 9. 优化器和学习率调度

必须使用：

```python
torch.optim.AdamW
```

不要对预训练 ViT 使用当前通用脚本中的普通 Adam 配置。

推荐：

```python
optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=base_lr,
    weight_decay=0.05,
)
```

Scheduler：

```text
Linear Warmup + Cosine Annealing
```

可以继续复用项目已有 warmup 与 cosine 逻辑，但必须修正：

- warmup 后 scheduler 的起始点；
- resume 后 scheduler 状态；
- 每个 epoch 日志记录真实 lr；
- 参数组学习率分别记录。

---

## 10. Loss 与增强策略

基础 loss：

```python
nn.CrossEntropyLoss(label_smoothing=0.1)
```

第一版建议：

- Linear probing：不使用 Mixup/CutMix；
- 部分解冻：先不使用 Mixup/CutMix；
- 全量微调：先完成无 Mixup/CutMix baseline，再添加增强实验。

原因：

- 微调阶段目标是先验证预训练权重有效；
- 过早叠加 Mixup、CutMix、EMA 会让结果归因困难；
- CIFAR-100 图片被放大到 224 后，过强增强可能破坏有限细节。

第二轮可选实验：

```yaml
mixup_alpha: 0.2
cutmix_alpha: 1.0
cutmix_prob: 0.5
```

但必须单独记录，不能与基础微调结果混淆。

---

## 11. 训练准确率解释

若使用：

- RandAugment；
- RandomErasing；
- Mixup；
- CutMix；
- Label Smoothing；
- Dropout；

训练准确率可能低于测试准确率。

因此建议增加 clean train evaluation：

```text
每 5 个 epoch：
使用无随机增强的训练集评估一次 clean_train_acc
```

最终记录：

```text
augmented train accuracy
clean train accuracy
test accuracy
```

---

## 12. Checkpoint 与断点续训

必须保存：

```text
best.pth
last.pth
```

`best.pth` 至少包括：

```python
{
    "epoch": epoch,
    "model_name": model_name,
    "mode": mode,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "best_test_accuracy": best_test_accuracy,
    "config": config,
}
```

支持：

```bash
python train_pretrained_vit.py --resume models/pretrained_vit/last.pth
```

恢复时必须恢复：

- epoch；
- model；
- optimizer；
- scheduler；
- AMP scaler；
- best accuracy；
- patience counter；
- 随机种子状态可选。

不要只保存模型权重后声称支持完整 resume。

---

## 13. 配置文件建议

新增：

```yaml
experiment_name: deit_tiny_cifar100_local

device:
  amp: true
  compile: false

data:
  dataset_root: D:\Study\cifar100
  image_size: 224
  train_batch_size: 32
  test_batch_size: 64
  num_workers: 4
  pin_memory: true

model:
  library: timm
  name: deit_tiny_patch16_224
  pretrained: true
  num_classes: 100
  mode: partial_finetune
  unfreeze_last_blocks: 2

train:
  epochs: 40
  optimizer: adamw
  backbone_lr: 0.00005
  head_lr: 0.0002
  weight_decay: 0.05
  warmup_epochs: 3
  label_smoothing: 0.1
  gradient_clip_norm: 1.0
  gradient_accumulation_steps: 1
  early_stop_patience: 12
  early_stop_min_delta: 0.001
  seed: 42

output:
  model_dir: models/pretrained_vit
  log_dir: logs/pretrained_vit
  result_dir: results/pretrained_vit
  save_every_epochs: 10
```

默认先执行：

```yaml
mode: partial_finetune
```

因为它在 6GB 显存、训练成本和准确率之间最平衡。

---

## 14. 必须完成的验证步骤

在正式训练前，依次完成：

### 14.1 模型加载测试

确认：

```text
成功下载预训练权重
分类头输出维度为 100
输入 shape 为 [B, 3, 224, 224]
输出 shape 为 [B, 100]
```

### 14.2 冻结状态检查

打印：

```text
总参数量
可训练参数量
冻结参数量
可训练参数比例
```

分别验证：

```text
linear_probe
partial_finetune
full_finetune
```

### 14.3 单 batch forward

确认无 shape error 和 OOM。

### 14.4 单 batch backward

确认只有预期参数产生 gradient。

### 14.5 小数据过拟合测试

从训练集中固定抽取 128～256 张图片，关闭大多数随机增强，训练到接近 100% train accuracy。

如果无法过拟合小数据，优先检查：

- 标签；
- normalization；
- 冻结逻辑；
- classifier；
- loss；
- optimizer；
- learning rate。

### 14.6 3～5 epoch smoke test

确认：

- loss 下降；
- accuracy 上升；
- checkpoint 正常；
- resume 正常；
- TensorBoard 正常；
- 显存稳定；
- 没有显存持续增长。

---

## 15. 评测指标

必须至少记录：

```text
Top-1 Accuracy
Top-5 Accuracy
Test Loss
Best Epoch
Parameter Count
Trainable Parameter Count
Peak GPU Memory
Average Epoch Time
Images per Second
Total Training Time
```

建议输出结果表：

| Model | Init | Mode | Input | Params | Trainable | Best Top-1 | Top-5 | Peak VRAM | Epoch Time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Custom ViT | Random | Full | 32 | ... | ... | ... | ... | ... | ... |
| DeiT-Tiny | ImageNet | Linear Probe | 224 | ... | ... | ... | ... | ... | ... |
| DeiT-Tiny | ImageNet | Last 2 Blocks | 224 | ... | ... | ... | ... | ... | ... |
| DeiT-Tiny | ImageNet | Full | 224 | ... | ... | ... | ... | ... | ... |

---

## 16. 与现有项目结果对比

现有 README 中已有：

```text
MobileNetV1: 73.09%
MobileViT: 70.99%
WideResNet-16-6: 74.10%
ConvNeXt-Tiny: 76.26%
```

新增实验后，README 中增加：

```text
Custom ViT from scratch
DeiT-Tiny linear probing
DeiT-Tiny partial fine-tuning
DeiT-Tiny full fine-tuning
```

必须注明：

- 是否使用预训练权重；
- 输入分辨率；
- 是否使用 Mixup/CutMix；
- 是否使用 AMP；
- 训练设备；
- 模型参数量；
- 可训练参数量。

不能只放准确率，不说明实验条件。

---

## 17. 推荐执行顺序

严格按以下顺序运行：

```text
1. DeiT-Tiny linear probing
2. DeiT-Tiny 解冻最后 2 blocks
3. DeiT-Tiny 全量微调
4. ViT-Tiny 预训练模型作为第二模型对比
5. 可选：加入 Mixup/CutMix
```

第一版不要同时测试过多模型，先确保 DeiT-Tiny 的完整流程稳定。

---

## 18. 不要做的事情

不要：

- 直接用 ViT-Base 在 6GB 显存上硬跑；
- 继续沿用 CIFAR-100 normalization；
- 用 32×32 输入直接加载 patch16-224 权重却不验证兼容性；
- 对全量预训练模型使用 `lr=0.002`；
- 同时修改数据、模型、增强和优化器后只报告一个结果；
- 覆盖已有 CNN checkpoint；
- 只保存最后一轮模型；
- 将 test set 用作超参数搜索训练数据；
- 在测试阶段开启随机增强；
- 忽略可训练参数检查；
- 把“加载预训练权重”误称为“从头预训练”。

---

## 19. 完成标准

任务完成必须满足：

- [ ] 可加载 `deit_tiny_patch16_224` 预训练权重；
- [ ] 输出类别数正确为 100；
- [ ] 具备 linear probing；
- [ ] 具备部分解冻；
- [ ] 具备全量微调；
- [ ] RTX 3060 6GB 上可稳定训练；
- [ ] AMP 正常；
- [ ] gradient accumulation 可配置；
- [ ] 支持完整 resume；
- [ ] 保存 best 和 last；
- [ ] 输出 Top-1、Top-5；
- [ ] 输出峰值显存和 epoch 时间；
- [ ] 与现有模型结果形成统一表格；
- [ ] 更新 README；
- [ ] 不破坏原有 CNN、MobileViT 和从零 ViT 实验。

---

## 20. 最终交付文件

Agent 完成后应提交：

```text
pretrained_vit.py
train_pretrained_vit.py
load_cifar100_pretrained_vit.py
evaluate_pretrained_vit.py
config_pretrained_vit_local.yaml
README_pretrained_vit.md
```

并提供：

```text
运行命令
依赖安装命令
模型结构摘要
冻结参数统计
smoke test 日志
正式训练日志
最佳 checkpoint 路径
最终指标表
已知限制
```

推荐运行命令：

```bash
python train_pretrained_vit.py \
  --config config_pretrained_vit_local.yaml
```

评测命令：

```bash
python evaluate_pretrained_vit.py \
  --config config_pretrained_vit_local.yaml \
  --checkpoint models/pretrained_vit/best.pth
```
