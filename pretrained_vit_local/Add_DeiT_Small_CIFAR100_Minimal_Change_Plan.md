# 增加 DeiT-Small 微调实验（最小改动任务）

## 目标

在现有 `pretrained_vit_local` 代码基础上，新增 `deit_small_patch16_224` 支持。

要求：

- 不重构现有训练框架；
- 不影响现有 DeiT-Tiny 三组实验；
- 继续保留三种并列微调模式：`linear_probe`、`partial_finetune`、`full_finetune`；
- 三种模式均从同一份原始 ImageNet 预训练权重独立开始；
- 适配 RTX 3060 6GB 本地训练。

## 1. 模型选择

新增模型配置：

```yaml
model:
  name: deit_small_patch16_224
  pretrained: true
  num_classes: 100
```

继续使用现有 `timm.create_model()` 逻辑：

```python
model = timm.create_model(
    config["model"]["name"],
    pretrained=config["model"]["pretrained"],
    num_classes=100,
)
```

不要复制一套 Small 模型代码。Tiny 和 Small 必须共用同一训练脚本，只通过配置切换模型。

## 2. 三种微调模式

### Linear Probe

- 冻结整个 backbone；
- 只训练分类头。

### Partial Fine-tuning

- 冻结 patch embedding 和前面 Transformer blocks；
- 解冻最后 2 个 blocks；
- 解冻 final norm 和分类头。

继续使用：

```yaml
mode: partial_finetune
unfreeze_last_blocks: 2
```

冻结逻辑不得写死层数，应通过 `len(model.blocks)` 动态处理。

### Full Fine-tuning

- 所有参数参与训练；
- 从原始 DeiT-Small 预训练权重开始；
- 不加载 Tiny checkpoint，也不加载前一个微调模式的 checkpoint。

## 3. 新增配置文件

仅新增：

```text
config_small_linear_probe.yaml
config_small_partial_finetune.yaml
config_small_full_finetune.yaml
```

可复制现有 Tiny 配置，只修改：

- `experiment_name`
- `model.name`
- batch size
- gradient accumulation
- 输出目录

## 4. RTX 3060 6GB 推荐配置

### Linear Probe

```yaml
train:
  epochs: 20
  batch_size: 32
  gradient_accumulation_steps: 1
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_epochs: 2
  amp: true
```

### Partial Fine-tuning

```yaml
train:
  epochs: 40
  batch_size: 16
  gradient_accumulation_steps: 2
  backbone_lr: 0.00003
  head_lr: 0.00015
  weight_decay: 0.05
  warmup_epochs: 3
  amp: true
  gradient_clip_norm: 1.0
```

### Full Fine-tuning

```yaml
train:
  epochs: 50
  batch_size: 8
  gradient_accumulation_steps: 4
  backbone_lr: 0.00001
  head_lr: 0.00005
  weight_decay: 0.05
  warmup_epochs: 5
  amp: true
  gradient_clip_norm: 1.0
```

如果显存充足，可逐步增加 batch size；如果 OOM，优先减小 batch 并增加梯度累积，保持 effective batch size 接近 32。

## 5. 输出隔离

Small 实验不得覆盖 Tiny 结果：

```text
logs/pretrained_vit_small_linear/
logs/pretrained_vit_small_partial/
logs/pretrained_vit_small_full/

models/pretrained_vit_small_linear/
models/pretrained_vit_small_partial/
models/pretrained_vit_small_full/

results/pretrained_vit_small_linear/
results/pretrained_vit_small_partial/
results/pretrained_vit_small_full/
```

## 6. 正式训练前检查

每种模式先运行一个 batch，并输出：

```text
模型名称
总参数量
可训练参数量
冻结参数量
可训练比例
输入 shape
输出 shape
峰值显存
```

确认：

- 输出为 `[B, 100]`；
- AMP 正常；
- 冻结参数符合预期；
- forward、backward、optimizer step 正常；
- checkpoint 和 resume 正常；
- 无 OOM。

## 7. 结果对比

在现有结果表中增加：

| Model | Mode | Params | Trainable Params | Best Top-1 | Top-5 | Peak VRAM | Training Time |
|---|---|---:|---:|---:|---:|---:|---:|
| DeiT-Tiny | Partial FT | 5.54M | 0.91M | 75.84% | ... | ... | ... |
| DeiT-Tiny | Full FT | 5.54M | 5.54M | 84.96% | ... | ... | ... |
| DeiT-Small | Linear Probe | ... | ... | ... | ... | ... | ... |
| DeiT-Small | Partial FT | ... | ... | ... | ... | ... | ... |
| DeiT-Small | Full FT | ... | ... | ... | ... | ... | ... |

重点比较：

- Small 相比 Tiny 的准确率提升；
- 参数量、显存和训练时间增加；
- 三种微调模式的差异；
- 精度收益是否值得额外资源成本。

## 8. 完成标准

- [ ] 现有 Tiny 三种模式仍可正常运行；
- [ ] Small 通过配置加载，不复制训练脚本；
- [ ] Small 支持三种独立微调模式；
- [ ] partial 模式动态解冻最后 2 个 blocks；
- [ ] RTX 3060 6GB 上能够稳定训练；
- [ ] checkpoint、resume、TensorBoard、summary 正常；
- [ ] Small 与 Tiny 输出目录完全隔离；
- [ ] README 增加 Tiny/Small 对比；
- [ ] 不进行无关重构。
