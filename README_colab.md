# ConvNeXt CIFAR-100 — Colab T4 训练指南

## 前置准备

- Google 账号（Colab 免费 T4 GPU）
- 本地已有的 cifar100 项目文件

---

## 步骤一：本地打包项目

在 PowerShell 中执行：

```powershell
cd d:\Study\cifar100
tar -a -c -f cifar100.zip convnext_cifar.py train_convnext.py load_cifar100.py mobilenetv1.py mobilevit.py wide_resnet.py config.yaml config_colab.yaml
```

> 只打包代码和配置，不包含数据集（cifar-100-python/）和模型（models/、logs/），zip 大小约几十 KB。

---

## 步骤二：上传 zip 到 Google Drive

1. 浏览器打开 [drive.google.com](https://drive.google.com)
2. 将 `cifar100.zip` 拖入 Drive 根目录
3. 等待上传完成

---

## 步骤三：VSCode 中运行 notebook

用 VSCode Colab 插件打开 `train_colab.ipynb`，按顺序运行每个 cell：

### Cell 1 — 安装依赖
```
pip install pyyaml tensorboard
```

### Cell 2 — 挂载 Drive + 解压项目
- 首次运行会弹出 Google 授权提示，点击授权即可
- 自动从 Drive 读取 `cifar100.zip` 并解压到 `/content/cifar100_project`
- 会打印解压后的目录结构，确认文件是否完整

### Cell 3 — 下载 CIFAR-100 数据集
- 自动下载到 `/content/cifar100`（约 170MB）

### Cell 4 — 确认 GPU
- 显示 GPU 型号和显存大小
- 确认是 Tesla T4 (16GB)

### Cell 5 — 开始训练
- 自动查找项目目录并启动训练
- 每 10 个 epoch 保存 checkpoint 并同步到 `Google Drive/cifar100_checkpoints/`
- best 模型也会实时同步
- 训练参数见 `config_colab.yaml`

### Cell 6 — 确认同步状态
- 列出 Drive 中已同步的所有 checkpoint
- 显示断点续训命令

---

## 断点续训（Colab 被断开后）

Colab 免费版约 4-12 小时会断开连接。断开后：

1. 重新打开 notebook，运行 Cell 1-4
2. 修改 Cell 5 的命令，加上 `--resume` 参数：

```python
!python train_convnext.py --config config_colab.yaml \
    --resume /content/drive/MyDrive/cifar100_checkpoints/cifar100_best.pth \
    --drive-sync /content/drive/MyDrive/cifar100_checkpoints
```

训练会从断点 epoch 继续，不会丢失进度。

---

## config_colab.yaml 参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| `train_batch_size` | 1400 | T4 显存 16GB，约用 15GB |
| `test_batch_size` | 4096 | 评估无梯度，可以开很大 |
| `lr` | 0.005 | 配合大 batch 线性缩放 |
| `epoch_num` | 300 | 最大训练轮数 |
| `warmup_epochs` | 10 | 学习率预热 |
| `early_stop_patience` | 40 | 连续 40 轮无提升则停止 |
| `amp` | true | FP16 混合精度，加速 + 省显存 |
| `dims` | [48,96,192,384] | 比本地版大 4x，更强的模型 |
| `depths` | [3,3,9,3] | 28 个 block |
| `drop_path_rate` | 0.2 | 随机深度正则 |

---

## 训练完成后

最佳模型保存在：
- **Colab 本地**：`models/convnext/cifar100_best.pth`
- **Google Drive**：`cifar100_checkpoints/cifar100_best.pth`

下载模型：从 Drive 直接下载 `cifar100_best.pth` 即可。

推理时加载 EMA 模型：
```python
import torch
from convnext_cifar import convnext_tiny_cifar100

model = convnext_tiny_cifar100(num_classes=100, dims=[48,96,192,384], depths=[3,3,9,3])
ckpt = torch.load('cifar100_best.pth')
model.load_state_dict(ckpt['ema_state_dict'])  # 用 ema_state_dict，不是 model_state_dict
model.eval()
```
