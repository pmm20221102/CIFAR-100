# GPU 显存统计机制修改任务

## 目标

修改现有训练脚本中的 GPU 显存统计与日志显示方式。

当前类似：

```text
GPU Mem: 89/1572 MB (max 1220)
```

这种格式容易把 `reserved` 误解为显卡总显存，也没有区分 PyTorch 内部显存和整张 GPU 的实际使用情况。

要求在**不大改训练框架**的前提下，统一显存统计函数和日志格式。

---

## 1. 必须区分的显存指标

使用以下 PyTorch API：

```python
torch.cuda.memory_allocated(device)
torch.cuda.memory_reserved(device)
torch.cuda.max_memory_allocated(device)
torch.cuda.max_memory_reserved(device)
```

含义：

- `allocated`：当前真正被 Tensor 占用的显存
- `reserved`：PyTorch CUDA allocator 当前保留的显存
- `peak_allocated`：当前统计周期内 Tensor 实际占用峰值
- `peak_reserved`：当前统计周期内 PyTorch 保留显存峰值

不要把 `reserved` 写成总显存。

整张 GPU 的显存信息使用：

```python
free_bytes, total_bytes = torch.cuda.mem_get_info(device)
device_used_bytes = total_bytes - free_bytes
```

含义：

- `device_total`：GPU 总显存
- `device_free`：GPU 当前空闲显存
- `device_used`：整个设备当前已使用显存

`device_used` 会包含：

- 当前训练进程
- CUDA Context
- Windows 图形界面占用
- 其他 GPU 程序
- 驱动和缓存开销

---

## 2. 新增统一统计函数

建议增加：

```python
def bytes_to_mb(value: int) -> float:
    return value / (1024 ** 2)


def get_gpu_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}

    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    free_memory, total_memory = torch.cuda.mem_get_info(device)
    device_used = total_memory - free_memory

    return {
        "allocated_mb": bytes_to_mb(allocated),
        "reserved_mb": bytes_to_mb(reserved),
        "peak_allocated_mb": bytes_to_mb(peak_allocated),
        "peak_reserved_mb": bytes_to_mb(peak_reserved),
        "device_used_mb": bytes_to_mb(device_used),
        "device_free_mb": bytes_to_mb(free_memory),
        "device_total_mb": bytes_to_mb(total_memory),
    }
```

---

## 3. 修改日志格式

将：

```text
GPU Mem: 89/1572 MB (max 1220)
```

改为更明确的格式：

```text
GPU | alloc 89 MB | reserved 1572 MB | peak_alloc 1220 MB | device 2140/6144 MB
```

如果日志允许更详细，可显示：

```text
Torch Alloc: 89 MB
Torch Reserved: 1572 MB
Peak Allocated: 1220 MB
Peak Reserved: 1572 MB
Device Used: 2140/6144 MB
```

其中：

```text
device used / device total
```

才是最接近 `nvidia-smi` 的整张显卡使用情况。

---

## 4. 每个 Epoch 重置峰值

如果日志希望显示“当前 epoch 的峰值”，必须在每个 epoch 开始时执行：

```python
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
```

示例：

```python
for epoch in range(start_epoch, epochs):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_one_epoch(...)
    evaluate(...)

    gpu_stats = get_gpu_memory_stats(device)
```

否则 `max_memory_allocated()` 会一直记录程序启动后的历史最高值。

---

## 5. 读取显存前同步 CUDA

CUDA 运算是异步的。

在以下位置读取时间和显存前执行：

```python
torch.cuda.synchronize(device)
```

建议位置：

- epoch 结束
- validation 结束
- 统计 epoch duration 前
- 统计显存峰值前

避免读取到尚未完成的 CUDA 操作状态。

---

## 6. 统一日志格式函数

建议增加：

```python
def format_gpu_memory(stats: dict[str, float]) -> str:
    if not stats:
        return "GPU: N/A"

    return (
        f"GPU | alloc {stats['allocated_mb']:.0f} MB"
        f" | reserved {stats['reserved_mb']:.0f} MB"
        f" | peak_alloc {stats['peak_allocated_mb']:.0f} MB"
        f" | peak_reserved {stats['peak_reserved_mb']:.0f} MB"
        f" | device {stats['device_used_mb']:.0f}/"
        f"{stats['device_total_mb']:.0f} MB"
    )
```

训练日志统一调用此函数，不要在不同脚本中重复写不同格式。

---

## 7. 记录全程峰值

每个 epoch 的 peak stats 会被重置，因此还需要单独维护整个训练过程的最大值：

```python
global_peak_allocated_mb = 0.0
global_peak_reserved_mb = 0.0
global_peak_device_used_mb = 0.0
```

每个 epoch 结束后更新：

```python
global_peak_allocated_mb = max(
    global_peak_allocated_mb,
    gpu_stats["peak_allocated_mb"],
)

global_peak_reserved_mb = max(
    global_peak_reserved_mb,
    gpu_stats["peak_reserved_mb"],
)

global_peak_device_used_mb = max(
    global_peak_device_used_mb,
    gpu_stats["device_used_mb"],
)
```

注意：

- `device_used_mb` 是某一读取时刻的设备使用量
- 它不一定能捕捉到 epoch 中间的瞬时设备峰值
- 真正可靠的训练进程峰值仍以 `peak_allocated` 为主

---

## 8. Summary JSON 增加字段

最终 summary 中增加：

```json
{
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "gpu_total_memory_mb": 6144,
  "peak_torch_allocated_mb": 1220,
  "peak_torch_reserved_mb": 1572,
  "peak_device_used_mb": 2140
}
```

GPU 名称可使用：

```python
torch.cuda.get_device_name(device)
```

总显存使用：

```python
torch.cuda.get_device_properties(device).total_memory
```

或者复用 `torch.cuda.mem_get_info()` 返回的 total。

---

## 9. 不要频繁调用 empty_cache

不要为了让 `reserved` 数值变小而在每个 batch 或 epoch 调用：

```python
torch.cuda.empty_cache()
```

原因：

- 不会释放仍被 Tensor 使用的显存
- 可能降低训练速度
- 会导致 CUDA allocator 反复申请显存
- 可能增加显存碎片和性能波动

只允许在以下场景使用：

- 完整训练阶段结束
- 删除一个模型并准备加载另一个模型
- 明确删除大型临时对象后
- OOM 恢复流程

正常训练中不要调用。

---

## 10. 可选调试模式

可增加配置：

```yaml
debug:
  log_gpu_every_n_batches: 0
```

规则：

- `0`：关闭 batch 级显存日志
- 大于 0：每隔 N 个 batch 打印一次

仅在排查 OOM 或显存泄漏时启用，默认关闭。

---

## 11. 验证要求

修改完成后必须验证：

- [ ] CUDA 设备下能正常输出所有显存字段
- [ ] CPU 模式下不会报错
- [ ] `reserved` 不再被误写成总显存
- [ ] 每个 epoch 的峰值会重置
- [ ] summary 保存全程峰值
- [ ] 日志中的总显存接近 RTX 3060 的实际值
- [ ] 与 `nvidia-smi` 的整体使用趋势一致
- [ ] 不影响训练结果、checkpoint 和 resume
- [ ] 不在正常训练中频繁调用 `empty_cache`
- [ ] Tiny、Small 和三种微调模式共用同一统计函数

---

## 12. 最终建议日志示例

```text
Epoch [10/50] |
Train Loss: 1.3382 |
Train Acc: 0.8155 |
Test Loss: 0.7070 |
Test Top1: 0.8128 |
Top5: 0.9729 |
Best: 0.8128 |
GPU | alloc 89 MB | reserved 1572 MB | peak_alloc 1220 MB | peak_reserved 1572 MB | device 2140/6144 MB |
Epoch: 00:02:03 |
ETA: 01:01:47
```

核心原则：

```text
allocated / reserved = PyTorch allocator 指标
device used / total = 整张 GPU 的实际使用情况
peak allocated = 最适合比较不同模型训练显存成本的指标
```
