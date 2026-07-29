import torch
import os
import argparse
import json
import shutil
import copy
import torch.nn as nn
import time
from pathlib import Path

import yaml

from mobilenetv1 import mobilenetv1_small
from mobilevit import mobilevit_small
from convnext_cifar import convnext_tiny_cifar100
from wide_resnet import wide_resnet_cifar100
from vit_cifar import vit_tiny_cifar100
from load_cifar100 import get_dataloaders


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(model_buffer)


def mixup_data(inputs, targets, alpha=0.2):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    return mixed_inputs, targets, targets[index], lam


def rand_bbox(size, lam):
    _, _, h, w = size
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = torch.randint(0, w, (1,)).item()
    cy = torch.randint(0, h, (1,)).item()

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)
    return x1, y1, x2, y2


def cutmix_data(inputs, targets, alpha=1.0):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(inputs.size(0), device=inputs.device)

    mixed_inputs = inputs.clone()
    x1, y1, x2, y2 = rand_bbox(inputs.size(), lam)
    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (inputs.size(-1) * inputs.size(-2)))
    return mixed_inputs, targets, targets[index], lam


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def build_model(model_name, model_config):
    model_params = model_config.get('params', {})

    if model_name == 'mobilenet':
        return mobilenetv1_small(**model_params)
    if model_name == 'mobilevit':
        return mobilevit_small(**model_params)
    if model_name == 'convnext':
        return convnext_tiny_cifar100(**model_params)
    if model_name == 'wide_resnet':
        return wide_resnet_cifar100(**model_params)
    if model_name == 'vit':
        return vit_tiny_cifar100(**model_params)

    raise ValueError(f"不支持的模型类型: {model_name}")


def save_checkpoint(path, state):
    torch.save(state, str(path))


def load_checkpoint(path, net, optimizer, scheduler, scaler, ema_model=None):
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_state_dict' in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'scaler_state_dict' in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    if 'ema_state_dict' in ckpt and ema_model is not None:
        ema_model.load_state_dict(ckpt['ema_state_dict'])
    return ckpt


def print_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU 训练")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch 版本: {torch.__version__}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU 显存: {total_mem:.1f} GB")


if __name__ == '__main__':
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    parser = argparse.ArgumentParser(description='CIFAR-100 训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='断点续训 checkpoint 路径')
    parser.add_argument('--drive-sync', type=str, default=None, help='Google Drive 同步目录')
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / args.config if not os.path.isabs(args.config) else Path(args.config)
    config = load_config(config_path)

    train_config = config.get('train', {})
    data_config = config.get('data', {})
    models_config = config.get('models', {})
    active_model_name = config.get('active_model', 'mobilenet')
    model_config = models_config.get(active_model_name, {})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print_gpu_info()
    print(f"使用设备: {device}")
    print(f"当前模型: {active_model_name}")
    print(f"模型参数: {model_config.get('params', {})}")
    print("=" * 60)

    # 训练超参数
    epoch_num = train_config.get('epoch_num', 300)
    lr = train_config.get('lr', 0.002)
    warmup_epochs = train_config.get('warmup_epochs', 5)
    early_stop_patience = train_config.get('early_stop_patience', 30)
    early_stop_min_delta = train_config.get('early_stop_min_delta', 0.001)
    mixup_alpha = train_config.get('mixup_alpha', 0.0)
    cutmix_alpha = train_config.get('cutmix_alpha', 1.0)
    cutmix_prob = train_config.get('cutmix_prob', 0.0)
    ema_decay = train_config.get('ema_decay', 0.999)
    use_ema = train_config.get('use_ema', False)
    use_amp = train_config.get('amp', False) and device.type == 'cuda'
    optimizer_name = train_config.get('optimizer', 'adam')
    grad_clip_norm = train_config.get('grad_clip_norm', 0.0)
    accumulation_steps = train_config.get('accumulation_steps', 1)

    # AMP 兼容性处理
    if use_amp:
        try:
            scaler = torch.amp.GradScaler('cuda')
            print("AMP 已启用 (torch.amp.GradScaler)")
        except TypeError:
            scaler = torch.cuda.amp.GradScaler()
            print("AMP 已启用 (torch.cuda.amp.GradScaler)")
    else:
        scaler = None
        if device.type == 'cuda':
            print("AMP 已禁用")

    # 构建模型
    net = build_model(active_model_name, model_config).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # EMA 模型
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(net).to(device)
        ema_model.eval()
        print(f"EMA 已启用, decay={ema_decay}")

    # 数据加载
    train_dataloader, test_dataloader, train_size, test_size = get_dataloaders(
        dataset_root=data_config.get('dataset_root', r"D:\Study\cifar100"),
        train_batch_size=data_config.get('train_batch_size', 256),
        test_batch_size=data_config.get('test_batch_size', 512),
        num_workers=data_config.get('num_workers', 4),
    )
    print(f"训练集数量: {train_size}")
    print(f"测试集数量: {test_size}")

    # 损失函数
    loss_fn = nn.CrossEntropyLoss(label_smoothing=train_config.get('label_smoothing', 0.05))

    # 优化器
    weight_decay = train_config.get('weight_decay', 5e-4)
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999),
        )
    else:
        optimizer = torch.optim.Adam(
            net.parameters(), lr=lr, weight_decay=weight_decay,
        )
    print(f"优化器: {optimizer_name}")

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epoch_num - warmup_epochs))

    # 目录
    base_model_dir = Path(data_config.get('model_dir', 'models'))
    base_log_dir = Path(data_config.get('log_dir', 'logs'))
    model_dir = base_model_dir / active_model_name
    log_dir = base_log_dir / active_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 断点续训
    start_epoch = 0
    best_test_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    global_step = 0

    if args.resume:
        print(f"从 checkpoint 恢复: {args.resume}")
        ckpt = load_checkpoint(args.resume, net, optimizer, scheduler, scaler, ema_model)
        start_epoch = ckpt.get('epoch', 0)
        best_test_accuracy = ckpt.get('best_test_accuracy', 0.0)
        best_epoch = ckpt.get('best_epoch', 0)
        patience_counter = ckpt.get('patience_counter', 0)
        global_step = ckpt.get('global_step', 0)
        print(f"恢复到 epoch {start_epoch}, best_acc={best_test_accuracy:.4f}")

    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(log_dir))
    print(f"模型保存目录: {model_dir}")
    print(f"日志保存目录: {log_dir}")
    print("=" * 60)

    # Mixup/CutMix 是否启用
    use_mixup = mixup_alpha > 0 or cutmix_prob > 0
    if use_mixup:
        print(f"Mixup alpha={mixup_alpha}, CutMix alpha={cutmix_alpha}, prob={cutmix_prob}")

    step = global_step
    train_start_time = time.time()

    for epoch in range(start_epoch, epoch_num):
        epoch_start_time = time.time()
        net.train()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        if epoch < warmup_epochs:
            warmup_factor = float(epoch + 1) / float(max(1, warmup_epochs))
            current_lr = lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        optimizer.zero_grad()

        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Mixup / CutMix
            use_cutmix = use_mixup and torch.rand(1).item() < cutmix_prob and cutmix_alpha > 0
            use_mix = use_mixup and (not use_cutmix) and mixup_alpha > 0

            if use_cutmix:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, cutmix_alpha)
            elif use_mix:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha)
            else:
                targets_a = labels
                targets_b = labels
                lam = 1.0

            # Forward
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = net(inputs)
                    if lam < 1.0:
                        loss = lam * loss_fn(outputs, targets_a) + (1 - lam) * loss_fn(outputs, targets_b)
                    else:
                        loss = loss_fn(outputs, labels)
            else:
                outputs = net(inputs)
                if lam < 1.0:
                    loss = lam * loss_fn(outputs, targets_a) + (1 - lam) * loss_fn(outputs, targets_b)
                else:
                    loss = loss_fn(outputs, labels)

            scaled_loss = loss / accumulation_steps

            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # 梯度累积
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                if use_amp and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)
                elif grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if use_ema:
                    update_ema(net, ema_model, ema_decay)

                optimizer.zero_grad()

            # 统计
            with torch.no_grad():
                predicted = outputs.argmax(dim=1)
                if lam >= 1.0:
                    correct = (predicted == labels).sum().item()
                    total_count = labels.size(0)
                else:
                    correct_a = (predicted == targets_a).sum().item()
                    correct_b = (predicted == targets_b).sum().item()
                    correct = lam * correct_a + (1 - lam) * correct_b
                    total_count = labels.size(0)

            sum_loss += loss.item()
            sum_correct += correct
            sum_total += total_count

            writer.add_scalar('train_loss', loss.item(), global_step=step)
            step += 1

        train_loss = sum_loss / len(train_dataloader)
        train_accuracy = sum_correct / sum_total
        writer.add_scalar('train_accuracy', train_accuracy, global_step=epoch)

        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - train_start_time
        avg_epoch_time = elapsed_time / max(1, epoch - start_epoch + 1)
        remaining_epochs = epoch_num - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs

        # GPU 显存
        gpu_mem_str = ""
        if device.type == 'cuda':
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
            gpu_mem_str = f" | GPU Mem: {peak_alloc:.0f}/{peak_reserved:.0f} MB"

        print(
            f"Epoch [{epoch+1}/{epoch_num}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Epoch: {format_seconds(epoch_time)} | "
            f"ETA: {format_seconds(eta_seconds)}"
            f"{gpu_mem_str}"
        )

        # 周期 checkpoint
        if (epoch + 1) % 20 == 0:
            periodic_path = model_dir / f'cifar100_{epoch+1}.pth'
            periodic_state = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_test_accuracy': best_test_accuracy,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'global_step': step,
            }
            if scaler is not None:
                periodic_state['scaler_state_dict'] = scaler.state_dict()
            if use_ema and ema_model is not None:
                periodic_state['ema_state_dict'] = ema_model.state_dict()
            save_checkpoint(periodic_path, periodic_state)
            print(f"周期 checkpoint 已保存: {periodic_path}")

        # LR scheduler
        if epoch >= warmup_epochs:
            scheduler.step()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)

        # 验证
        eval_model = ema_model if use_ema else net
        eval_model.eval()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0

        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = eval_model(inputs)
                        loss = loss_fn(outputs, labels)
                else:
                    outputs = eval_model(inputs)
                    loss = loss_fn(outputs, labels)

                predicted = outputs.argmax(dim=1)
                correct = (predicted == labels).sum().item()
                sum_loss += loss.item()
                sum_correct += correct
                sum_total += labels.size(0)

        test_loss = sum_loss / len(test_dataloader)
        test_accuracy = sum_correct / sum_total
        writer.add_scalar('test_loss', test_loss, global_step=epoch)
        writer.add_scalar('test_accuracy', test_accuracy, global_step=epoch)

        # 保存最佳
        if test_accuracy > (best_test_accuracy + early_stop_min_delta):
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = model_dir / 'cifar100_best.pth'
            best_state = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_test_accuracy': best_test_accuracy,
                'test_loss': test_loss,
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'global_step': step,
            }
            if scaler is not None:
                best_state['scaler_state_dict'] = scaler.state_dict()
            if use_ema and ema_model is not None:
                best_state['ema_state_dict'] = ema_model.state_dict()
            save_checkpoint(best_path, best_state)
            print(f"         | 新最佳模型已保存: {best_path} (Acc={best_test_accuracy:.4f})")

            # 同步到 Drive
            if args.drive_sync:
                drive_best = Path(args.drive_sync) / 'cifar100_best.pth'
                drive_best.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_path, drive_best)
                print(f"         | 已同步 best 到 Drive: {drive_best}")
        else:
            patience_counter += 1

        # 保存 last
        last_path = model_dir / 'cifar100_last.pth'
        last_state = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_test_accuracy': best_test_accuracy,
            'best_epoch': best_epoch,
            'patience_counter': patience_counter,
            'global_step': step,
        }
        if scaler is not None:
            last_state['scaler_state_dict'] = scaler.state_dict()
        if use_ema and ema_model is not None:
            last_state['ema_state_dict'] = ema_model.state_dict()
        save_checkpoint(last_path, last_state)

        # 同步 last 到 Drive
        if args.drive_sync:
            drive_last = Path(args.drive_sync) / 'cifar100_last.pth'
            drive_last.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(last_path, drive_last)

        writer.add_scalar('best_test_accuracy', best_test_accuracy, global_step=epoch)

        print(f"         | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_accuracy:.4f} | "
              f"Best: {best_test_accuracy:.4f} (Epoch {best_epoch}) | "
              f"EarlyStop: {patience_counter}/{early_stop_patience}\n")

        if patience_counter >= early_stop_patience:
            print(f"Early stopping 触发，连续 {early_stop_patience} 个 epoch 无提升。")
            break

    writer.close()

    # 保存训练摘要
    results_dir = project_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    summary = {
        'model': active_model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_test_accuracy': best_test_accuracy,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1 - start_epoch,
        'optimizer': optimizer_name,
        'lr': lr,
        'use_amp': use_amp,
        'use_ema': use_ema,
        'use_mixup': use_mixup,
        'label_smoothing': train_config.get('label_smoothing', 0.0),
        'weight_decay': weight_decay,
        'grad_clip_norm': grad_clip_norm,
        'accumulation_steps': accumulation_steps,
        'device': str(device),
    }
    if device.type == 'cuda':
        summary['gpu'] = torch.cuda.get_device_name(0)
        summary['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
    summary_path = results_dir / f'{active_model_name}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"训练摘要已保存: {summary_path}")

    # 同步摘要到 Drive
    if args.drive_sync:
        drive_summary = Path(args.drive_sync) / f'{active_model_name}_summary.json'
        shutil.copy2(summary_path, drive_summary)

    print("训练完成！")
