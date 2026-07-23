import torch
import os
import torch.nn as nn
import time
from pathlib import Path

import yaml

from mobilenetv1 import mobilenetv1_small
from mobilevit import mobilevit_small
from convnext_cifar import convnext_tiny_cifar100
from wide_resnet import wide_resnet_cifar100
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

    raise ValueError(f"不支持的模型类型: {model_name}")

if __name__ == '__main__':
    # 抑制 TensorBoard 依赖带来的 tensorflow INFO 级别日志。
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from torch.utils.tensorboard import SummaryWriter

    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / 'config.yaml'
    config = load_config(config_path)

    train_config = config.get('train', {})
    data_config = config.get('data', {})
    models_config = config.get('models', {})
    active_model_name = config.get('active_model', 'mobilenet')
    model_config = models_config.get(active_model_name, {})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"使用设备: {device}")
    
    epoch_num = train_config.get('epoch_num', 300)
    lr = train_config.get('lr', 0.002)
    warmup_epochs = train_config.get('warmup_epochs', 5)
    early_stop_patience = train_config.get('early_stop_patience', 30)
    early_stop_min_delta = train_config.get('early_stop_min_delta', 0.001)
    mixup_alpha = train_config.get('mixup_alpha', 0.2)
    cutmix_alpha = train_config.get('cutmix_alpha', 1.0)
    cutmix_prob = train_config.get('cutmix_prob', 0.0)
    ema_decay = train_config.get('ema_decay', 0.999)
    
    net = build_model(active_model_name, model_config).to(device)
    print(f"模型已加载到 {device}")
    print(f"当前模型: {active_model_name}")
    print(f"模型参数: {model_config.get('params', {})}")

    train_dataloader, test_dataloader, train_size, test_size = get_dataloaders(
        dataset_root=data_config.get('dataset_root', r"D:\Study\cifar100"),
        train_batch_size=data_config.get('train_batch_size', 256),
        test_batch_size=data_config.get('test_batch_size', 512),
        num_workers=data_config.get('num_workers', 4),
    )
    print(f"训练集数量: {train_size}")
    print(f"测试集数量: {test_size}")
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=train_config.get('label_smoothing', 0.05))

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=lr,
        weight_decay=train_config.get('weight_decay', 5e-4),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epoch_num - warmup_epochs))

    base_model_dir = Path(data_config.get('model_dir', 'models'))
    base_log_dir = Path(data_config.get('log_dir', 'logs'))
    model_dir = base_model_dir / active_model_name
    log_dir = base_log_dir / active_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))
    print(f"模型保存目录: {model_dir}")
    print(f"日志保存目录: {log_dir}")
    step = 0
    train_start_time = time.time()
    best_test_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epoch_num):
        epoch_start_time = time.time()
        net.train()
        sum_loss = 0
        sum_accuracy = 0

        if epoch < warmup_epochs:
            warmup_factor = float(epoch + 1) / float(max(1, warmup_epochs))
            current_lr = lr * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predicted = outputs.argmax(dim=1)
            correct = (predicted == labels).sum().item()
            sum_loss += loss.item()
            sum_accuracy += correct

            writer.add_scalar('train_loss', loss.item(), global_step=step)
            step += 1
        
        train_loss = sum_loss / len(train_dataloader)
        train_accuracy = sum_accuracy / len(train_dataloader.dataset)
        writer.add_scalar('train_accuracy', train_accuracy, global_step=epoch)

        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - train_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = epoch_num - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs

        print(
            f"Epoch [{epoch+1}/{epoch_num}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Epoch Time: {format_seconds(epoch_time)} | "
            f"ETA: {format_seconds(eta_seconds)}"
        )
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = model_dir / f'cifar100_{epoch+1}.pth'
            torch.save(net.state_dict(), str(checkpoint_path))
            print(f"模型已保存: {checkpoint_path}")
        
        if epoch >= warmup_epochs:
            scheduler.step()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        
        net.eval()
        sum_loss = 0
        sum_accuracy = 0
        
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                outputs = net(inputs)
                loss = loss_fn(outputs, labels)

                predicted = outputs.argmax(dim=1)
                correct = (predicted == labels).sum().item()
                sum_loss += loss.item()
                sum_accuracy += correct
        
        test_loss = sum_loss / len(test_dataloader)
        test_accuracy = sum_accuracy / len(test_dataloader.dataset)
        writer.add_scalar('test_loss', test_loss, global_step=epoch)
        writer.add_scalar('test_accuracy', test_accuracy, global_step=epoch)

        if test_accuracy > (best_test_accuracy + early_stop_min_delta):
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = model_dir / 'cifar100_best.pth'
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_test_accuracy': best_test_accuracy,
                    'test_loss': test_loss,
                },
                str(best_path)
            )
            print(f"         | 新最佳模型已保存: {best_path} (Acc={best_test_accuracy:.4f})")
        else:
            patience_counter += 1

        writer.add_scalar('best_test_accuracy', best_test_accuracy, global_step=epoch)
        
        print(f"         | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Accuracy: {test_accuracy:.4f} | "
              f"Best: {best_test_accuracy:.4f} (Epoch {best_epoch}) | "
              f"EarlyStop: {patience_counter}/{early_stop_patience}\n")

        if patience_counter >= early_stop_patience:
            print(f"Early stopping 触发，连续 {early_stop_patience} 个 epoch 无提升。")
            break
    
    writer.close()
    print("训练完成！")
