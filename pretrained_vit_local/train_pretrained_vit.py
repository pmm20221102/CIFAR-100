from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    TENSORBOARD_AVAILABLE = False

from pretrained_vit import (
    build_pretrained_model,
    build_param_groups,
    count_params,
    freeze_all,
    unfreeze_all,
    unfreeze_head,
    unfreeze_last_blocks,
)
from load_cifar100_pretrained_vit import get_dataloaders


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def bytes_to_mb(value: int) -> float:
    return value / (1024 ** 2)


def get_gpu_memory_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    return {
        "allocated_mb": bytes_to_mb(torch.cuda.memory_allocated(device)),
        "reserved_mb": bytes_to_mb(torch.cuda.memory_reserved(device)),
        "peak_allocated_mb": bytes_to_mb(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_mb": bytes_to_mb(torch.cuda.max_memory_reserved(device)),
        "device_used_mb": bytes_to_mb(torch.cuda.mem_get_info(device)[1] - torch.cuda.mem_get_info(device)[0]),
        "device_free_mb": bytes_to_mb(torch.cuda.mem_get_info(device)[0]),
        "device_total_mb": bytes_to_mb(torch.cuda.mem_get_info(device)[1]),
    }


def format_gpu_memory(stats: dict) -> str:
    if not stats:
        return "GPU: N/A"
    return (
        f"GPU | alloc {stats['allocated_mb']:.0f} MB"
        f" | reserved {stats['reserved_mb']:.0f} MB"
        f" | peak_alloc {stats['peak_allocated_mb']:.0f} MB"
        f" | peak_reserved {stats['peak_reserved_mb']:.0f} MB"
        f" | device {stats['device_used_mb']:.0f}/{stats['device_total_mb']:.0f} MB"
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_mode(model: nn.Module, mode: str, unfreeze_last_blocks_count: int = 2):
    if mode == "linear_probe":
        freeze_all(model)
        unfreeze_head(model)
    elif mode == "partial_finetune":
        unfreeze_last_blocks(model, num_blocks=unfreeze_last_blocks_count)
    elif mode == "full_finetune":
        unfreeze_all(model)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_optimizer(
    model: nn.Module,
    mode: str,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
):
    if mode == "linear_probe":
        # only head trainable; use all head params since freeze_all was applied before unfreeze_head
        head_params = list(model.get_classifier().parameters())
        return torch.optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)

    groups = build_param_groups(model, backbone_lr=backbone_lr, head_lr=head_lr)
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: nn.Module, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        top1 = logits.argmax(dim=1)
        top1_correct += (top1 == labels).sum().item()

        top5 = logits.topk(k=min(5, logits.size(1)), dim=1).indices
        top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    top1_acc = top1_correct / max(total, 1)
    top5_acc = top5_correct / max(total, 1)
    return avg_loss, top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(description="Pretrained ViT fine-tuning on CIFAR-100 (local RTX 3060)")
    parser.add_argument("--config", type=str, default="config_pretrained_vit_local.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / args.config if not os.path.isabs(args.config) else Path(args.config)
    config = load_config(str(config_path))

    device_cfg = config.get("device", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    output_cfg = config.get("output", {})

    use_amp = bool(device_cfg.get("amp", True)) and torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Seed (optional but helpful for reproducibility)
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    # Build model
    model = build_pretrained_model(
        model_name=model_cfg.get("name", "deit_tiny_patch16_224"),
        num_classes=model_cfg.get("num_classes", 100),
        pretrained=bool(model_cfg.get("pretrained", True)),
    ).to(device)

    mode = model_cfg.get("mode", "partial_finetune")
    unfreeze_last_blocks_count = model_cfg.get("unfreeze_last_blocks", 2)
    apply_mode(model, mode=mode, unfreeze_last_blocks_count=unfreeze_last_blocks_count)

    total_params, trainable_params, frozen_params = count_params(model)
    print(f"Mode: {mode}")
    print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Frozen params: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
    print(f"Trainable ratio: {trainable_params/max(total_params,1):.4f}")

    # Data
    train_loader, test_loader, clean_train_loader, train_size, test_size = get_dataloaders(
        dataset_root=data_cfg.get("dataset_root", r"D:\Study\cifar100"),
        image_size=data_cfg.get("image_size", 224),
        train_batch_size=data_cfg.get("train_batch_size", 32),
        test_batch_size=data_cfg.get("test_batch_size", 64),
        num_workers=data_cfg.get("num_workers", 0),
    )

    # Optimizer / scheduler
    epochs = train_cfg.get("epochs", 40)
    warmup_epochs = train_cfg.get("warmup_epochs", 3)
    backbone_lr = train_cfg.get("backbone_lr", 5e-5)
    head_lr = train_cfg.get("head_lr", 2e-4)
    weight_decay = train_cfg.get("weight_decay", 0.05)
    grad_clip_norm = train_cfg.get("gradient_clip_norm", 1.0)
    accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)

    optimizer = build_optimizer(
        model,
        mode=mode,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=0.0)

    # AMP
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    # Output dirs
    model_dir = Path(output_cfg.get("model_dir", "models/pretrained_vit"))
    log_dir = Path(output_cfg.get("log_dir", "logs/pretrained_vit"))
    result_dir = Path(output_cfg.get("result_dir", "results/pretrained_vit"))
    save_every_epochs = output_cfg.get("save_every_epochs", 10)

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir)) if TENSORBOARD_AVAILABLE else None

    # Resume
    start_epoch = 0
    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = train_cfg.get("early_stop_patience", 12)
    min_delta = train_cfg.get("early_stop_min_delta", 0.001)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_test_acc = ckpt.get("best_test_accuracy", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}, best_acc={best_test_acc:.4f}")

    # Training
    label_smoothing = train_cfg.get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    print(f"Epochs: {epochs}, warmup: {warmup_epochs}, accumulation: {accumulation_steps}")
    print(f"AMP: {use_amp}, device: {device}")
    print(f"TensorBoard: {log_dir}")
    print("=" * 70)

    total_start = time.time()

    # Store target lr before warmup overrides it
    if not optimizer.param_groups or "initial_lr" not in optimizer.param_groups[0]:
        for pg in optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

    global_peak_allocated_mb = 0.0
    global_peak_reserved_mb = 0.0
    global_peak_device_used_mb = 0.0

    for epoch in range(start_epoch, epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()

        # Linear warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / max(1, warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * warmup_factor

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
                scaled_loss = loss / accumulation_steps

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if step % accumulation_steps == 0:
                if grad_clip_norm and grad_clip_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        # step scheduler after warmup
        if epoch + 1 > warmup_epochs:
            scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Evaluate
        test_loss, test_top1, test_top5 = evaluate(model, test_loader, device)

        # Clean train eval every 5 epochs
        clean_train_acc = None
        if (epoch + 1) % 5 == 0:
            clean_train_loss, clean_train_top1, _ = evaluate(model, clean_train_loader, device)
            clean_train_acc = clean_train_top1

        # Best check
        improved = test_top1 > best_test_acc + min_delta
        if improved:
            best_test_acc = test_top1
            best_epoch = epoch + 1
            patience_counter = 0
            # save best
            best_state = {
                "epoch": epoch + 1,
                "model_name": model_cfg.get("name", "deit_tiny_patch16_224"),
                "mode": mode,
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "best_test_accuracy": best_test_acc,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
            }
            torch.save(best_state, model_dir / "best.pth")
        else:
            patience_counter += 1

        # save last
        last_state = {
            "epoch": epoch + 1,
            "model_name": model_cfg.get("name", "deit_tiny_patch16_224"),
            "mode": mode,
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_test_accuracy": best_test_acc,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
        }
        torch.save(last_state, model_dir / "last.pth")

        # periodic checkpoint
        if save_every_epochs and (epoch + 1) % save_every_epochs == 0:
            periodic_state = {
                "epoch": epoch + 1,
                "model_name": model_cfg.get("name", "deit_tiny_patch16_224"),
                "mode": mode,
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "best_test_accuracy": best_test_acc,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
            }
            torch.save(periodic_state, model_dir / f"cifar100_{epoch+1}.pth")

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - total_start
        eta = (total_elapsed / max(epoch + 1 - start_epoch, 1)) * max(epochs - (epoch + 1), 0)

        # GPU mem stats
        gpu_stats = get_gpu_memory_stats(device)
        if gpu_stats:
            global_peak_allocated_mb = max(global_peak_allocated_mb, gpu_stats["peak_allocated_mb"])
            global_peak_reserved_mb = max(global_peak_reserved_mb, gpu_stats["peak_reserved_mb"])
            global_peak_device_used_mb = max(global_peak_device_used_mb, gpu_stats["device_used_mb"])
        mem_str = format_gpu_memory(gpu_stats)

        lr_backbone = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else 0.0
        lr_head = optimizer.param_groups[-1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Top1: {test_top1:.4f} | Top5: {test_top5:.4f} | "
            f"Best: {best_test_acc:.4f} (Epoch {best_epoch}) | Patience: {patience_counter}/{patience_limit} | "
            f"{mem_str} | Epoch: {format_seconds(epoch_time)} | ETA: {format_seconds(eta)}"
        )
        if clean_train_acc is not None:
            print(f"  Clean Train Top1: {clean_train_acc:.4f}")

        # TensorBoard
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch + 1)
            writer.add_scalar("train/top1", train_acc, epoch + 1)
            writer.add_scalar("test/loss", test_loss, epoch + 1)
            writer.add_scalar("test/top1", test_top1, epoch + 1)
            writer.add_scalar("test/top5", test_top5, epoch + 1)
            writer.add_scalar("lr/backbone", lr_backbone, epoch + 1)
            writer.add_scalar("lr/head", lr_head, epoch + 1)
            writer.add_scalar("best/top1", best_test_acc, epoch + 1)
            writer.add_scalar("patience", patience_counter, epoch + 1)
            if clean_train_acc is not None:
                writer.add_scalar("train/clean_top1", clean_train_acc, epoch + 1)

        # Early stopping
        if patience_counter >= patience_limit:
            print("Early stopping triggered.")
            break

    total_time = time.time() - total_start
    print(f"Training completed in {format_seconds(total_time)}.")

    # Summary JSON
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "N/A"
    gpu_total_mb = bytes_to_mb(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else 0
    summary = {
        "model_name": model_cfg.get("name", "deit_tiny_patch16_224"),
        "mode": mode,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "best_test_accuracy": best_test_acc,
        "best_epoch": best_epoch,
        "total_epochs_run": epochs,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_total_memory_mb": gpu_total_mb,
        "peak_torch_allocated_mb": global_peak_allocated_mb,
        "peak_torch_reserved_mb": global_peak_reserved_mb,
        "peak_device_used_mb": global_peak_device_used_mb,
        "amp": use_amp,
        "weight_decay": weight_decay,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "label_smoothing": label_smoothing,
        "gradient_clip_norm": grad_clip_norm,
        "accumulation_steps": accumulation_steps,
        "total_training_time_s": total_time,
    }
    summary_path = result_dir / "pretrained_vit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
