from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from pretrained_vit import build_pretrained_model, count_params
from train_pretrained_vit import apply_mode
from load_cifar100_pretrained_vit import get_dataloaders


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained ViT on CIFAR-100")
    parser.add_argument("--config", type=str, default="config_pretrained_vit_local.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/pretrained_vit/best.pth")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    config_path = project_dir / args.config if not os.path.isabs(args.config) else Path(args.config)
    config = load_config(str(config_path))

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    output_cfg = config.get("output", {})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_pretrained_model(
        model_name=model_cfg.get("name", "deit_tiny_patch16_224"),
        num_classes=model_cfg.get("num_classes", 100),
        pretrained=False,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Mode: prefer saved config, fallback to current config
    mode = ckpt.get("mode", model_cfg.get("mode", "partial_finetune"))
    unfreeze_last_blocks_count = model_cfg.get("unfreeze_last_blocks", 2)
    apply_mode(model, mode=mode, unfreeze_last_blocks_count=unfreeze_last_blocks_count)

    total_params, trainable_params, frozen_params = count_params(model)

    # Data
    _, test_loader, _, train_size, test_size = get_dataloaders(
        dataset_root=data_cfg.get("dataset_root", r"D:\Study\cifar100"),
        image_size=data_cfg.get("image_size", 224),
        train_batch_size=data_cfg.get("train_batch_size", 32),
        test_batch_size=data_cfg.get("test_batch_size", 64),
        num_workers=data_cfg.get("num_workers", 0),
    )

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    with torch.inference_mode():
        for images, labels in test_loader:
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
    elapsed = time.time() - start

    avg_loss = total_loss / max(total, 1)
    top1_acc = top1_correct / max(total, 1)
    top5_acc = top5_correct / max(total, 1)
    images_per_sec = total / max(elapsed, 1e-6)

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None

    result = {
        "checkpoint": str(args.checkpoint),
        "model_name": model_cfg.get("name", "deit_tiny_patch16_224"),
        "mode": mode,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "test_samples": total,
        "test_loss": avg_loss,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "elapsed_s": elapsed,
        "images_per_sec": images_per_sec,
        "peak_gpu_memory_mb": peak_mb,
    }

    print(json.dumps(result, indent=2))

    # Save result
    result_dir = Path(output_cfg.get("result_dir", "results/pretrained_vit"))
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / "eval_pretrained_vit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
