from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import timm


def build_pretrained_model(
    model_name: str = "deit_tiny_patch16_224",
    num_classes: int = 100,
    pretrained: bool = True,
) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def _head_param_names(model: nn.Module) -> List[str]:
    # timm classification heads are usually `model.get_classifier()` -> module name `head` or `fc`
    head = model.get_classifier()
    # find module name for head
    head_module_name = None
    for n, m in model.named_modules():
        if m is head:
            head_module_name = n
            break
    if head_module_name is None:
        return []
    # collect all parameter names under that module
    prefix = head_module_name + "."
    head_param_names = [n for n, _ in model.named_parameters() if n == head_module_name or n.startswith(prefix)]
    return head_param_names


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_head(model: nn.Module) -> None:
    head_names = _head_param_names(model)
    for n, p in model.named_parameters():
        if n in head_names:
            p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 2) -> None:
    # Common timm ViT structure: patch embed, blocks, norm, head
    # We unfreeze: last N blocks, norm, head
    if not hasattr(model, "blocks"):
        raise ValueError("Model does not have `model.blocks` attribute; cannot unfreeze last blocks.")

    blocks = model.blocks
    total_blocks = len(blocks)

    # freeze first
    freeze_all(model)

    # unfreeze last N blocks
    for idx in range(total_blocks - num_blocks, total_blocks):
        for p in blocks[idx].parameters():
            p.requires_grad = True

    # unfreeze final norm if present
    if hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad = True

    # unfreeze head
    unfreeze_head(model)


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def build_param_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
) -> List[dict]:
    head_names = _head_param_names(model)

    backbone_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in head_names:
            head_params.append(p)
        else:
            backbone_params.append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    return groups


def count_params(model: nn.Module) -> Tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen
