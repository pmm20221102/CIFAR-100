import os

import timm
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader


def build_transforms(image_size: int = 224, is_training: bool = True):
    # If timm config can resolve, prefer it; otherwise fallback to standard ImageNet transforms
    try:
        data_cfg = timm.data.resolve_model_data_config(model=None)
        # Some timm versions require a model instance; if failed, fallback
        train_tfms = timm.data.create_transform(
            input_size=data_cfg["input_size"],
            is_training=True,
            mean=data_cfg["mean"],
            std=data_cfg["std"],
            interpolation=data_cfg.get("interpolation", "bicubic"),
            crop_pct=data_cfg.get("crop_pct", 0.875),
        )
        val_tfms = timm.data.create_transform(
            input_size=data_cfg["input_size"],
            is_training=False,
            mean=data_cfg["mean"],
            std=data_cfg["std"],
            interpolation=data_cfg.get("interpolation", "bicubic"),
            crop_pct=data_cfg.get("crop_pct", 0.875),
        )
        return train_tfms if is_training else val_tfms
    except Exception:
        pass

    # Fallback: explicit 224 pipeline
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if is_training:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3), value="random"),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_dataloaders(
    dataset_root: str,
    image_size: int = 224,
    train_batch_size: int = 32,
    test_batch_size: int = 64,
    num_workers: int = 0,
):
    expected_dir = os.path.join(dataset_root, "cifar-100-python")
    if not os.path.isdir(expected_dir):
        raise FileNotFoundError(
            f"未找到 CIFAR-100 目录: {expected_dir}. "
            "请将官方解压后的 cifar-100-python 放到该路径下。"
        )

    train_transform = build_transforms(image_size=image_size, is_training=True)
    test_transform = build_transforms(image_size=image_size, is_training=False)

    # Clean train transform (no random augmentation) for periodic clean accuracy eval
    clean_train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_dataset = CIFAR100(root=dataset_root, train=True, download=False, transform=train_transform)
    test_dataset = CIFAR100(root=dataset_root, train=False, download=False, transform=test_transform)
    clean_train_dataset = CIFAR100(root=dataset_root, train=True, download=False, transform=clean_train_transform)

    # Windows: keep num_workers small to avoid issues
    num_workers = max(0, min(num_workers, 4))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    clean_train_dataloader = DataLoader(
        clean_train_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_dataloader, test_dataloader, clean_train_dataloader, len(train_dataset), len(test_dataset)
