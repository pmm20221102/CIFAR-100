import os
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                         std=(0.2675, 0.2565, 0.2761)),
    transforms.RandomErasing(p=0.35, scale=(0.02, 0.25), ratio=(0.3, 3.3), value='random'),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                         std=(0.2675, 0.2565, 0.2761))
])

def get_dataloaders(dataset_root=r"D:\Study\cifar100", train_batch_size=192, test_batch_size=256,
                    num_workers=4):
    expected_dir = os.path.join(dataset_root, "cifar-100-python")
    if not os.path.isdir(expected_dir):
        raise FileNotFoundError(
            f"未找到 CIFAR-100 目录: {expected_dir}. "
            "请将官方解压后的 cifar-100-python 放到该路径下。"
        )

    train_dataset = CIFAR100(root=dataset_root, train=True, download=False, transform=train_transform)
    test_dataset = CIFAR100(root=dataset_root, train=False, download=False, transform=test_transform)

    # Windows 下多进程 DataLoader 需要避免在模块导入阶段创建对象。
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

    return train_dataloader, test_dataloader, len(train_dataset), len(test_dataset)
