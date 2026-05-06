import torch
import torch.nn as nn
from typing import Iterable


class LayerNormChannel(nn.Module):
    """LayerNorm over channels for (N,C,H,W) tensors by applying LN on last dim after permute."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (N,C,H,W) -> (N,H,W,C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNormChannel(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtCIFAR(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 100,
        depths: Iterable[int] = (2, 2, 6, 2),
        dims: Iterable[int] = (32, 64, 160, 320),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1.0,
    ):
        super().__init__()
        depths = list(depths)
        dims = list(dims)

        # Stem: keep 32x32 spatial resolution using 3x3 conv stride=1
        self.stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1), LayerNormChannel(dims[0]))

        self.stages = nn.ModuleList()
        total_blocks = sum(depths)
        block_idx = 0
        for stage_idx, d in enumerate(depths):
            blocks = []
            for _ in range(d):
                dp = drop_path_rate * block_idx / max(1, total_blocks - 1)
                blocks.append(ConvNeXtBlock(dims[stage_idx], drop_path=dp, layer_scale_init_value=layer_scale_init_value))
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))
            if stage_idx < len(depths) - 1:
                self.stages.append(nn.Sequential(LayerNormChannel(dims[stage_idx]), nn.Conv2d(dims[stage_idx], dims[stage_idx + 1], kernel_size=2, stride=2)))

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.stages:
            x = layer(x)

        # global average pool
        x = x.mean([-2, -1])  # (N, C)
        x = self.norm(x)
        return self.head(x)


def convnext_tiny_cifar100(num_classes=100, depths=(2, 2, 6, 2), dims=(32, 64, 160, 320), drop_path_rate=0.1, layer_scale_init_value=1.0):
    return ConvNeXtCIFAR(num_classes=num_classes, depths=depths, dims=dims, drop_path_rate=float(drop_path_rate), layer_scale_init_value=float(layer_scale_init_value))

