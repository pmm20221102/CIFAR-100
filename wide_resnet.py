import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(x)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out += shortcut
        return out


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth should be 6n+4, e.g. 16, 22, 28, 40")

        n = (depth - 4) // 6
        nStages = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.in_planes = nStages[0]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._make_layer(nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._make_layer(nStages[3], n, dropout_rate, stride=2)
        self.bn = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)

        self._init_weights()

    def _make_layer(self, planes, blocks, dropout_rate, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(WideBasic(self.in_planes, planes, dropout_rate, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def wide_resnet_cifar100(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100):
    return WideResNet(
        depth=depth,
        widen_factor=widen_factor,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
    )
