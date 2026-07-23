import torch
import torch.nn as nn


def conv_bn_act(in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channels = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn_act(in_channels, hidden_channels, kernel_size=1, padding=0))
        layers.append(
            conv_bn_act(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                groups=hidden_channels,
            )
        )
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out


class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, transformer_dim, depth=2, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.shortcut = nn.Identity()
        self.local_rep = nn.Sequential(
            conv_bn_act(in_channels, in_channels, kernel_size=3),
            conv_bn_act(in_channels, in_channels, kernel_size=1, padding=0),
        )
        self.proj_in = conv_bn_act(in_channels, transformer_dim, kernel_size=1, padding=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=int(transformer_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.proj_out = conv_bn_act(transformer_dim, in_channels, kernel_size=1, padding=0)
        self.fusion = nn.Sequential(
            conv_bn_act(in_channels * 2, in_channels, kernel_size=3),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        local_features = self.local_rep(x)
        transformer_input = self.proj_in(local_features)

        batch_size, channels, height, width = transformer_input.shape
        tokens = transformer_input.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        transformer_features = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)

        transformer_features = self.proj_out(transformer_features)
        fused = torch.cat([local_features, transformer_features], dim=1)
        return self.fusion(fused) + self.shortcut(x)


class MobileViT(nn.Module):
    def __init__(self, num_classes=100, dropout=0.1):
        super().__init__()

        self.stem = nn.Sequential(
            conv_bn_act(3, 32, kernel_size=3, stride=1),
            InvertedResidual(32, 32, stride=1, expand_ratio=2),
        )

        self.stage1 = nn.Sequential(
            InvertedResidual(32, 48, stride=2, expand_ratio=2),
            InvertedResidual(48, 48, stride=1, expand_ratio=2),
        )

        self.mobilevit1 = MobileViTBlock(48, transformer_dim=96, depth=2, num_heads=4, dropout=dropout)

        self.stage2 = nn.Sequential(
            InvertedResidual(48, 64, stride=2, expand_ratio=2),
            InvertedResidual(64, 64, stride=1, expand_ratio=2),
        )

        self.mobilevit2 = MobileViTBlock(64, transformer_dim=128, depth=2, num_heads=4, dropout=dropout)

        self.head = nn.Sequential(
            conv_bn_act(64, 128, kernel_size=1, padding=0),
            nn.Dropout(p=dropout),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.mobilevit1(x)
        x = self.stage2(x)
        x = self.mobilevit2(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilevit_small(num_classes=100, dropout=0.1):
    return MobileViT(num_classes=num_classes, dropout=dropout)
