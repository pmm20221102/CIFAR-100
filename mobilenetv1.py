import torch
import torch.nn.functional as F
import torch.nn as nn


def make_divisible(value, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class MobileNet(nn.Module):
    def conv_dw(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, 
                     padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def __init__(self, width_mult=1.25, dropout=0.3, stage512_depth=3):
        super(MobileNet, self).__init__()

        c32 = make_divisible(32 * width_mult)
        c64 = make_divisible(64 * width_mult)
        c128 = make_divisible(128 * width_mult)
        c256 = make_divisible(256 * width_mult)
        c512 = make_divisible(512 * width_mult)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c32),
            nn.ReLU(inplace=True),
        )
        self.conv_dw2 = self.conv_dw(c32, c32, 1)
        self.conv_dw3 = self.conv_dw(c32, c64, 1)

        self.conv_dw4 = self.conv_dw(c64, c64, 1)
        self.conv_dw5 = self.conv_dw(c64, c128, 2)

        self.conv_dw6 = self.conv_dw(c128, c128, 1)
        self.conv_dw7 = self.conv_dw(c128, c256, 2)

        self.conv_dw8 = self.conv_dw(c256, c256, 1)
        self.conv_dw9 = self.conv_dw(c256, c512, 2)

        stage512_blocks = []
        for _ in range(stage512_depth):
            stage512_blocks.append(self.conv_dw(c512, c512, 1))
        self.stage512 = nn.Sequential(*stage512_blocks)

        self.dropout = nn.Dropout(p=dropout)

        # 改为 100 类
        self.fc = nn.Linear(c512, 100)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_dw2(out)
        out = self.conv_dw3(out)
        out = self.conv_dw4(out)
        out = self.conv_dw5(out)
        out = self.conv_dw6(out)
        out = self.conv_dw7(out)
        out = self.conv_dw8(out)
        out = self.conv_dw9(out)
        out = self.stage512(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def mobilenetv1_small(width_mult=1.25, dropout=0.3, stage512_depth=3):
    return mobilenet(width_mult=width_mult, dropout=dropout, stage512_depth=stage512_depth)
