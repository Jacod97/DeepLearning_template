import torch.nn as nn
from torch import Tensor

from models.cnn.dim import get_conv, get_norm

class ResidualBlock(nn.Module):
    expansion = 1 # 최종 출력 채널 = out_channels * expansion

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dim: int = 2):
        super().__init__()
        Conv = get_conv(dim)
        Norm = get_norm(dim)

        self.conv1 = Conv(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.batch_norm1 = Norm(out_channels)
        self.conv2 = Conv(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.batch_norm2 = Norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                Conv(in_channels, out_channels * self.expansion, kernel_size=1,
                     stride=stride, bias=False),
                Norm(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)
