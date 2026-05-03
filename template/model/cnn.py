import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    expansion = 1  # 출력 채널 = out_channels × expansion. BasicBlock은 그대로 유지.

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity() # 항등 함수로 이해

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x) # Residual(잔차) Learning: output에 x를 다시 더해주는 skip(Shortcut)-connection임
        return self.relu(out)


class BottleneckBlock(nn.Module):
    expansion = 4  # 1x1로 줄였다 3x3 거치고 1x1로 4배 늘려 출력. ResNet-50/101/152용.

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # 1x1 conv: 채널 압축 (squeeze)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv: 메인 연산. stride는 여기에 적용.
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv: 4배로 확장 (expand)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) # Residual(잔차) Learning: skip-connection
        return self.relu(out)


class CNN(nn.Module):
    """ResNet을 가져다 왔습니다...

    block / num_blocks 조합으로 ResNet-18/34/50/101/152 모두 표현 가능:
        ResNet-18:  block=ResidualBlock,   num_blocks=(2, 2, 2, 2)
        ResNet-34:  block=ResidualBlock,   num_blocks=(3, 4, 6, 3)
        ResNet-50:  block=BottleneckBlock, num_blocks=(3, 4, 6, 3)
        ResNet-101: block=BottleneckBlock, num_blocks=(3, 4, 23, 3)
        ResNet-152: block=BottleneckBlock, num_blocks=(3, 8, 36, 3)
    """
    def __init__(
        self,
        num_classes: int,
        block: ResidualBlock | BottleneckBlock,
        num_blocks: tuple[int, int, int, int] = (2, 2, 2, 2),
    ):
        super().__init__()
        self.block = block
        self.in_channels = 64  # _make_layer가 호출되며 점점 갱신됨

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes) # Bottleneck이면 2048

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """첫 블록만 stride 적용, 이후 블록들은 stride=1로 쌓는다. expansion 반영해 다음 in_channels 갱신."""
        layers = [self.block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * self.block.expansion
        for _ in range(num_blocks - 1):
            layers.append(self.block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """입력 이미지를 ResNet에 통과시켜 logits 반환"""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out) # H, W를 1x1로
        out = out.flatten(1) # 평탄화 (B, 512 * expansion)
        out = self.fc(out)
        return out
