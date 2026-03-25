"""
Three model architectures of increasing complexity.
All accept input shape  (batch, 1, n_mels, time_steps).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ═══════════════════════════════════════════════════════════
# 1) Lightweight CNN  —  fast, good for real-time on M2
# ═══════════════════════════════════════════════════════════
class LightCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(True), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(True), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(True), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


# ═══════════════════════════════════════════════════════════
# 2) ResNet-style  —  more accurate
# ═══════════════════════════════════════════════════════════
class _ResBlock(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inc, outc, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(outc)
        self.skip  = nn.Sequential()
        if stride != 1 or inc != outc:
            self.skip = nn.Sequential(
                nn.Conv2d(inc, outc, 1, stride, bias=False),
                nn.BatchNorm2d(outc))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class DeepfakeResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.inc = 32
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._layer(32,  2, 1)
        self.layer2 = self._layer(64,  2, 2)
        self.layer3 = self._layer(128, 2, 2)
        self.layer4 = self._layer(256, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, 128),
            nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, num_classes))

    def _layer(self, outc, n, stride):
        layers = [_ResBlock(self.inc, outc, stride)]
        self.inc = outc
        for _ in range(1, n):
            layers.append(_ResBlock(outc, outc))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.pool(x).flatten(1)
        return self.head(x)


# ═══════════════════════════════════════════════════════════
# 3) SE-ResNet  —  best accuracy, slightly heavier
# ═══════════════════════════════════════════════════════════
class _SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c // r, bias=False), nn.ReLU(True),
            nn.Linear(c // r, c, bias=False), nn.Sigmoid())

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        return x * self.fc(s).unsqueeze(-1).unsqueeze(-1)


class _SEResBlock(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inc, outc, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(outc)
        self.se    = _SE(outc)
        self.skip  = nn.Sequential()
        if stride != 1 or inc != outc:
            self.skip = nn.Sequential(
                nn.Conv2d(inc, outc, 1, stride, bias=False),
                nn.BatchNorm2d(outc))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + self.skip(x))


class SEResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.inc = 32
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._layer(32,  2, 1)
        self.layer2 = self._layer(64,  2, 2)
        self.layer3 = self._layer(128, 2, 2)
        self.layer4 = self._layer(256, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, 128),
            nn.ReLU(True), nn.Linear(128, num_classes))

    def _layer(self, outc, n, stride):
        layers = [_SEResBlock(self.inc, outc, stride)]
        self.inc = outc
        for _ in range(1, n):
            layers.append(_SEResBlock(outc, outc))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(self.pool(x).flatten(1))


# ═══════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "light_cnn": LightCNN,
    "resnet":    DeepfakeResNet,
    "se_resnet": SEResNet,
}


def get_model(name="resnet", num_classes=2):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Choose from {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](num_classes)
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model '{name}': {total:,} params ({train:,} trainable)")
    return model


if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 251)
    for name in MODEL_REGISTRY:
        m = get_model(name)
        print(f"  {name} output: {m(x).shape}\n")