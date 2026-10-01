"""CNN from the original paper (ConvNet2 in its conv2.py), for 1x64x1536 piano rolls."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(c_in, c_out, kernel, pool_stride, conv_stride=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=conv_stride, padding=1),
        nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=pool_stride))


class ConvNet(nn.Module):
    FLAT = 1024 * 2 * 3

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            _block(1, 16, 3, 2, conv_stride=(1, 4)),
            _block(16, 32, 3, 2),
            _block(32, 64, 3, 2),
            _block(64, 128, 3, (1, 2)),
            _block(128, 256, 3, (1, 2)),
            _block(256, 512, 2, 2),
            _block(512, 1024, 2, 2),
        )
        self.fc1 = nn.Linear(self.FLAT, 512)
        self.fc2 = nn.Linear(512, 96)
        self.fc3 = nn.Linear(96, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x).flatten(1)
        out = F.dropout(F.relu(self.fc1(out)), p=0.3, training=self.training)
        out = F.dropout(F.relu(self.fc2(out)), p=0.3, training=self.training)
        return self.fc3(out)
