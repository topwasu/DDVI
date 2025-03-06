import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

from .simple import MultiWayLinear


def resnet(num_outs, hidden_size, num_channels):
    model = resnet18()
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = MultiWayLinear(512, hidden_size, num_outs)
    return model


