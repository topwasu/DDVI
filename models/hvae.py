import torch
import torch.nn.functional as F
from torch import nn


class Q2(nn.Module):
    def __init__(self, img_processor, fc):
        super().__init__()
        self.img_processor = img_processor
        self.fc = fc
    
    def forward(self, x, z):
        zx = self.img_processor(x)
        return self.fc(torch.cat((zx, z), 1))