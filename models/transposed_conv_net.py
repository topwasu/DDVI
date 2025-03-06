import torch
import torch.nn as nn
import torch.nn.functional as F

class TransposedConvNet(nn.Module):
    def __init__(self, input_size=2):
        super().__init__()
        self.feature_dim = 64
        self.img_channels = 3
        self.fc = nn.Linear(input_size, self.feature_dim)
        self.t_convs = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.img_channels, kernel_size=4, stride=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, self.feature_dim, 1, 1)
        x = self.t_convs(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvNet(nn.Module):
    def __init__(self, output_size=2, fc=None):
        super().__init__()
        self.feature_dim = 128
        self.img_channels = 3
        self.convs = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Flatten()
        )
        if fc is not None:
            self.fc = fc
        else:
            self.fc = nn.Linear(self.feature_dim, output_size)

    def forward(self, x):
        z = self.convs(x)
        z = self.fc(z)
        return z
    
class BareConvNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Flatten()
        )

    def forward(self, x):
        z = self.convs(x)
        return z