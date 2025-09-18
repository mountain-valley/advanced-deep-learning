import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )

    def forward(self, x):
        x = x + self.net(x)
        x = F.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1), nn.ReLU(inplace=True),   # 96 -> 48
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True), # 48 -> 24
            nn.Conv2d(256, dim, 3, 1, 1), nn.ReLU(inplace=True), # 24 -> 24 (kernel 3, stride 1, padding 1 keeps size)
            ResBlock(dim),  # No size change
            ResBlock(dim)   # No size change
        )
    
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ConvTranspose2d(dim, 256, 4, 2, 1), nn.ReLU(inplace=True), # 24 -> 48
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True), # 48 -> 96
            ResBlock(128),
            nn.Conv2d(128, 3, 3, 1, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
