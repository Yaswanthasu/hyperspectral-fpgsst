import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.attention import SEBlock

self.se = SEBlock(channels=32)
...
x = F.relu(self.bn3(self.conv3(x)))
x = self.se(x.squeeze(2))  # apply channel attention after 3D conv

class Simple3DCNN(nn.Module):
    """
    Baseline 3D CNN for Hyperspectral Image Classification.
    Input:  (B, C, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, in_bands, num_classes):
        super(Simple3DCNN, self).__init__()

        # 3D convolution layers
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1))
        self.bn1 = nn.BatchNorm3d(8)

        self.conv2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.bn2 = nn.BatchNorm3d(16)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(32)

        # 3D max pooling
        self.pool = nn.MaxPool3d((2, 2, 2))

        # Fully connected layers
        # Flatten to linear (adjust spatial size depending on your patch size)
        self.fc1 = nn.Linear(32 * ((in_bands // 8) + 1) * 1 * 1, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Input shape: (B, C, H, W) → add spectral dim
        x = x.unsqueeze(1)  # (B, 1, C, H, W)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
