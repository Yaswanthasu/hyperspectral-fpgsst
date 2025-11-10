import torch
import torch.nn as nn

# -----------------------------------------------------
# 1️⃣ Squeeze-and-Excitation (SE) Block
# -----------------------------------------------------
class SEBlock(nn.Module):
    """Channel attention via squeeze and excitation."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


# -----------------------------------------------------
# 2️⃣ Channel and Spatial Attention (CBAM)
# -----------------------------------------------------
class CBAMBlock(nn.Module):
    """Channel + spatial attention combined."""
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        # Channel attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # Channel attention
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True).view(b, c)
        max_pool, _ = torch.max(x, dim=(2, 3), keepdim=True)
        max_pool = max_pool.view(b, c)
        ca = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid_spatial(self.conv_spatial(sa))
        x = x * sa
        return x
