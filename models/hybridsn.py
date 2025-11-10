{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "765af200",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch.nn as nn\n",
    "import torch\n",
    "\n",
    "class HybridSN(nn.Module):\n",
    "    def __init__(self, in_bands, num_classes):\n",
    "        super().__init__()\n",
    "        # spectral 3D part (reduces spectral dim)\n",
    "        self.spec = nn.Sequential(\n",
    "            nn.Conv3d(1, 8, kernel_size=(7,1,1), padding=(3,0,0)), nn.ReLU(),\n",
    "            nn.Conv3d(8, 16, kernel_size=(5,1,1), padding=(2,0,0)), nn.ReLU(),\n",
    "        )\n",
    "        # after spectral convs, collapse spectral axis and use 2D convs\n",
    "        self.spatial = nn.Sequential(\n",
    "            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),\n",
    "            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),\n",
    "            nn.AdaptiveAvgPool2d(1),\n",
    "        )\n",
    "        self.fc = nn.Linear(64, num_classes)\n",
    "\n",
    "    def forward(self, x):\n",
    "        # x: (batch, bands, H, W) or (batch, 1, bands, H, W) depending on your loader\n",
    "        if x.dim() == 4:  # (B,C,H,W)\n",
    "            x = x.unsqueeze(1)  # (B,1,C,H,W)\n",
    "        x = self.spec(x)     # (B, C', C_spec', H, W)\n",
    "        # collapse spectral dimension (C_spec') into channel\n",
    "        x = x.squeeze(3).squeeze(3) if x.shape[3]==1 else x  # handle shape carefully\n",
    "        # better to reshape so you have (B, channels, H, W)\n",
    "        b, c, s, h, w = x.shape\n",
    "        x = x.view(b, c*s, h, w)\n",
    "        x = self.spatial(x)\n",
    "        x = x.view(b, -1)\n",
    "        return self.fc(x)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
