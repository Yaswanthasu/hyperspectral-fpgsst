# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# models/fpgsst.py
"""
Lightweight FPGSST implementation (Frequency-Prompt Guided Spectral-Spatial Transformer).
Designed to slot into your existing train.py with minimal changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# -------------------------
# FFT helper
# -------------------------
def get_fft_lowfreq_features(img: torch.Tensor, keep: int = 4) -> torch.Tensor:
    """
    img: (B, H, W) float tensor (single-channel)
    Returns tensor of shape (B, keep*keep) containing magnitude of low-frequency FFT coefficients.
    """
    # compute 2D FFT and center
    fft = torch.fft.fft2(img)            # (B, H, W), complex
    fft = torch.fft.fftshift(fft, dim=(-2, -1))
    mag = torch.abs(fft)                 # (B, H, W)
    B, H, W = mag.shape
    ch = keep
    start_h = (H - ch) // 2
    start_w = (W - ch) // 2
    block = mag[:, start_h:start_h+ch, start_w:start_w+ch]  # (B, ch, ch)
    return block.reshape(B, -1)  # (B, ch*ch)


# -------------------------
# transformer utility
# -------------------------
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, nhead=4, nlayers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # x: (B, seq_len, embed_dim)
        return self.encoder(x, src_key_padding_mask=mask)


# -------------------------
# Frequency Prompt Module
# -------------------------
class FrequencyPromptModule(nn.Module):
    def __init__(self, patch_size=9, keep=4, embed_dim=64):
        super().__init__()
        self.keep = keep
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.fc = nn.Sequential(
            nn.Linear(keep * keep, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, bands, H, W)
        # average across bands -> (B, H, W)
        img = x.mean(dim=1)
        fft_feats = get_fft_lowfreq_features(img, keep=self.keep)  # (B, keep*keep)
        prompt = self.fc(fft_feats)  # (B, embed_dim)
        return prompt


# -------------------------
# Spectral Former
# -------------------------
class SpectralFormer(nn.Module):
    def __init__(self, in_bands, embed_dim=64, nhead=4, nlayers=2, ff_dim=256, dropout=0.1):
        super().__init__()
        self.in_bands = in_bands
        self.embed_dim = embed_dim
        self.spec_proj = nn.Linear(1, embed_dim)
        self.transformer = SimpleTransformerEncoder(embed_dim, nhead=nhead, nlayers=nlayers,
                                                    dim_feedforward=ff_dim, dropout=dropout)
        self.pool_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, bands, H, W) -> produce (B, H*W, embed_dim)
        B, bands, H, W = x.shape
        # reshape -> (B*H*W, bands, 1)
        x_seq = x.permute(0, 2, 3, 1).reshape(-1, bands, 1)
        emb = self.spec_proj(x_seq)                      # (B*H*W, bands, embed_dim)
        out = self.transformer(emb)                      # (B*H*W, bands, embed_dim)
        pooled = out.mean(dim=1)                         # (B*H*W, embed_dim)
        pooled = self.pool_proj(pooled)                  # (B*H*W, embed_dim)
        pooled = pooled.view(B, H * W, self.embed_dim)   # (B, H*W, embed_dim)
        return pooled


# -------------------------
# Spatial Former
# -------------------------
class SpatialFormer(nn.Module):
    def __init__(self, embed_dim=64, nhead=4, nlayers=2, ff_dim=256, dropout=0.1):
        super().__init__()
        self.transformer = SimpleTransformerEncoder(embed_dim, nhead=nhead, nlayers=nlayers,
                                                    dim_feedforward=ff_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, N, E)
        B, N, E = x.shape
        if prompt is not None:
            p = prompt.unsqueeze(1).expand(-1, N, -1)  # (B, N, E)
            x = x + p
        out = self.transformer(x)  # (B, N, E)
        return out


# -------------------------
# Fusion + Classifier
# -------------------------
class SpectralSpatialFusion(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, spe_tokens: torch.Tensor, spa_tokens: torch.Tensor) -> torch.Tensor:
        x = torch.cat([spe_tokens, spa_tokens], dim=-1)  # (B, N, 2E)
        out = self.fuse(x)  # (B, N, E)
        return out


class FPGSST(nn.Module):
    def __init__(self, in_bands, num_classes, patch_size=9,
                 embed_dim=64, spec_nhead=4, spec_nlayers=2, spa_nhead=4, spa_nlayers=2,
                 fft_keep=4):
        super().__init__()
        self.in_bands = in_bands
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.fpm = FrequencyPromptModule(patch_size=patch_size, keep=fft_keep, embed_dim=embed_dim)
        self.speformer = SpectralFormer(in_bands, embed_dim=embed_dim,
                                        nhead=spec_nhead, nlayers=spec_nlayers,
                                        ff_dim=embed_dim * 4)
        self.spaformer = SpatialFormer(embed_dim=embed_dim, nhead=spa_nhead, nlayers=spa_nlayers,
                                       ff_dim=embed_dim * 4)
        self.fusion = SpectralSpatialFusion(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, bands, H, W)
        returns logits: (B, num_classes)
        """
        B, bands, H, W = x.shape
        spe_tokens = self.speformer(x)              # (B, H*W, E)
        prompt = self.fpm(x)                        # (B, E)
        spa_tokens = self.spaformer(spe_tokens, prompt=prompt)  # (B, H*W, E)
        fused = self.fusion(spe_tokens, spa_tokens)            # (B, H*W, E)
        pooled = fused.mean(dim=1)                              # (B, E)
        logits = self.classifier(pooled)                        # (B, num_classes)
        return logits

