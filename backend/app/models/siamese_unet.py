"""FC-Siam-diff: a fully-convolutional Siamese U-Net for change detection.

Reference architecture: R. C. Daudt, B. Le Saux, A. Boulch, "Fully Convolutional
Siamese Networks for Change Detection", ICIP 2018 (arXiv:1810.08462).

How it works
------------
* ONE encoder with shared weights processes BEFORE and AFTER separately
  ("Siamese"), so both images are mapped into the same feature space.
* At every encoder level the skip connection passed to the decoder is the
  absolute difference |f_before - f_after|, so the decoder sees *where the
  features disagree* at multiple scales.
* The decoder upsamples back to full resolution and outputs one logit per pixel;
  sigmoid(logit) is the change probability.

This is an independent re-implementation, not a port of the authors' weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH_NAME = "fc_siam_diff"


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, n: int = 2, dropout: float = 0.0):
        layers: list[nn.Module] = []
        for i in range(n):
            layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
                       nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class FCSiamDiff(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 16, depth: int = 4, dropout: float = 0.2):
        super().__init__()
        self.in_channels, self.base_channels, self.depth = in_channels, base_channels, depth
        chs = [base_channels * 2 ** i for i in range(depth)]  # e.g. 16, 32, 64, 128
        self.encoders = nn.ModuleList()
        c = in_channels
        for i, ch in enumerate(chs):
            self.encoders.append(ConvBlock(c, ch, n=2 if i < 2 else 3, dropout=dropout))
            c = ch
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            ch = chs[i]
            self.ups.append(nn.ConvTranspose2d(c, ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(ch * 2, ch, n=2 if i < 2 else 3, dropout=dropout if i > 0 else 0.0))
            c = ch
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    @property
    def stride(self) -> int:
        return 2 ** self.depth

    def _encode(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        return skips, x

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        s1, _ = self._encode(before)
        s2, x = self._encode(after)
        for up, dec, f1, f2 in zip(self.ups, self.decoders, reversed(s1), reversed(s2)):
            x = up(x)
            x = dec(torch.cat([x, torch.abs(f1 - f2)], dim=1))
        return self.head(x)  # logits, shape (N, 1, H, W)

    def arch_config(self) -> dict:
        return {"in_channels": self.in_channels, "base_channels": self.base_channels, "depth": self.depth}


def build_model(arch: str, config: dict) -> nn.Module:
    if arch != ARCH_NAME:
        raise ValueError(f"Unsupported architecture '{arch}' (expected '{ARCH_NAME}').")
    return FCSiamDiff(**{k: config[k] for k in ("in_channels", "base_channels", "depth") if k in config})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
