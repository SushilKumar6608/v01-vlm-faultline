"""Small CNN baseline, trained from scratch, for the single frame and for the 8-tile sheet.

Both inputs go through the same trunk (applied to each tile on its own, weights shared). Each tile's feature map is reduced to 16
channels; the maps of all tiles are stacked along the channel axis in time order, so the head can compare the same place in
different tiles (this is what the sheet is for); a 3x3 conv, pooling and a linear layer follow. With one tile the head is the same
with 16 input channels. The tiles' timestamps are in the pixels, as for the vision-language models.
"""

from __future__ import annotations

import torch
from torch import nn

from ..sim import config as C

N_CLASSES = len(C.CLASSES)


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class TileCNN(nn.Module):
    """Input (B, T, 3, H, W) float in [0, 1]; T = 1 (single frame) or 8 (sheet). Output (B, 5) logits."""

    def __init__(self, n_tiles: int, width: int = 24, reduce: int = 16, hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.n_tiles = n_tiles
        self.trunk = nn.Sequential(_block(3, width), _block(width, 2 * width), _block(2 * width, 64), _block(64, 64))
        self.reduce = nn.Sequential(nn.Conv2d(64, reduce, 1, bias=False), nn.BatchNorm2d(reduce), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Conv2d(reduce * n_tiles, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden * 8, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        if t != self.n_tiles:
            raise ValueError(f"model built for {self.n_tiles} tile(s), got {t}")
        x = (x - 0.5) / 0.25
        f = self.reduce(self.trunk(x.flatten(0, 1)))  # (B*T, r, h, w)
        f = f.reshape(b, t * f.shape[1], f.shape[2], f.shape[3])
        return self.head(f)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
