"""HandSegNet's MobileNetV3-Small U-Net with panel heads.

[B, 3, H, W] RGB -> ([B, 6, H, W] dense outputs, [B, 1] panel-present logit). Dense channels:
  0 cell-area logit, 1 gridline logit, 2..5 within-cell phase (sin/cos 2 pi u, sin/cos pi v), unnormalised.
"""

import torch
import torch.nn.functional as F
from torch import nn

from segkit.models.handseg import HandSegNet

N_DENSE = 6


class PanelNet(HandSegNet):
    def __init__(self, encoder: str = "mobilenetv3_small_100", pretrained: bool = False,
                 decoder_channels: tuple[int, ...] = (96, 64, 32)):
        super().__init__(encoder, pretrained, decoder_channels)
        self.head = nn.Conv2d(decoder_channels[-1], N_DENSE, 1)


class PanelProbabilityHead(nn.Module):
    """Deployment wrapper: mask and line probabilities, phase pairs normalised to unit length."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dense, present = self.net(x)
        probs = torch.sigmoid(dense[:, :2])
        pu = F.normalize(dense[:, 2:4], dim=1)
        pv = F.normalize(dense[:, 4:6], dim=1)
        return torch.cat([probs, pu, pv], dim=1), torch.sigmoid(present)
