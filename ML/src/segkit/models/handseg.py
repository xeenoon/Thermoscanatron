"""MobileNet encoder + small U-Net-ish decoder, plus a hand-present head.

[B, 3, H, W] RGB -> ([B, 1, H, W] mask logits, [B, 1] hand-present logit).

Options:
  in_chans=4    a 4th input channel carries the previous frame's mask (0/1), for the phone's per-frame model;
  aux_classes   an extra 1x1 head on the decoder predicting background / skin / hair / clothing. Only used as a
                training signal (it teaches the features what skin is *not*); in eval mode forward() still
                returns just (mask, present), so export and the phone are unchanged.
"""

import timm
import torch
import torch.nn.functional as F
from torch import nn


def conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    """Upsample x2, concat skip, two 3x3 convs."""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(conv_bn_relu(cin + cskip, cout), conv_bn_relu(cout, cout))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class HandSegNet(nn.Module):
    def __init__(self, encoder: str = "mobilenetv3_small_100", pretrained: bool = False,
                 decoder_channels: tuple[int, ...] = (96, 64, 32), in_chans: int = 3, aux_classes: int = 0):
        super().__init__()
        # Feature maps at 1/4, 1/8, 1/16, 1/32.
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True,
                                         out_indices=(1, 2, 3, 4), in_chans=in_chans)
        c4, c8, c16, c32 = self.encoder.feature_info.channels()
        d16, d8, d4 = decoder_channels
        self.up16 = UpBlock(c32, c16, d16)
        self.up8 = UpBlock(d16, c8, d8)
        self.up4 = UpBlock(d8, c4, d4)
        self.head = nn.Conv2d(d4, 1, 1)
        self.aux = nn.Conv2d(d4, aux_classes, 1) if aux_classes else None
        # "Is there a hand in this crop at all?" from the deepest features.
        self.presence = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c32, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f4, f8, f16, f32 = self.encoder(x)
        y = self.up16(f32, f16)
        y = self.up8(y, f8)
        y = self.up4(y, f4)
        mask = F.interpolate(self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False)
        if self.aux is not None and self.training:
            aux = F.interpolate(self.aux(y), size=x.shape[-2:], mode="bilinear", align_corners=False)
            return mask, self.presence(f32), aux
        return mask, self.presence(f32)


def load_init(model: nn.Module, state: dict) -> None:
    """Load weights for fine-tuning, tolerating a different input channel count (3-channel checkpoint into a
    4-channel model: the extra channel starts at zero, so the model begins exactly as the checkpoint) and a missing
    or new aux head."""
    own = model.state_dict()
    for k, v in state.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k] = v
        elif own[k].dim() == 4 and own[k].shape[0] == v.shape[0] and own[k].shape[2:] == v.shape[2:]:
            w = torch.zeros_like(own[k])
            n = min(v.shape[1], w.shape[1])
            w[:, :n] = v[:, :n]
            own[k] = w
    model.load_state_dict(own)


class ProbabilityHead(nn.Module):
    """Deployment wrapper: logits -> probabilities, so the phone gets [0, 1] directly."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask, present = self.net(x)
        return torch.sigmoid(mask), torch.sigmoid(present)
