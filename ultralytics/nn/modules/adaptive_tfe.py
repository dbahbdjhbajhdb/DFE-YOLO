"""TFE alignment with ASFF-style, content-dependent spatial scale weights.

Requires only PyTorch, so this file can also be imported outside Ultralytics.
Input order: high, middle, low spatial resolution. Output matches the middle.
"""

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ("AdaptiveTFE",)


def _cbs(c1, c2, kernel=1):
    return nn.Sequential(
        nn.Conv2d(c1, c2, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(),
    )


class AdaptiveTFE(nn.Module):
    """Fuse three adjacent scales at the middle input's spatial resolution.

    Args:
        c1: Three input channel counts, ordered high-to-low resolution.
        c2: Common aligned channel count and final output channel count.
        weight_channels: Per-branch compression width for the weight predictor.
        fusion: 'adaptive', 'mean', or 'concat' for controlled ablations.

    Adaptive weights have shape [B, 3, H, W] and sum to one over dimension 1.
    They are shared across channels, but vary by image and spatial location.
    'concat' adds a 1x1 projection back to c2, preserving the external interface.
    It is an interface-matched control, not an exact paper TFE reproduction.
    """

    def __init__(self, c1, c2, weight_channels=8, fusion="adaptive"):
        super().__init__()
        if len(c1) != 3 or min(*c1, c2, weight_channels) < 1:
            raise ValueError("Expected three positive input channel counts and positive widths")
        if fusion not in {"adaptive", "mean", "concat"}:
            raise ValueError("fusion must be 'adaptive', 'mean', or 'concat'")
        self.fusion = fusion
        self.project = nn.ModuleList(_cbs(c, c2) for c in c1)
        self.high_refine = _cbs(c2, c2, 3)
        self.low_refine = _cbs(c2, c2, 3)
        if fusion == "adaptive":
            self.weight_reduce = nn.ModuleList(
                nn.Sequential(nn.Conv2d(c2, weight_channels, 1), nn.SiLU()) for _ in c1
            )
            self.weight_logits = nn.Conv2d(3 * weight_channels, 3, 1)
            # Start near equal weighting without blocking the reducer gradients.
            nn.init.normal_(self.weight_logits.weight, std=0.01)
            nn.init.zeros_(self.weight_logits.bias)
        self.concat_project = _cbs(3 * c2, c2) if fusion == "concat" else nn.Identity()
        self.refine = _cbs(c2, c2, 3)

    def _align(self, inputs):
        if len(inputs) != 3:
            raise ValueError("Expected [high_resolution, middle_resolution, low_resolution]")
        high, middle, low = inputs
        target = middle.shape[-2:]
        if any(h < m or m < l for h, m, l in zip(high.shape[-2:], target, low.shape[-2:])):
            raise ValueError("Input spatial resolutions must be ordered high >= middle >= low")
        high, middle, low = [project(x) for project, x in zip(self.project, inputs)]
        # Adaptive pooling handles odd sizes and non-square feature maps too.
        high = F.adaptive_max_pool2d(high, target) + F.adaptive_avg_pool2d(high, target)
        high = self.high_refine(high)
        low = self.low_refine(F.interpolate(low, size=target, mode="nearest"))
        return high, middle, low

    def forward_with_weights(self, inputs):
        """Return (output, spatial_weights); weights are None for the concat control."""
        aligned = self._align(inputs)
        if self.fusion == "concat":
            fused = self.concat_project(torch.cat(aligned, dim=1))
            weights = None
        else:
            if self.fusion == "adaptive":
                # Only compact gating descriptors are concatenated. The actual
                # feature fusion below is a weighted sum, retaining c2 channels.
                descriptors = torch.cat(
                    [reduce(x) for reduce, x in zip(self.weight_reduce, aligned)], dim=1
                )
                logits = self.weight_logits(descriptors)
                weights = logits.float().softmax(dim=1).to(aligned[0].dtype)
            else:
                b, _, h, w = aligned[1].shape
                weights = aligned[1].new_full((b, 3, h, w), 1.0 / 3.0)
            fused = sum(weights[:, i : i + 1] * x for i, x in enumerate(aligned))
        return self.refine(fused), weights

    def forward(self, inputs):
        return self.forward_with_weights(inputs)[0]
