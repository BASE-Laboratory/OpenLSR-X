"""Loss functions, normalization layers, and activations for OpenLSR-X."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19


def custom_sigmoid(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Numerically stable parameterized sigmoid ``1 / (1 + exp(-k * x))``.

    Uses ``torch.sigmoid(k * x)`` so mixed-precision (FP16/BF16) training never
    overflows ``torch.exp`` on large negative inputs.
    """
    return torch.sigmoid(k * x)


class PixelNorm(nn.Module):
    """Channel-wise pixel feature vector normalization (Karras et al., 2018)."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x.square(), dim=1, keepdim=True) + self.epsilon)


class VGGLoss(nn.Module):
    """Perceptual feature-space loss computed with frozen VGG-19 features.

    Inputs in ``[0, 1]`` with shape ``(B, 1, H, W)`` or ``(B, 3, H, W)`` are
    normalized with ImageNet channel statistics before feature extraction.
    Frozen VGG weights are intentionally excluded from ``state_dict`` so model
    checkpoints remain compact and strictly loadable across configurations.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        vgg_layers: int = 18,
        reduction: Literal["mse", "l1"] = "mse",
        pretrained: bool = True,
        normalize_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.vgg_layers = int(vgg_layers)
        self.pretrained = bool(pretrained)
        self.normalize_inputs = bool(normalize_inputs)

        weights = VGG19_Weights.DEFAULT if self.pretrained else None
        backbone = vgg19(weights=weights).features[: self.vgg_layers].eval()
        for param in backbone.parameters():
            param.requires_grad = False
        # Store outside _modules so frozen ImageNet weights (~550 MB) never bloat checkpoints
        self._vgg_holder: tuple[nn.Module] = (backbone,)

        self.criterion: nn.Module = nn.MSELoss() if reduction == "mse" else nn.L1Loss()

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _apply(self, fn):  # type: ignore[override]
        super()._apply(fn)
        self._vgg_holder[0]._apply(fn)
        return self

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.normalize_inputs:
            x = (x - self.mean.to(dtype=x.dtype, device=x.device)) / self.std.to(
                dtype=x.dtype, device=x.device
            )
        return x

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        backbone = self._vgg_holder[0]
        backbone.eval()
        gen_in = self._prepare(generated)
        tgt_in = self._prepare(target)
        gen_features = backbone(gen_in)
        with torch.no_grad():
            tgt_features = backbone(tgt_in)
        return self.criterion(gen_features, tgt_features)
