"""Loss functions, normalization layers, and activations for OpenLSR-X."""

from __future__ import annotations

import warnings
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
    """Channel-wise pixel feature vector normalization (Karras et al., 2018).

    Upcasts internally to ``float32`` before computing ``x^2`` and ``rsqrt`` so
    mixed-precision (``16-mixed`` / FP16) training never overflows ``65504`` or
    underflows ``epsilon`` to ``0.0`` (``0 / 0 = NaN``).
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = max(float(epsilon), 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.to(torch.float32)
        normed = x_f32 * torch.rsqrt(torch.mean(x_f32.square(), dim=1, keepdim=True) + self.epsilon)
        return normed.to(dtype=x.dtype)


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
        allow_offline_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.vgg_layers = int(vgg_layers)
        self.pretrained = bool(pretrained)
        self.normalize_inputs = bool(normalize_inputs)

        backbone = self._load_vgg_features(
            pretrained=self.pretrained,
            vgg_layers=self.vgg_layers,
            allow_offline_fallback=allow_offline_fallback,
        )
        for param in backbone.parameters():
            param.requires_grad = False
        # Store outside _modules so frozen ImageNet weights never bloat checkpoints
        self._vgg_holder: tuple[nn.Module] = (backbone,)

        self.criterion: nn.Module = nn.MSELoss() if reduction == "mse" else nn.L1Loss()

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @staticmethod
    def _load_vgg_features(
        pretrained: bool,
        vgg_layers: int,
        allow_offline_fallback: bool,
    ) -> nn.Module:
        if pretrained:
            try:
                full = vgg19(weights=VGG19_Weights.DEFAULT)
            except Exception as exc:
                if not allow_offline_fallback:
                    raise
                warnings.warn(
                    f"Unable to load pretrained VGG19 weights ({exc}); "
                    "falling back to random frozen VGG19 features on air-gapped node.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                full = vgg19(weights=None)
        else:
            full = vgg19(weights=None)
        features = full.features[:vgg_layers].eval()
        del full  # Immediately release ~500 MB dense classifier parameters
        return features

    def _apply(self, fn):  # type: ignore[override]
        super()._apply(fn)
        self._vgg_holder[0]._apply(fn)
        return self

    def _prepare(self, x: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        out = x.to(dtype=target_dtype)
        if out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)
        elif out.shape[1] != 3:
            c_mid = out.shape[1] // 2
            out = out[:, c_mid : c_mid + 1].repeat(1, 3, 1, 1)
        if self.normalize_inputs:
            out = (out - self.mean.to(dtype=target_dtype, device=out.device)) / self.std.to(
                dtype=target_dtype, device=out.device
            )
        return out

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        backbone = self._vgg_holder[0]
        backbone.eval()
        param = next(backbone.parameters(), None)
        if param is not None and param.device != generated.device:
            backbone.to(device=generated.device)
        weight_dtype = param.dtype if param is not None else torch.float32

        gen_in = self._prepare(generated, target_dtype=weight_dtype)
        tgt_in = self._prepare(target, target_dtype=weight_dtype)
        gen_features = backbone(gen_in)
        with torch.no_grad():
            tgt_features = backbone(tgt_in)
        return self.criterion(gen_features.to(torch.float32), tgt_features.to(torch.float32))
