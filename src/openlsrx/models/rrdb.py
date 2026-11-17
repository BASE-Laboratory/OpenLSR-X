"""Normalization-free RRDB (ESRGAN-style) 2D generator with arbitrary-scale XCT upsampling."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..losses import custom_sigmoid
from .blocks import UpsampleBlock
from .submodels import OutputActivation


class DenseResidualBlock(nn.Module):
    """5-layer densely connected convolutional block without batch/pixel normalization.

    Omitting feature normalization preserves absolute X-ray attenuation contrasts across
    batches (Wang et al., ESRGAN, ECCV Workshops 2018).
    """

    def __init__(
        self,
        channels: int = 64,
        growth_channels: int = 32,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.res_scale = float(res_scale)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1, padding_mode="reflect")
        self.conv2 = nn.Conv2d(
            channels + growth_channels, growth_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.conv3 = nn.Conv2d(
            channels + 2 * growth_channels, growth_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.conv4 = nn.Conv2d(
            channels + 3 * growth_channels, growth_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.conv5 = nn.Conv2d(
            channels + 4 * growth_channels, channels, 3, 1, 1, padding_mode="reflect"
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + self.res_scale * x5


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block chaining three ``DenseResidualBlock`` units."""

    def __init__(
        self,
        channels: int = 64,
        growth_channels: int = 32,
        res_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.res_scale = float(res_scale)
        self.rdb1 = DenseResidualBlock(channels, growth_channels, res_scale=res_scale)
        self.rdb2 = DenseResidualBlock(channels, growth_channels, res_scale=res_scale)
        self.rdb3 = DenseResidualBlock(channels, growth_channels, res_scale=res_scale)

    def forward(self, x: Tensor) -> Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return x + self.res_scale * out


class ArbitraryScaleUpsample(nn.Module):
    """Hybrid sub-pixel + resize-convolution upsampler supporting any scale >= 1.0.

    - Exact powers of two (``1, 2, 4, 8``) use learned ``UpsampleBlock`` (``PixelShuffle``) stages.
    - Non-power-of-two beamline ratios (e.g., ``2.5`` for I13-2 ``0.8125 -> 0.325 um`` or ``5.0``
      for ``1.625 -> 0.325 um``) combine ``floor(log2(s))`` sub-pixel stages with an exact
      bicubic resize-convolution refinement stage.
    """

    def __init__(self, channels: int = 64, scale_factor: float | int = 2) -> None:
        super().__init__()
        sf = float(scale_factor)
        if sf < 1.0:
            raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}.")
        self.scale_factor = sf

        n_pow2 = int(math.floor(math.log2(sf) + 1e-9))
        self.pow2_scale = 2**n_pow2
        self.residual_scale = sf / float(self.pow2_scale)

        self.pow2_blocks = nn.Sequential(
            *[UpsampleBlock(in_channels=channels, scale_factor=2) for _ in range(n_pow2)]
        )
        self.has_residual = not math.isclose(self.residual_scale, 1.0, rel_tol=1e-5, abs_tol=1e-5)
        self.refine_conv: nn.Module | None = (
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, padding_mode="reflect"),
                nn.LeakyReLU(0.2, inplace=True),
            )
            if self.has_residual
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        target_h = int(round(x.shape[-2] * self.scale_factor))
        target_w = int(round(x.shape[-1] * self.scale_factor))
        out = self.pow2_blocks(x)
        if self.has_residual and self.refine_conv is not None:
            out = F.interpolate(out, size=(target_h, target_w), mode="bicubic", align_corners=False)
            out = self.refine_conv(out)
        return out


class RRDBGenerator(nn.Module):
    """ESRGAN-style RRDB Generator adapted for quantitative single-channel XCT slices."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        growth_channels: int = 32,
        num_blocks: int = 8,
        super_res_factor: float | int = 2,
        sigmoid_k: float = 1.0,
        output_activation: OutputActivation = "clamp",
        out_img_channels: int = 1,
    ) -> None:
        super().__init__()
        if output_activation not in ("custom_sigmoid", "sigmoid", "clamp", "identity"):
            raise ValueError(
                f"Unsupported output_activation={output_activation!r}. "
                "Expected 'custom_sigmoid', 'sigmoid', 'clamp', or 'identity'."
            )

        self.in_channels = int(in_channels)
        self.out_img_channels = int(out_img_channels)
        self.out_channels = int(out_channels)
        self.growth_channels = int(growth_channels)
        self.num_blocks = int(num_blocks)
        self.super_res_factor = float(super_res_factor)
        self.sigmoid_k = float(sigmoid_k)
        self.output_activation: OutputActivation = output_activation

        self.conv_first = nn.Conv2d(
            self.in_channels, self.out_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.trunk = nn.Sequential(
            *[
                RRDB(channels=self.out_channels, growth_channels=self.growth_channels)
                for _ in range(self.num_blocks)
            ]
        )
        self.trunk_conv = nn.Conv2d(
            self.out_channels, self.out_channels, 3, 1, 1, padding_mode="reflect"
        )
        self.upsample = ArbitraryScaleUpsample(
            channels=self.out_channels, scale_factor=self.super_res_factor
        )
        final_conv = nn.Conv2d(
            self.out_channels, self.out_img_channels, 3, 1, 1, padding_mode="reflect"
        )
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)
        final_conv.weight.data.mul_(0.1)
        self.hr_head = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            final_conv,
        )

    def _apply_output_activation(self, x: Tensor) -> Tensor:
        if self.output_activation == "custom_sigmoid":
            return custom_sigmoid(x, k=self.sigmoid_k)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "clamp":
            return torch.clamp(x, 0.0, 1.0)
        return x

    def forward(self, x: Tensor) -> Tensor:
        target_h = int(round(x.shape[-2] * self.super_res_factor))
        target_w = int(round(x.shape[-1] * self.super_res_factor))
        if x.shape[1] == self.out_img_channels:
            center = x
        else:
            c_mid = x.shape[1] // 2
            center = x[:, c_mid : c_mid + self.out_img_channels, :, :]
        base = F.interpolate(center, size=(target_h, target_w), mode="bicubic", align_corners=False)
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(feat))
        feat = feat + trunk
        up = self.upsample(feat)
        residual = self.hr_head(up)
        if self.output_activation in ("clamp", "identity"):
            return self._apply_output_activation(base + residual)
        return self._apply_output_activation(residual)
