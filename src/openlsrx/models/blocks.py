"""Core neural network building blocks for the OpenLSR-X SRGAN architecture."""

from __future__ import annotations

from typing import Literal

import torch.nn.functional as F
from torch import Tensor, nn

from ..losses import PixelNorm

NormType = Literal["pixel", "batch", "instance", "none"]


def _build_norm(norm_type: NormType, channels: int, enabled: bool) -> nn.Module:
    if not enabled or norm_type == "none":
        return nn.Identity()
    if norm_type == "pixel":
        return PixelNorm()
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(
        f"Unsupported norm_type={norm_type!r}. Expected 'pixel', 'batch', 'instance', or 'none'."
    )


def _safe_conv2d(conv: nn.Conv2d, x: Tensor) -> Tensor:
    """Execute ``conv(x)`` falling back from ``reflect`` to ``replicate`` on tiny spatial inputs."""
    if conv.padding_mode == "reflect" and isinstance(conv.padding, tuple):
        ph, pw = int(conv.padding[0]), int(conv.padding[1])
        if (ph > 0 or pw > 0) and (x.shape[-2] <= ph or x.shape[-1] <= pw):
            padded = F.pad(x, (pw, pw, ph, ph), mode="replicate")
            return F.conv2d(
                padded,
                conv.weight,
                conv.bias,
                stride=conv.stride,
                padding=0,
                dilation=conv.dilation,
                groups=conv.groups,
            )
    return conv(x)


class ConvBlock(nn.Module):
    """2D convolution followed by optional normalization and PReLU/LeakyReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        discriminator: bool = False,
        use_act: bool = True,
        use_bn: bool = True,
        norm_type: NormType = "pixel",
        **kwargs,
    ) -> None:
        super().__init__()
        use_norm = bool(use_bn) and norm_type != "none"
        use_bias = not (use_norm and norm_type in ("batch", "instance"))
        if "bias" in kwargs:
            use_bias = bool(kwargs.pop("bias"))

        self.conv = nn.Conv2d(in_channels, out_channels, bias=use_bias, **kwargs)
        self.bn = _build_norm(norm_type=norm_type, channels=out_channels, enabled=use_norm)
        self.use_act = bool(use_act)
        self.act: nn.Module = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn(_safe_conv2d(self.conv, x))
        return self.act(out) if self.use_act else out


class UpsampleBlock(nn.Module):
    """Sub-pixel convolution upsampling block using ``nn.PixelShuffle`` and reflect padding."""

    def __init__(self, in_channels: int, scale_factor: int = 2) -> None:
        super().__init__()
        if scale_factor < 1:
            raise ValueError(f"scale_factor must be >= 1, got {scale_factor}.")
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * (scale_factor**2),
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.pixel_shuffle(_safe_conv2d(self.conv, x)))


class ResidualBlock(nn.Module):
    """Residual block with reflect-padded convolutions and configurable normalization."""

    def __init__(
        self,
        in_channels: int,
        norm_type: NormType = "pixel",
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.block = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            use_bn=use_bn,
            norm_type=norm_type,
        )
        self.block2 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            use_act=False,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            use_bn=use_bn,
            norm_type=norm_type,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.block2(self.block(x))
        return out + x
