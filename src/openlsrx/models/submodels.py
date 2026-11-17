"""Generator and Discriminator submodels for OpenLSR-X."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from ..losses import custom_sigmoid
from .blocks import ConvBlock, NormType, ResidualBlock, UpsampleBlock

OutputActivation = Literal["custom_sigmoid", "sigmoid", "clamp", "identity"]
DEFAULT_DISC_FEATURES: tuple[int, ...] = (64, 64, 128, 128, 256, 256, 512, 512)


class Generator(nn.Module):
    """SRResNet / SRGAN Generator adapted for single-channel XCT data with reflect padding."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        num_blocks: int = 10,
        super_res_factor: int = 4,
        sigmoid_k: float = 1.0,
        norm_type: NormType = "pixel",
        output_activation: OutputActivation = "custom_sigmoid",
    ) -> None:
        super().__init__()
        if super_res_factor < 1 or (super_res_factor & (super_res_factor - 1)) != 0:
            raise ValueError(
                f"super_res_factor must be a positive power of 2 (1, 2, 4, 8, ...), got {super_res_factor}."
            )
        if output_activation not in ("custom_sigmoid", "sigmoid", "clamp", "identity"):
            raise ValueError(
                f"Unsupported output_activation={output_activation!r}. "
                "Expected 'custom_sigmoid', 'sigmoid', 'clamp', or 'identity'."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_blocks = int(num_blocks)
        self.super_res_factor = int(super_res_factor)
        self.sigmoid_k = float(sigmoid_k)
        self.norm_type: NormType = norm_type
        self.output_activation: OutputActivation = output_activation

        self.initial = ConvBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=9,
            stride=1,
            padding=4,
            padding_mode="reflect",
            use_bn=False,
        )
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(self.out_channels, norm_type=self.norm_type)
                for _ in range(self.num_blocks)
            ]
        )
        self.last_res = ConvBlock(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            use_act=False,
            norm_type=self.norm_type,
        )

        num_upsamples = int(math.log2(self.super_res_factor))
        self.upsample = nn.Sequential(
            *[
                UpsampleBlock(in_channels=self.out_channels, scale_factor=2)
                for _ in range(num_upsamples)
            ]
        )

        self.final_conv = ConvBlock(
            in_channels=self.out_channels,
            out_channels=self.in_channels,
            kernel_size=9,
            stride=1,
            padding=4,
            padding_mode="reflect",
            use_bn=False,
            use_act=False,
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
        initial = self.initial(x)
        out = self.res_blocks(initial)
        out = self.last_res(out) + initial
        out = self.upsample(out)
        out = self.final_conv(out)
        return self._apply_output_activation(out)


class Discriminator(nn.Module):
    """SRGAN Discriminator outputting raw logits for ``BCEWithLogitsLoss``."""

    def __init__(
        self,
        in_channels: int = 1,
        features: Sequence[int] | None = None,
        adaptive_size: int = 6,
        norm_type: NormType = "pixel",
    ) -> None:
        super().__init__()
        feature_list = list(DEFAULT_DISC_FEATURES if features is None else features)
        if not feature_list:
            raise ValueError("Discriminator features must contain at least one channel size.")

        disc_blocks: list[nn.Module] = []
        curr_channels = int(in_channels)
        for idx, feature in enumerate(feature_list):
            disc_blocks.append(
                ConvBlock(
                    in_channels=curr_channels,
                    out_channels=int(feature),
                    discriminator=True,
                    kernel_size=3,
                    stride=1 + (idx % 2),
                    padding=1,
                    use_act=True,
                    use_bn=(idx != 0),
                    norm_type=norm_type,
                )
            )
            curr_channels = int(feature)

        self.disc_blocks = nn.Sequential(*disc_blocks)
        self.highest_feature = int(feature_list[-1])
        self.adaptive_size = int(adaptive_size)
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.adaptive_size, self.adaptive_size)),
            nn.Flatten(),
            nn.Linear(
                self.highest_feature * self.adaptive_size * self.adaptive_size,
                self.highest_feature * 2,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.highest_feature * 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.dense(self.disc_blocks(x))
