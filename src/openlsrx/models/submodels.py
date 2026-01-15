"""
This model contains all of submodels such
as the Generator and Discriminator
"""

import pytorch_lightning as pl
from torch import nn
from .blocks import ConvBlock, UpsampleBlock, ResidualBlock
import math
from ..losses import custom_sigmoid


class Generator(pl.LightningModule):
    """
    Describes the whole Generator Network as described in SRGAN paper
    https://arxiv.org/abs/1609.04802v5
    in_channels = 1 (grayscale)
    out_channels = 64 (from paper)
    res_blocks = 16 (from paper)
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=64,
        num_blocks=10,
        super_res_factor=4,
        sigmoid_k=1.0,
    ):
        super().__init__()

        self.sigmoid_k = float(sigmoid_k)

        self.initial = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=9,
            stride=1,
            padding=4,
            padding_mode="reflect",
            use_bn=False,
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(out_channels) for block in range(num_blocks)]
        )
        # code above creates a list of length num_blocks of ResidualBlocks
        # these are then sequentially executed

        self.last_res = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            use_act=False,
        )

        self.num_upsamples = [i for i in range(int(math.log2(super_res_factor)))]

        self.upsample = nn.Sequential(
            *[
                UpsampleBlock(in_channels=out_channels, scale_factor=2)
                for factor in self.num_upsamples
            ]
        )

        self.final_conv = ConvBlock(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=9,
            stride=1,
            padding=4,
            padding_mode="reflect",
            use_bn=False,
            use_act=False,
        )

    def forward(self, x):

        initial = self.initial(x)
        out = self.res_blocks(initial)
        out = self.last_res(out)
        out = out + initial
        out = self.upsample(out)
        out = self.final_conv(out)

        # return torch.sigmoid(out)
        return custom_sigmoid(out, k=self.sigmoid_k)


class Discriminator(pl.LightningModule):
    """
    This class describes the Discriminator architecture as
    described in paper https://arxiv.org/abs/1609.04802v5
    """

    def __init__(
        self,
        in_channels=1,
        features=[64, 64, 128, 128, 256, 256, 512, 512],
        adaptive_size=6,
    ):
        super().__init__()

        disc_blocks = []

        for idx, feature in enumerate(features):
            disc_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=feature,
                    discriminator=True,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.disc_blocks = nn.Sequential(*disc_blocks)

        self.highest_feature = features[-1]
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

    def forward(self, x):

        out = self.disc_blocks(x)
        out = self.dense(out)

        # note: BCEWithLogitsLoss includes a sigmoid layer
        # so do not apply sigmoid here

        return out
