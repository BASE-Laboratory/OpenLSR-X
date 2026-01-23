"""
The purpose of this file is to define all of the building blocks of
the used SRGAN architecture as seen in original paper
https://arxiv.org/abs/1609.04802v5
"""

import pytorch_lightning as pl
from torch import nn

from ..losses import PixelNorm


class ConvBlock(pl.LightningModule):
    """
    Conv -> BatchNorm -> ReLU activation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        # self.bn = nn.BatchNorm2d(out_channels) if use_bn is True else nn.Identity()
        self.bn = PixelNorm() if use_bn is True else nn.Identity()
        # self.bn = nn.InstanceNorm2d(out_channels) if use_bn is True else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator is True
            else nn.PReLU(num_parameters=out_channels)
        )
        self.use_act = use_act

    def forward(self, x):

        return (
            self.act(self.bn(self.conv(x)))
            if self.use_act is True
            else self.bn(self.conv(x))
        )


class UpsampleBlock(pl.LightningModule):
    """
    This blocks performs the upsampling using
    nn.PixelShuffle with a specific scale factor
    """

    def __init__(self, in_channels, scale_factor):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            in_channels * scale_factor**2,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.pixel_shuffle(self.conv(x)))


class ResidualBlock(pl.LightningModule):
    """
    Residual blocks from SRGAN Generator
    """

    def __init__(self, in_channels):
        super().__init__()

        self.block = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )

        self.block2 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            use_act=False,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )

    def forward(self, x):
        """
        Pass through residual blocks, then concatenate output with input
        to form the Elementwise Sum
        """
        out = self.block(x)
        out = self.block2(out)

        return out + x
