"""Neural network models and blocks for OpenLSR-X."""

from .blocks import ConvBlock, NormType, ResidualBlock, UpsampleBlock
from .rrdb import RRDB, ArbitraryScaleUpsample, DenseResidualBlock, RRDBGenerator
from .sr3d import (
    Generator3D,
    PairedSubCubeDataset,
    ResidualBlock3D,
    SR3DResNet,
    build_3d_subcube_dataloaders,
    super_resolve_volume_3d,
)
from .srgan import SRGAN, BackboneType, GANLossType
from .submodels import Discriminator, Generator, OutputActivation
from .swinir import (
    ResidualSwinBlock,
    SwinIRGenerator,
    SwinTransformerLayer,
    WindowAttention2D,
)

__all__ = [
    "ArbitraryScaleUpsample",
    "BackboneType",
    "ConvBlock",
    "DenseResidualBlock",
    "Discriminator",
    "GANLossType",
    "Generator",
    "Generator3D",
    "NormType",
    "OutputActivation",
    "PairedSubCubeDataset",
    "RRDB",
    "RRDBGenerator",
    "ResidualBlock",
    "ResidualBlock3D",
    "ResidualSwinBlock",
    "SR3DResNet",
    "SRGAN",
    "SwinIRGenerator",
    "SwinTransformerLayer",
    "UpsampleBlock",
    "WindowAttention2D",
    "build_3d_subcube_dataloaders",
    "super_resolve_volume_3d",
]
