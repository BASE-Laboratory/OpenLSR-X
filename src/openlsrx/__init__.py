"""OpenLSR-X: Learnable 2D-to-3D Super-Resolution and Petrophysical Benchmark Suite for XCT."""

from .baselines import upsample_2d, upsample_3d
from .data import (
    DatasetObject,
    PairedSliceDataset,
    PairedVolumeDataModule,
    SyntheticPoreSpyDataModule,
    generate_porespy_volume,
    load_volume_any,
)
from .inference import (
    fuse_orthogonal_volumes,
    super_resolve_slice_stack,
    super_resolve_volume_orthogonal,
    super_resolve_volume_single_axis,
)
from .losses import PixelNorm, VGGLoss, custom_sigmoid
from .metrics import evaluate_image_metrics, evaluate_petrophysical_metrics
from .models import SRGAN, ConvBlock, Discriminator, Generator, ResidualBlock, UpsampleBlock

__version__ = "0.2.0"

__all__ = [
    "ConvBlock",
    "DatasetObject",
    "Discriminator",
    "Generator",
    "PairedSliceDataset",
    "PairedVolumeDataModule",
    "PixelNorm",
    "ResidualBlock",
    "SRGAN",
    "SyntheticPoreSpyDataModule",
    "UpsampleBlock",
    "VGGLoss",
    "__version__",
    "custom_sigmoid",
    "evaluate_image_metrics",
    "evaluate_petrophysical_metrics",
    "fuse_orthogonal_volumes",
    "generate_porespy_volume",
    "load_volume_any",
    "super_resolve_slice_stack",
    "super_resolve_volume_orthogonal",
    "super_resolve_volume_single_axis",
    "upsample_2d",
    "upsample_3d",
]
