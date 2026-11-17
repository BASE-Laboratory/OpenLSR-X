"""Classical interpolation baselines for 2D slice and 3D volumetric XCT upsampling."""

from .interpolation import Interp2DMode, Interp3DMode, upsample_2d, upsample_3d

__all__ = ["Interp2DMode", "Interp3DMode", "upsample_2d", "upsample_3d"]
