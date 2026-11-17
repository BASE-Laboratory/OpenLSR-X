"""2D/2.5D-to-3D orthogonal XCT super-resolution inference and epistemic uncertainty."""

from .orthogonal import (
    AxisName,
    FusionMode,
    compute_orthogonal_uncertainty,
    fuse_orthogonal_volumes,
    super_resolve_slice_stack,
    super_resolve_volume_orthogonal,
    super_resolve_volume_single_axis,
)

__all__ = [
    "AxisName",
    "FusionMode",
    "compute_orthogonal_uncertainty",
    "fuse_orthogonal_volumes",
    "super_resolve_slice_stack",
    "super_resolve_volume_orthogonal",
    "super_resolve_volume_single_axis",
]
