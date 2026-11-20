"""OME-Zarr v0.4 data-contract exporter and deprecated legacy OpenImpala input exporter."""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from basecoat import Volume, VolumeSet, write_ome_zarr, write_volume

from ..metrics.petrophysics import ThresholdMethod, segment_pore_space

ImpalaDirection = Literal["x", "y", "z"]
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def export_volume_ome_zarr(
    volume: np.ndarray,
    store_path: str | Path,
    *,
    voxel_size_um: float | tuple[float, float, float] | None = None,
    include_pore_labels: bool = True,
    threshold_method: ThresholdMethod = "otsu",
    fixed_threshold: float | None = None,
    pore_is_low: bool = True,
    pore_label: int = 1,
    solid_label: int = 0,
    multiscale_levels: int = 2,
    overwrite: bool = True,
) -> Path:
    """Export a 3D XCT volume (and optional mode-downsampled pore labels) as an OME-Zarr v0.4 store.

    Follows the BASE Lab ``ECOSYSTEM.md`` data contract via ``basecoat`` so Layer 3 solvers
    and Layer 4 evaluators consume calibrated volumes and labels without tool-specific glue.
    """
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(
            f"export_volume_ome_zarr expects a 3D volume (D, H, W), got shape {arr.shape}."
        )

    img_vol = Volume.from_array(arr, voxel_size_um=voxel_size_um)
    if not include_pore_labels:
        return write_ome_zarr(
            img_vol, store_path, multiscale_levels=multiscale_levels, overwrite=overwrite
        )

    pore_mask, thresh_val = segment_pore_space(
        arr,
        method=threshold_method,
        fixed_value=fixed_threshold,
        pore_is_low=pore_is_low,
    )
    labeled = np.where(pore_mask, np.uint8(pore_label), np.uint8(solid_label))
    lbl_vol = Volume(
        labeled,
        axes=img_vol.axes,
        label_names={int(solid_label): "solid", int(pore_label): "pore"},
        provenance={"threshold_method": threshold_method, "threshold_value": float(thresh_val)},
    )
    vset = VolumeSet(img_vol, labels={"phase": lbl_vol})
    return write_ome_zarr(
        vset, store_path, multiscale_levels=multiscale_levels, overwrite=overwrite
    )


def export_for_openimpala(
    volume_or_mask: np.ndarray,
    output_dir: str | Path,
    name: str = "sample",
    threshold_method: ThresholdMethod = "otsu",
    fixed_threshold: float | None = None,
    pore_is_low: bool = True,
    pore_label: int = 1,
    solid_label: int = 0,
    direction: ImpalaDirection = "x",
    max_grid_size: int = 64,
    *,
    _warn_deprecated: bool = True,
) -> tuple[Path, Path]:
    """Export a 3D XCT volume or binary mask as uncompressed 8-bit TIFF + AMReX ParmParse ``.inputs``.

    .. deprecated:: 0.2.2
        Tool-to-solver bespoke exporters are deprecated under ``ECOSYSTEM.md`` and will be
        removed in ``v0.3.0`` once ``OpenImpala v5.1`` reads ``.ome.zarr`` directly; prefer
        :func:`export_volume_ome_zarr` (or ``basecoat.write_ome_zarr``) for new workflows.
    """
    if _warn_deprecated:
        warnings.warn(
            "export_for_openimpala is deprecated under ECOSYSTEM.md and scheduled for removal "
            "in OpenLSR-X v0.3.0 once OpenImpala v5.1 accepts OME-Zarr directly; prefer "
            "openlsrx.io.export_volume_ome_zarr (or basecoat.write_ome_zarr).",
            DeprecationWarning,
            stacklevel=2,
        )

    clean_name = str(name).strip()
    if not clean_name or ".." in clean_name or _SAFE_NAME_RE.fullmatch(clean_name) is None:
        raise ValueError(
            f"Invalid OpenImpala sample name {name!r}: expected alphanumeric/underscore/hyphen/dot "
            "identifier without path separators, '..', quotes, or whitespace."
        )

    arr = np.asarray(volume_or_mask)
    if arr.ndim != 3:
        raise ValueError(f"OpenImpala requires a 3D volume (D, H, W), got shape {arr.shape}.")
    if direction not in ("x", "y", "z"):
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}.")
    if not (0 <= pore_label <= 255 and 0 <= solid_label <= 255 and pore_label != solid_label):
        raise ValueError(
            f"pore_label ({pore_label}) and solid_label ({solid_label}) must be distinct uint8 integers."
        )

    if arr.dtype == bool:
        pore_mask = arr
    else:
        pore_mask, _ = segment_pore_space(
            arr.astype(np.float32, copy=False),
            method=threshold_method,
            fixed_value=fixed_threshold,
            pore_is_low=pore_is_low,
        )

    labeled = np.where(pore_mask, np.uint8(pore_label), np.uint8(solid_label))
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tiff_path = out_root / f"{clean_name}.tif"
    inputs_path = out_root / f"{clean_name}_openimpala.inputs"

    write_volume(
        Volume.uncalibrated(
            labeled, label_names={int(solid_label): "solid", int(pore_label): "pore"}
        ),
        tiff_path,
        compression=None,
        overwrite=True,
    )

    depth_z, height_y, width_x = labeled.shape
    porosity = float(pore_mask.mean())
    blocking_factor = max(1, math.gcd(width_x, math.gcd(height_y, math.gcd(depth_z, 8))))
    eff_max_grid = max(
        blocking_factor,
        ((max(1, int(max_grid_size)) + blocking_factor - 1) // blocking_factor) * blocking_factor,
    )
    inputs_text = (
        "# Generated by OpenLSR-X for BASE-Laboratory/OpenImpala (AMReX Tortuosity)\n"
        f"# Sample: {name} | Shape (Z, Y, X) = ({depth_z}, {height_y}, {width_x}) | Porosity = {porosity:.6f}\n"
        'eb2.geom_type = "tiff"\n'
        f'torteff.tiff_filename = "{tiff_path.name}"\n'
        f"torteff.phase = {pore_label}\n"
        f'torteff.direction = "{direction}"\n'
        f"amr.n_cell = {width_x} {height_y} {depth_z}\n"
        f"amr.blocking_factor = {blocking_factor}\n"
        f"amr.max_grid_size = {eff_max_grid}\n"
        "torteff.verbose = 1\n"
    )
    inputs_path.write_text(inputs_text, encoding="utf-8")
    return tiff_path, inputs_path
