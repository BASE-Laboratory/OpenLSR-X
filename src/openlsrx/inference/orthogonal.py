"""2D-to-3D single-axis and multi-axis orthogonal XCT volume super-resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import torch
from scipy.ndimage import zoom
from torch import nn

AxisName = Literal["yz", "xz", "xy"]
FusionMode = Literal["mean", "median", "fourier"]

AXIS_TO_DIM: dict[AxisName, int] = {
    "yz": 0,  # slices along dim 0 have in-plane axes (1, 2)
    "xz": 1,  # slices along dim 1 have in-plane axes (0, 2)
    "xy": 2,  # slices along dim 2 have in-plane axes (0, 1)
}


def _move_slice_axis_first(vol: np.ndarray, axis: AxisName) -> np.ndarray:
    dim = AXIS_TO_DIM[axis]
    return np.moveaxis(vol, dim, 0)


def _move_slice_axis_back(vol: np.ndarray, axis: AxisName) -> np.ndarray:
    dim = AXIS_TO_DIM[axis]
    return np.moveaxis(vol, 0, dim)


@torch.no_grad()
def super_resolve_slice_stack(
    generator: nn.Module,
    slices: np.ndarray | torch.Tensor,
    batch_size: int = 8,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Run a 2D ``Generator`` over a ``(N, h, w)`` stack of slices and return ``(N, H, W)``."""
    arr = np.asarray(slices, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D slice stack (N, h, w), got shape {arr.shape}.")

    if device is None:
        try:
            dev = next(generator.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    was_training = generator.training
    generator.eval()
    outputs: list[np.ndarray] = []
    try:
        for start in range(0, arr.shape[0], max(1, int(batch_size))):
            batch = torch.from_numpy(arr[start : start + batch_size]).unsqueeze(1).to(dev)
            pred = generator(batch).squeeze(1).to(torch.float32).cpu().numpy()
            outputs.append(pred)
    finally:
        if was_training:
            generator.train()

    return np.concatenate(outputs, axis=0)


def super_resolve_volume_single_axis(
    generator: nn.Module,
    low_volume: np.ndarray,
    scale_factor: int = 2,
    axis: AxisName = "yz",
    interpolate_stack_axis: bool = True,
    batch_size: int = 8,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Super-resolve a 3D XCT volume ``(D, H, W)`` using a 2D generator along ``axis``.

    Args:
        generator: Trained 2D super-resolution generator with upsampling factor ``scale_factor``.
        low_volume: Input 3D volume. If ``interpolate_stack_axis=True`` (for true isotropic
            3D LR scans of shape ``(d, h, w)``), the stacking axis is first cubic-interpolated
            by ``scale_factor`` before applying the 2D generator to each orthogonal plane.
            Set ``interpolate_stack_axis=False`` when ``low_volume`` is already high-resolution
            along ``axis`` (anisotropic slab input).
        scale_factor: Spatial upsampling factor of ``generator``.
        axis: Orthogonal plane to super-resolve (``'yz'``, ``'xz'``, or ``'xy'``).
        interpolate_stack_axis: Whether to upsample the slicing axis by ``scale_factor``.
        batch_size: Number of 2D slices per forward pass.
        device: Torch device override.
    """
    vol = np.asarray(low_volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}.")

    ordered = _move_slice_axis_first(vol, axis)  # (N_axis, h, w)
    if interpolate_stack_axis and scale_factor > 1:
        ordered = zoom(
            ordered,
            zoom=(scale_factor, 1, 1),
            order=3,
            mode="reflect",
        ).astype(np.float32, copy=False)
        ordered = np.clip(ordered, 0.0, 1.0)

    sr_ordered = super_resolve_slice_stack(
        generator=generator,
        slices=ordered,
        batch_size=batch_size,
        device=device,
    )
    return _move_slice_axis_back(sr_ordered, axis)


def fuse_orthogonal_volumes(
    volumes: Mapping[AxisName, np.ndarray],
    mode: FusionMode = "mean",
    clamp_01: bool = True,
) -> np.ndarray:
    """Fuse aligned single-axis 3D super-resolved volumes into an isotropic 3D reconstruction.

    Supported fusion modes:
    - ``'mean'``: Equal-weight voxel-wise average across provided axes.
    - ``'median'``: Voxel-wise median (robust against isolated slice artifacts).
    - ``'fourier'``: 3D Fourier-space directional fusion weighting each single-axis
      reconstruction by its in-plane spatial frequencies ``k_in^2 / (k_x^2 + k_y^2 + k_z^2)``.
    """
    if not volumes:
        raise ValueError("At least one orthogonal volume is required for fusion.")

    items = [(ax, np.asarray(v, dtype=np.float32)) for ax, v in volumes.items()]
    ref_shape = items[0][1].shape
    for ax, v in items:
        if v.ndim != 3 or v.shape != ref_shape:
            raise ValueError(
                f"All orthogonal volumes must be 3D with identical shape {ref_shape}; "
                f"axis {ax!r} has shape {v.shape}."
            )

    if len(items) == 1:
        out = items[0][1].copy()
        return np.clip(out, 0.0, 1.0) if clamp_01 else out

    if mode == "mean":
        stacked = np.stack([v for _, v in items], axis=0)
        out = np.mean(stacked, axis=0, dtype=np.float32)
    elif mode == "median":
        stacked = np.stack([v for _, v in items], axis=0)
        out = np.median(stacked, axis=0).astype(np.float32)
    elif mode == "fourier":
        d, h, w = ref_shape
        kz = np.fft.fftfreq(d)[:, None, None] ** 2
        ky = np.fft.fftfreq(h)[None, :, None] ** 2
        kx = np.fft.fftfreq(w)[None, None, :] ** 2
        k_dims = [kz, ky, kx]

        weights: list[np.ndarray] = []
        spectra: list[np.ndarray] = []
        for ax, v in items:
            stack_dim = AXIS_TO_DIM[ax]
            in_plane_energy = sum(k_dims[i] for i in range(3) if i != stack_dim)
            weights.append(np.asarray(in_plane_energy, dtype=np.float64) + 1e-12)
            spectra.append(np.fft.fftn(v))

        weight_sum = sum(weights)
        fused_spec = sum(w_i * s_i for w_i, s_i in zip(weights, spectra, strict=True)) / weight_sum
        # Preserve exact mean DC component across axes.
        fused_spec[0, 0, 0] = np.mean([s[0, 0, 0] for s in spectra])
        out = np.fft.ifftn(fused_spec).real.astype(np.float32)
    else:
        raise ValueError(
            f"Unsupported fusion mode={mode!r}. Expected 'mean', 'median', or 'fourier'."
        )

    return np.clip(out, 0.0, 1.0) if clamp_01 else out


def super_resolve_volume_orthogonal(
    generator: nn.Module,
    low_volume: np.ndarray,
    scale_factor: int = 2,
    axes: Sequence[AxisName] = ("yz", "xz", "xy"),
    fusion: FusionMode = "fourier",
    batch_size: int = 8,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray]:
    """Super-resolve an isotropic 3D LR volume along multiple orthogonal planes and fuse them."""
    per_axis: dict[AxisName, np.ndarray] = {}
    for ax in axes:
        per_axis[ax] = super_resolve_volume_single_axis(
            generator=generator,
            low_volume=low_volume,
            scale_factor=scale_factor,
            axis=ax,
            interpolate_stack_axis=True,
            batch_size=batch_size,
            device=device,
        )

    results: dict[str, np.ndarray] = {str(k): v for k, v in per_axis.items()}
    results["fused"] = fuse_orthogonal_volumes(per_axis, mode=fusion)
    return results
