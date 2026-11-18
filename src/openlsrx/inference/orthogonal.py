"""2D-to-3D and 2.5D-to-3D multi-axis orthogonal XCT super-resolution & view-disagreement uncertainty."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn

from ..data.transforms import build_strided_context_windows, zoom_cell_centered

AxisName = Literal["yz", "xz", "xy"]
FusionMode = Literal["mean", "median", "fourier", "confidence", "confidence_fourier"]

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


def _build_slice_context_windows(
    slices: np.ndarray,
    context_slices: int,
    context_stride: int = 1,
) -> np.ndarray:
    """Construct ``(N, context_slices, h, w)`` adjacent slice windows at ``context_stride``."""
    return build_strided_context_windows(
        slices, context_slices=context_slices, context_stride=context_stride
    )


@torch.no_grad()
def super_resolve_slice_stack(
    generator: nn.Module,
    slices: np.ndarray | torch.Tensor,
    batch_size: int = 8,
    device: torch.device | str | None = None,
    context_slices: int | None = None,
    context_stride: int = 1,
) -> np.ndarray:
    """Run a 2D or 2.5D generator over a ``(N, h, w)`` stack of slices and return ``(N, H, W)``."""
    arr = np.asarray(slices, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D slice stack (N, h, w), got shape {arr.shape}.")

    ctx = (
        int(context_slices)
        if context_slices is not None
        else int(getattr(generator, "in_channels", 1))
    )
    windows = _build_slice_context_windows(arr, ctx, context_stride=context_stride)

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
    step = max(1, int(batch_size))
    try:
        for start in range(0, windows.shape[0], step):
            batch_np = np.ascontiguousarray(windows[start : start + step], dtype=np.float32)
            batch = torch.from_numpy(batch_np).to(dev)
            pred = generator(batch).squeeze(1).to(torch.float32).cpu().numpy()
            outputs.append(pred)
    finally:
        if was_training:
            generator.train()

    return np.concatenate(outputs, axis=0)


def super_resolve_volume_single_axis(
    generator: nn.Module,
    low_volume: np.ndarray,
    scale_factor: int | float = 2,
    axis: AxisName = "yz",
    interpolate_stack_axis: bool = True,
    batch_size: int = 8,
    device: torch.device | str | None = None,
    context_slices: int | None = None,
) -> np.ndarray:
    """Super-resolve a 3D XCT volume ``(D, H, W)`` using a 2D/2.5D generator along ``axis``.

    Uses ``zoom_cell_centered`` (``grid_mode=True``) along the stacking axis so half-voxel
    cell centers remain strictly aligned with ``PixelShuffle`` across all 3 orthogonal sweeps,
    and samples 2.5D context neighbors at native-LR spacing (``context_stride = round(zoom_n)``).
    """
    vol = np.asarray(low_volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}.")

    ordered = _move_slice_axis_first(vol, axis)  # (N_axis, h, w)
    sf = float(scale_factor)
    ctx_stride = 1
    if interpolate_stack_axis and sf > 1.0:
        target_n = max(1, int(round(ordered.shape[0] * sf)))
        zoom_n = target_n / float(ordered.shape[0])
        ordered = zoom_cell_centered(ordered, zoom_factors=(zoom_n, 1.0, 1.0), order=3)
        ordered = np.clip(ordered, 0.0, 1.0)
        ctx_stride = max(1, int(round(zoom_n)))

    sr_ordered = super_resolve_slice_stack(
        generator=generator,
        slices=ordered,
        batch_size=batch_size,
        device=device,
        context_slices=context_slices,
        context_stride=ctx_stride,
    )
    return _move_slice_axis_back(sr_ordered, axis)


def compute_orthogonal_uncertainty(
    volumes: Mapping[AxisName, np.ndarray],
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Compute zero-overhead voxel-wise orthogonal view-disagreement uncertainty (``n=3`` sweeps)."""
    if not volumes:
        raise ValueError("At least one orthogonal volume is required.")
    items = [(ax, np.asarray(v, dtype=np.float32)) for ax, v in volumes.items()]
    ref_shape = items[0][1].shape
    for ax, v in items:
        if v.ndim != 3 or v.shape != ref_shape:
            raise ValueError(
                f"All orthogonal volumes must be 3D with identical shape {ref_shape}; "
                f"axis {ax!r} has shape {v.shape}."
            )

    if len(items) == 1:
        mean_v = items[0][1].copy()
        zeros = np.zeros_like(mean_v, dtype=np.float32)
        return {"mean": mean_v, "std": zeros, "range": zeros.copy(), "cv": zeros.copy()}

    stacked = np.stack([v for _, v in items], axis=0)
    mean_v = np.mean(stacked, axis=0, dtype=np.float32)
    range_v = np.ptp(stacked, axis=0).astype(np.float32)
    std_v = np.std(stacked, axis=0, ddof=1, dtype=np.float32)
    std_v = np.where(range_v == 0.0, np.float32(0.0), std_v)
    cv_v = (std_v / (np.abs(mean_v) + float(eps))).astype(np.float32)
    return {"mean": mean_v, "std": std_v, "range": range_v, "cv": cv_v}


def _fourier_directional_fuse(
    items: list[tuple[AxisName, np.ndarray]],
    ref_shape: tuple[int, ...],
    psf_cutoff: float | None = None,
) -> np.ndarray:
    """Streaming ``float32`` / ``complex64`` ``rfftn`` directional synthesis (6x lower RAM, NaN-free)."""
    d, h, w = ref_shape
    kz = (np.fft.fftfreq(d).astype(np.float32)[:, None, None]) ** 2
    ky = (np.fft.fftfreq(h).astype(np.float32)[None, :, None]) ** 2
    kx = (np.fft.rfftfreq(w).astype(np.float32)[None, None, :]) ** 2
    k_dims = [kz, ky, kx]

    rfft_shape = (d, h, w // 2 + 1)
    acc_spec = np.zeros(rfft_shape, dtype=np.complex64)
    weight_sum = np.zeros(rfft_shape, dtype=np.float32)
    dc_values: list[complex] = []

    if psf_cutoff is not None and psf_cutoff > 0.0:
        use_otf = True
        inv_two_sigma_sq = np.float32(1.0 / max(2.0 * (psf_cutoff**2), 1e-12))
    else:
        use_otf = False
        inv_two_sigma_sq = np.float32(0.0)

    for ax, v in items:
        v_f32 = np.ascontiguousarray(v, dtype=np.float32)
        stack_dim = AXIS_TO_DIM[ax]
        in_plane_energy = sum(k_dims[i] for i in range(3) if i != stack_dim)
        if use_otf:
            otf_stack = np.exp(-k_dims[stack_dim] * inv_two_sigma_sq).astype(np.float32, copy=False)
            # Positive floor added AFTER OTF roll-off guarantees w_ax >= 1e-6 > 0 everywhere (never 0/0 = NaN)
            w_ax = (in_plane_energy * otf_stack + np.float32(1e-6)).astype(np.float32, copy=False)
        else:
            w_ax = (in_plane_energy + np.float32(1e-6)).astype(np.float32, copy=False)

        spec = np.fft.rfftn(v_f32).astype(np.complex64, copy=False)
        dc_values.append(complex(spec[0, 0, 0]))
        acc_spec += w_ax * spec
        weight_sum += w_ax

    fused_spec = acc_spec / np.maximum(weight_sum, np.float32(1e-12))
    fused_spec[0, 0, 0] = np.complex64(sum(dc_values) / len(dc_values))
    return np.fft.irfftn(fused_spec, s=(d, h, w), axes=(0, 1, 2)).astype(np.float32, copy=False)


def _through_plane_roughness(vol: np.ndarray, stack_dim: int) -> np.ndarray:
    """Local stacking-axis discontinuity magnitude used to weight 2-axis consensus."""
    if vol.shape[stack_dim] < 2:
        return np.zeros_like(vol, dtype=np.float32)
    diff = np.abs(np.diff(vol, axis=stack_dim))
    pad_spec = [(0, 0), (0, 0), (0, 0)]
    pad_spec[stack_dim] = (0, 1)
    padded = np.pad(diff, pad_spec, mode="edge")
    return gaussian_filter(padded, sigma=1.0, mode="reflect").astype(np.float32, copy=False)


def _confidence_consensus_fuse(items: list[tuple[AxisName, np.ndarray]]) -> np.ndarray:
    """Inverse-disagreement spatial consensus suppressing isolated single-axis blur/artifacts."""
    stacked = np.stack([v for _, v in items], axis=0)
    if stacked.shape[0] == 2:
        # With only 2 axes, median == mean identically; weight inversely by local stacking roughness
        r0 = _through_plane_roughness(items[0][1], AXIS_TO_DIM[items[0][0]])
        r1 = _through_plane_roughness(items[1][1], AXIS_TO_DIM[items[1][0]])
        w0 = 1.0 / ((r0 + np.float32(1e-3)) ** 2)
        w1 = 1.0 / ((r1 + np.float32(1e-3)) ** 2)
        return np.asarray((w0 * stacked[0] + w1 * stacked[1]) / (w0 + w1), dtype=np.float32)

    med = np.median(stacked, axis=0).astype(np.float32, copy=False)
    weights = np.empty_like(stacked, dtype=np.float32)
    for idx in range(stacked.shape[0]):
        dev = gaussian_filter(np.abs(stacked[idx] - med), sigma=1.0, mode="reflect")
        weights[idx] = 1.0 / ((dev + np.float32(1e-4)) ** 2)
    w_sum = np.sum(weights, axis=0, dtype=np.float32)
    return np.asarray(np.sum(weights * stacked, axis=0, dtype=np.float32) / w_sum, dtype=np.float32)


def _outlier_suppressed_fourier_fuse(
    items: list[tuple[AxisName, np.ndarray]],
    ref_shape: tuple[int, ...],
    psf_cutoff: float | None = None,
) -> np.ndarray:
    """Outlier-clamped directional Fourier synthesis (suppresses single-axis defects without stacking noise).

    Instead of blending spatial consensus directly into high-contrast pore walls (which re-injects
    slice-stacking jitter wherever ``sigma_ortho`` rises), each single-axis sweep is Winsorized
    toward the perpendicular-axes consensus wherever the remaining axes agree tightly, and the
    cleaned sweeps are then synthesized via directional 3D Fourier fusion.
    """
    v_conf = _confidence_consensus_fuse(items)
    if len(items) < 3:
        v_four = _fourier_directional_fuse(items, ref_shape, psf_cutoff=psf_cutoff)
        return np.asarray(0.5 * (v_conf + v_four), dtype=np.float32)

    cleaned_items: list[tuple[AxisName, np.ndarray]] = []
    n_axes = len(items)
    for idx, (ax, v_ax) in enumerate(items):
        others = [items[j][1] for j in range(n_axes) if j != idx]
        other_spread = np.ptp(np.stack(others, axis=0), axis=0).astype(np.float32)
        # Tolerance scales with perpendicular-axes spread; when perpendicular axes agree (spread ~ 0),
        # isolated out-of-plane blur or corruption on v_ax is pulled cleanly to v_conf.
        tol = 1.25 * other_spread + np.float32(0.015)
        v_clean = v_conf + np.clip(v_ax - v_conf, -tol, tol)
        cleaned_items.append((ax, v_clean.astype(np.float32, copy=False)))

    return _fourier_directional_fuse(cleaned_items, ref_shape, psf_cutoff=psf_cutoff)


def fuse_orthogonal_volumes(
    volumes: Mapping[AxisName, np.ndarray],
    mode: FusionMode = "mean",
    clamp_01: bool = True,
    psf_cutoff: float | None = None,
) -> np.ndarray:
    """Fuse aligned single-axis 3D super-resolved volumes into an isotropic 3D reconstruction."""
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
        out = _fourier_directional_fuse(items, ref_shape, psf_cutoff=psf_cutoff)
    elif mode == "confidence":
        out = _confidence_consensus_fuse(items)
    elif mode == "confidence_fourier":
        out = _outlier_suppressed_fourier_fuse(items, ref_shape, psf_cutoff=psf_cutoff)
    else:
        raise ValueError(
            f"Unsupported fusion mode={mode!r}. Expected 'mean', 'median', 'fourier', "
            "'confidence', or 'confidence_fourier'."
        )

    return np.clip(out, 0.0, 1.0) if clamp_01 else out


def super_resolve_volume_orthogonal(
    generator: nn.Module,
    low_volume: np.ndarray,
    scale_factor: int | float = 2,
    axes: Sequence[AxisName] = ("yz", "xz", "xy"),
    fusion: FusionMode = "fourier",
    batch_size: int = 8,
    device: torch.device | str | None = None,
    context_slices: int | None = None,
    return_uncertainty: bool = False,
    psf_cutoff: float | None = None,
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
            context_slices=context_slices,
        )

    results: dict[str, np.ndarray] = {str(k): v for k, v in per_axis.items()}
    results["fused"] = fuse_orthogonal_volumes(per_axis, mode=fusion, psf_cutoff=psf_cutoff)
    if return_uncertainty:
        unc = compute_orthogonal_uncertainty(per_axis)
        results["uncertainty_std"] = unc["std"]
        results["uncertainty_range"] = unc["range"]
    return results
