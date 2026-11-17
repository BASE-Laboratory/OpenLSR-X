"""2D-to-3D and 2.5D-to-3D multi-axis orthogonal XCT super-resolution & epistemic uncertainty."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, zoom
from torch import nn

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


def _build_slice_context_windows(slices: np.ndarray, context_slices: int) -> np.ndarray:
    """Construct ``(N, context_slices, h, w)`` reflect-padded adjacent slice windows."""
    if context_slices < 1 or context_slices % 2 == 0:
        raise ValueError(
            f"context_slices must be a positive odd integer (1, 3, 5, ...), got {context_slices}."
        )
    if context_slices == 1:
        return slices[:, np.newaxis, :, :]
    half = context_slices // 2
    padded = np.pad(slices, ((half, half), (0, 0), (0, 0)), mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(padded, window_shape=context_slices, axis=0)
    return np.moveaxis(win, -1, 1)


@torch.no_grad()
def super_resolve_slice_stack(
    generator: nn.Module,
    slices: np.ndarray | torch.Tensor,
    batch_size: int = 8,
    device: torch.device | str | None = None,
    context_slices: int | None = None,
) -> np.ndarray:
    """Run a 2D or 2.5D generator over a ``(N, h, w)`` stack of slices and return ``(N, H, W)``.

    When ``context_slices`` is ``None``, automatically inspects ``generator.in_channels`` so
    that ``2.5D`` multi-slice generators (``in_channels = 2k + 1``) seamlessly receive
    reflect-padded adjacent slice windows ``[z - k, ..., z + k]``.
    """
    arr = np.asarray(slices, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D slice stack (N, h, w), got shape {arr.shape}.")

    ctx = (
        int(context_slices)
        if context_slices is not None
        else int(getattr(generator, "in_channels", 1))
    )
    windows = _build_slice_context_windows(arr, ctx)

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
    """Super-resolve a 3D XCT volume ``(D, H, W)`` using a 2D/2.5D generator along ``axis``."""
    vol = np.asarray(low_volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}.")

    ordered = _move_slice_axis_first(vol, axis)  # (N_axis, h, w)
    sf = float(scale_factor)
    if interpolate_stack_axis and sf > 1.0:
        target_n = int(round(ordered.shape[0] * sf))
        zoom_n = target_n / float(ordered.shape[0])
        ordered = zoom(
            ordered,
            zoom=(zoom_n, 1, 1),
            order=3,
            mode="reflect",
        ).astype(np.float32, copy=False)
        ordered = np.clip(ordered, 0.0, 1.0)

    sr_ordered = super_resolve_slice_stack(
        generator=generator,
        slices=ordered,
        batch_size=batch_size,
        device=device,
        context_slices=context_slices,
    )
    return _move_slice_axis_back(sr_ordered, axis)


def compute_orthogonal_uncertainty(
    volumes: Mapping[AxisName, np.ndarray],
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Compute zero-overhead voxel-wise epistemic uncertainty across orthogonal SR reconstructions.

    Because each voxel ``x`` is reconstructed from perpendicular high-resolution sweep planes,
    inter-axis disagreement ``sigma_ortho(x) = std(V_yz, V_xz, V_xy)`` concentrates selectively
    at sub-resolution pore throats and multi-phase grain boundaries where out-of-plane blur or
    hallucination occurs.

    Returns:
        Dictionary with keys ``'mean'``, ``'std'``, ``'range'``, and ``'cv'`` (coefficient of
        variation), each of shape ``(D, H, W)`` in ``float32``.
    """
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
        w_ax = np.asarray(in_plane_energy, dtype=np.float64) + 1e-12
        if psf_cutoff is not None and psf_cutoff > 0.0:
            # Attenuate interpolated stacking-axis frequencies above optical OTF roll-off
            k_stack_sq = np.asarray(k_dims[stack_dim], dtype=np.float64)
            w_ax = w_ax * np.exp(-k_stack_sq / (2.0 * float(psf_cutoff) ** 2))
        weights.append(w_ax)
        spectra.append(np.fft.fftn(v))

    weight_sum = sum(weights)
    fused_spec = sum(w_i * s_i for w_i, s_i in zip(weights, spectra, strict=True)) / weight_sum
    fused_spec[0, 0, 0] = np.mean([s[0, 0, 0] for s in spectra])
    return np.fft.ifftn(fused_spec).real.astype(np.float32)


def _confidence_consensus_fuse(stacked: np.ndarray) -> np.ndarray:
    """Inverse-disagreement spatial consensus suppressing isolated single-axis blur/artifacts."""
    med = np.median(stacked, axis=0)
    weights = np.empty_like(stacked, dtype=np.float32)
    for idx in range(stacked.shape[0]):
        dev = gaussian_filter(np.abs(stacked[idx] - med), sigma=1.0, mode="reflect")
        weights[idx] = 1.0 / ((dev + 1e-4) ** 2)
    w_sum = np.sum(weights, axis=0, dtype=np.float32)
    return np.asarray(np.sum(weights * stacked, axis=0, dtype=np.float32) / w_sum, dtype=np.float32)


def fuse_orthogonal_volumes(
    volumes: Mapping[AxisName, np.ndarray],
    mode: FusionMode = "mean",
    clamp_01: bool = True,
    psf_cutoff: float | None = None,
) -> np.ndarray:
    """Fuse aligned single-axis 3D super-resolved volumes into an isotropic 3D reconstruction.

    Supported fusion modes:
    - ``'mean'``: Equal-weight voxel-wise average across provided axes.
    - ``'median'``: Voxel-wise median (robust against isolated slice artifacts).
    - ``'fourier'``: 3D Fourier-space directional fusion weighting each single-axis
      reconstruction by its in-plane spatial frequencies (and optional OTF ``psf_cutoff``).
    - ``'confidence'``: Inverse-disagreement local consensus weighting that suppresses the
      out-of-plane blurred axis when the two perpendicular axes agree on a sharp throat.
    - ``'confidence_fourier'``: Hybrid spectral-consensus fusion applying directional 3D
      Fourier synthesis in high-agreement bulk regions while transitioning smoothly to
      outlier-resistant consensus where inter-axis epistemic uncertainty ``sigma_ortho`` spikes.
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
        out = _fourier_directional_fuse(items, ref_shape, psf_cutoff=psf_cutoff)
    elif mode == "confidence":
        stacked = np.stack([v for _, v in items], axis=0)
        out = _confidence_consensus_fuse(stacked)
    elif mode == "confidence_fourier":
        stacked = np.stack([v for _, v in items], axis=0)
        v_fourier = _fourier_directional_fuse(items, ref_shape, psf_cutoff=psf_cutoff)
        v_conf = _confidence_consensus_fuse(stacked)
        sigma = np.std(stacked, axis=0, ddof=1, dtype=np.float32)
        alpha = np.clip(sigma / 0.08, 0.0, 1.0)
        out = ((1.0 - alpha) * v_fourier + alpha * v_conf).astype(np.float32)
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
