"""Vectorized XCT normalization, alignment, cell-centered resampling, and leakage-free 3D splitting."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, zoom
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

NormMode = Literal["minmax", "percentile", "fixed", "none"]
CrossScanCalibrationMode = Literal["none", "affine", "quantile"]


@dataclass(frozen=True)
class VolumeAlignment:
    """Configurable spatial ROI crop and in-plane rotation for raw 3D XCT volumes."""

    z_range: tuple[int, int] | None = None
    y_range: tuple[int, int] | None = None
    x_range: tuple[int, int] | None = None
    rotation_deg: float = 0.0


def sanitize_xct_volume(
    volume: np.ndarray,
    nan_fill: float | None = None,
) -> np.ndarray:
    """Replace non-finite (``NaN``, ``±Inf``) Savu/TomoPy reconstruction artifacts safely.

    Synchrotron reconstruction pipelines (e.g., cylindrical FOV masks, Paganin phase retrieval,
    or dead detector stripes) frequently emit ``NaN`` or ``±Inf`` voxels that would otherwise
    corrupt global percentiles, 3D convolutions, and FFTs across 100% of the volume.
    """
    raw = np.asarray(volume)
    orig_kind = raw.dtype.kind
    orig_itemsize = raw.dtype.itemsize
    arr = np.asarray(raw, dtype=np.float32)
    if orig_kind == "u":
        max_int = float((1 << (8 * orig_itemsize)) - 1)
        if max_int > 0.0 and float(np.nanmax(arr)) > 1.0:
            arr = arr / max_int

    finite_mask = np.isfinite(arr)
    if bool(finite_mask.all()):
        return arr

    if not bool(finite_mask.any()):
        fill = 0.0 if nan_fill is None else float(nan_fill)
        return np.full_like(arr, fill, dtype=np.float32)

    finite_vals = arr[finite_mask]
    f_min = float(finite_vals.min())
    f_max = float(finite_vals.max())
    default_fill = f_min if nan_fill is None else float(nan_fill)
    out = arr.copy()
    out[np.isnan(out)] = default_fill
    out[np.isneginf(out)] = f_min
    out[np.isposinf(out)] = f_max
    return out


def build_circular_fov_mask(shape: tuple[int, ...], radius_fraction: float = 0.98) -> np.ndarray:
    """Build an inscribed circular cylindrical XCT field-of-view mask along ``(H, W)``."""
    if len(shape) == 2:
        h, w = shape
        yy, xx = np.ogrid[:h, :w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        ry = max(h / 2.0 * float(radius_fraction), 1.0)
        rx = max(w / 2.0 * float(radius_fraction), 1.0)
        return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    if len(shape) == 3:
        _, h, w = shape
        mask_2d = build_circular_fov_mask((h, w), radius_fraction=radius_fraction)
        return np.broadcast_to(mask_2d[np.newaxis, :, :], shape).copy()
    raise ValueError(f"Expected 2D or 3D shape for circular FOV mask, got {shape}.")


def compute_normalization_bounds(
    volume: np.ndarray,
    mode: NormMode = "minmax",
    vmin: float | None = None,
    vmax: float | None = None,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute ``(lo, hi)`` intensity bounds on finite (and optionally masked) reference voxels."""
    arr = sanitize_xct_volume(volume)
    vals = arr[np.asarray(mask, dtype=bool)] if mask is not None else arr.ravel()
    if vals.size == 0:
        vals = arr.ravel()

    if mode == "none":
        return 0.0, 1.0
    if mode == "minmax":
        return float(vals.min()), float(vals.max())
    if mode == "percentile":
        lo, hi = (float(v) for v in np.percentile(vals, [lower_pct, upper_pct]))
        return lo, hi
    if mode == "fixed":
        if vmin is None or vmax is None:
            raise ValueError("vmin and vmax must be provided when mode='fixed'.")
        return float(vmin), float(vmax)
    raise ValueError(f"Unsupported normalization mode={mode!r}.")


def normalize_volume(
    volume: np.ndarray,
    mode: NormMode = "minmax",
    vmin: float | None = None,
    vmax: float | None = None,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
    eps: float = 1e-8,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized, ``NaN``-safe normalization of a 2D/3D XCT array into ``[0, 1]`` float32."""
    arr = sanitize_xct_volume(volume)
    if mode == "none":
        return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

    lo, hi = compute_normalization_bounds(
        arr,
        mode=mode,
        vmin=vmin,
        vmax=vmax,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
        mask=mask,
    )
    denom = hi - lo
    if abs(denom) < eps:
        return np.zeros_like(arr, dtype=np.float32)
    normed = (arr - lo) / denom
    out = np.clip(normed, 0.0, 1.0).astype(np.float32, copy=False)
    if mask is not None:
        out = np.where(np.asarray(mask, dtype=bool), out, 0.0).astype(np.float32, copy=False)
    return out


def zoom_cell_centered(
    arr: np.ndarray,
    zoom_factors: float | tuple[float, ...],
    order: int = 3,
) -> np.ndarray:
    """Cell-centered spline resampling matching ``PixelShuffle`` & ``F.interpolate(align_corners=False)``.

    Standard ``scipy.ndimage.zoom(..., grid_mode=False)`` aligns corner centers ``0 -> 0`` and
    ``N - 1 -> s*N - 1``, shifting interior coordinates by up to ``0.5 * (s - 1)`` HR voxels.
    Setting ``grid_mode=True`` preserves exact half-voxel cell centers across all magnifications.
    """
    return zoom(
        np.asarray(arr, dtype=np.float32),
        zoom=zoom_factors,
        order=order,
        mode="nearest",
        grid_mode=True,
    ).astype(np.float32, copy=False)


def rotate_slice(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2D slice around its center using reflection border padding."""
    if abs(angle_deg) < 1e-6:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_alignment(volume: np.ndarray, alignment: VolumeAlignment | None) -> np.ndarray:
    """Apply optional 3D bounding-box crop and slice-wise rotation to a 3D volume."""
    arr = sanitize_xct_volume(volume)
    if alignment is None:
        return arr

    z_sl = slice(*alignment.z_range) if alignment.z_range is not None else slice(None)
    y_sl = slice(*alignment.y_range) if alignment.y_range is not None else slice(None)
    x_sl = slice(*alignment.x_range) if alignment.x_range is not None else slice(None)
    cropped = np.ascontiguousarray(arr[z_sl, y_sl, x_sl], dtype=np.float32)

    if abs(alignment.rotation_deg) >= 1e-6:
        rotated = np.empty_like(cropped)
        for i in range(cropped.shape[0]):
            rotated[i] = rotate_slice(cropped[i], alignment.rotation_deg)
        cropped = rotated
    return cropped


def apply_contrast_shift(volume: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """Vectorized interior contrast shift preserving zero background."""
    if abs(shift) < 1e-8:
        return np.asarray(volume, dtype=np.float32)
    out = np.array(volume, dtype=np.float32, copy=True)
    mask = (out > 0.0) & (out < (1.0 - shift - 0.05))
    out[mask] += float(shift)
    return np.clip(out, 0.0, 1.0)


def degrade_xct_volume(
    hr_volume: np.ndarray,
    scale_factor: float = 2,
    psf_sigma: float = 0.8,
    noise_std: float = 0.02,
    poisson_peak: float | None = None,
    contrast_shift: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate realistic isotropic 3D low-resolution XCT acquisition from an HR volume.

    Supports both integer and fractional optical magnification ratios (e.g., ``2x``, ``2.5x``, ``4x``)
    via exact cell-centered box-area integration.
    """
    if scale_factor <= 0.0:
        raise ValueError(f"scale_factor must be > 0, got {scale_factor}.")
    hr = sanitize_xct_volume(hr_volume)
    if hr.ndim != 3:
        raise ValueError(f"Expected 3D HR volume (D, H, W), got shape {hr.shape}.")

    s_int = int(round(float(scale_factor)))
    is_int_scale = abs(float(scale_factor) - float(s_int)) < 1e-6
    if is_int_scale:
        for dim in hr.shape:
            if dim % s_int != 0:
                raise ValueError(
                    f"HR volume shape {hr.shape} must be divisible by scale_factor={s_int}."
                )

    rng = np.random.default_rng(seed)
    work = apply_contrast_shift(hr, shift=contrast_shift)
    if psf_sigma > 0.0:
        work = gaussian_filter(work, sigma=float(psf_sigma), mode="reflect")

    if is_int_scale:
        d, h, w = work.shape
        binned = work.reshape(d // s_int, s_int, h // s_int, s_int, w // s_int, s_int).mean(
            axis=(1, 3, 5), dtype=np.float32
        )
    else:
        target_lr = tuple(max(1, int(round(dim / float(scale_factor)))) for dim in work.shape)
        t_work = torch.from_numpy(np.ascontiguousarray(work)).unsqueeze(0).unsqueeze(0)
        binned = F.interpolate(t_work, size=target_lr, mode="area").squeeze(0).squeeze(0).numpy()

    if poisson_peak is not None and poisson_peak > 0.0:
        counts = rng.poisson(np.clip(binned, 0.0, 1.0) * float(poisson_peak))
        binned = counts.astype(np.float32) / float(poisson_peak)

    if noise_std > 0.0:
        binned = binned + rng.normal(0.0, float(noise_std), size=binned.shape).astype(np.float32)

    return np.clip(binned, 0.0, 1.0).astype(np.float32, copy=False)


def fit_cross_scan_calibration(
    lr_train: np.ndarray,
    hr_train: np.ndarray,
    scale_factor: float = 2,
    mode: CrossScanCalibrationMode = "none",
    psf_sigma: float = 0.0,
    n_quantiles: int = 257,
    mask_lr: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit a strictly ``train``-slab-only radiometric transfer mapping ``LR -> coarse(HR)``.

    Independent physical dual-scan X-ray micro-CT reconstructions (e.g., full-core ``18 um`` vs.
    interior ROI ``6 um`` scans) often differ in polychromatic beam-hardening curvature and
    greyscale windowing. Fitting a monotonic 1D transfer strictly on the ``train`` slab aligns
    ``LR`` attenuation with PSF-downsampled ``HR`` without leaking ``val`` or ``test`` statistics.
    """
    if mode == "none":
        return {"mode": "none"}

    lr_arr = sanitize_xct_volume(lr_train)
    hr_arr = sanitize_xct_volume(hr_train)
    hr_coarse = degrade_xct_volume(
        hr_arr,
        scale_factor=scale_factor,
        psf_sigma=max(0.0, float(psf_sigma)),
        noise_std=0.0,
        seed=0,
    )
    common = tuple(min(int(a), int(b)) for a, b in zip(lr_arr.shape, hr_coarse.shape, strict=True))
    lr_sub = lr_arr[: common[0], : common[1], : common[2]]
    hr_sub = hr_coarse[: common[0], : common[1], : common[2]]

    if mask_lr is not None:
        m_sub = np.asarray(mask_lr, dtype=bool)[: common[0], : common[1], : common[2]]
        x = lr_sub[m_sub].astype(np.float64, copy=False)
        y = hr_sub[m_sub].astype(np.float64, copy=False)
    else:
        x = lr_sub.ravel().astype(np.float64, copy=False)
        y = hr_sub.ravel().astype(np.float64, copy=False)

    if x.size == 0:
        return {"mode": "none"}

    if mode == "affine":
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        x_var = float(np.mean((x - x_mean) ** 2))
        if x_var < 1e-12:
            slope, intercept = 1.0, y_mean - x_mean
        else:
            cov_xy = float(np.mean((x - x_mean) * (y - y_mean)))
            slope = max(0.0, cov_xy / x_var)
            intercept = y_mean - slope * x_mean
        return {"mode": "affine", "slope": float(slope), "intercept": float(intercept)}

    if mode == "quantile":
        qs = np.linspace(0.0, 100.0, max(17, int(n_quantiles)))
        q_lr = np.percentile(x, qs)
        q_hr = np.percentile(y, qs)
        q_lr_unique, unique_idx = np.unique(q_lr, return_index=True)
        q_hr_unique = np.maximum.accumulate(q_hr[unique_idx])
        if q_lr_unique.size < 2:
            shift = float(y.mean() - x.mean())
            return {"mode": "affine", "slope": 1.0, "intercept": shift}
        span_x = float(q_lr[-2] - q_lr[1]) if q_lr.size >= 4 else float(q_lr[-1] - q_lr[0])
        span_y = float(q_hr[-2] - q_hr[1]) if q_hr.size >= 4 else float(q_hr[-1] - q_hr[0])
        tail_slope = max(0.0, span_y / span_x) if span_x > 1e-8 else 1.0
        return {
            "mode": "quantile",
            "xp": q_lr_unique.astype(np.float64).tolist(),
            "fp": q_hr_unique.astype(np.float64).tolist(),
            "tail_slope": float(tail_slope),
        }

    raise ValueError(
        f"Unsupported cross_scan_calibration mode={mode!r}. Expected 'none', 'affine', or 'quantile'."
    )


def apply_cross_scan_calibration(
    lr_volume: np.ndarray,
    params: dict[str, object] | None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a frozen ``train``-fitted cross-scan calibration map to a normalized LR volume."""
    arr = sanitize_xct_volume(lr_volume)
    if params is None or params.get("mode", "none") == "none":
        out = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    elif params.get("mode") == "affine":
        slope = float(params["slope"])  # type: ignore[arg-type]
        intercept = float(params["intercept"])  # type: ignore[arg-type]
        out = np.clip(arr * np.float32(slope) + np.float32(intercept), 0.0, 1.0).astype(
            np.float32, copy=False
        )
    elif params.get("mode") == "quantile":
        xp = np.asarray(params["xp"], dtype=np.float64)
        fp = np.asarray(params["fp"], dtype=np.float64)
        tail_slope = float(params.get("tail_slope", 1.0))  # type: ignore[arg-type]
        flat = arr.ravel().astype(np.float64, copy=False)
        mapped = np.interp(flat, xp, fp)
        below = flat < xp[0]
        above = flat > xp[-1]
        if np.any(below):
            mapped[below] = fp[0] + tail_slope * (flat[below] - xp[0])
        if np.any(above):
            mapped[above] = fp[-1] + tail_slope * (flat[above] - xp[-1])
        out = np.clip(mapped.reshape(arr.shape), 0.0, 1.0).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown cross-scan calibration params mode={params.get('mode')!r}.")

    if mask is not None:
        out = np.where(np.asarray(mask, dtype=bool), out, 0.0).astype(np.float32, copy=False)
    return out


def downsample_orthogonal_slabs(
    hr_volume: np.ndarray,
    scale_factor: float = 2,
    psf_sigma: float = 0.6,
    noise_std: float = 0.02,
    variable_slice_noise: bool = False,
    contrast_shift: float = 0.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create paired orthogonal LR slice stacks ``yz``, ``xz``, and ``xy`` from a 3D HR cube."""
    hr = sanitize_xct_volume(hr_volume)
    if hr.ndim != 3:
        raise ValueError(f"Expected 3D HR volume, got shape {hr.shape}.")

    rng = np.random.default_rng(seed)
    work = apply_contrast_shift(hr, shift=contrast_shift)
    if psf_sigma > 0.0:
        work = gaussian_filter(work, sigma=float(psf_sigma), mode="reflect")

    slabs: dict[str, np.ndarray] = {}
    axis_map = {"yz": 0, "xz": 1, "xy": 2}
    for name, dim in axis_map.items():
        ordered = np.moveaxis(work, dim, 0)
        n_slices, h_in, w_in = ordered.shape
        h_out = max(1, int(round(h_in / float(scale_factor))))
        w_out = max(1, int(round(w_in / float(scale_factor))))
        down = np.empty((n_slices, h_out, w_out), dtype=np.float32)
        for i in range(n_slices):
            sl = cv2.resize(
                ordered[i],
                dsize=(w_out, h_out),
                interpolation=cv2.INTER_LANCZOS4,
            )
            sigma = (
                float(noise_std) * float(np.sqrt(rng.uniform(0.2, 1.0)))
                if variable_slice_noise
                else float(noise_std)
            )
            if sigma > 0.0:
                sl = sl + rng.normal(0.0, sigma, size=sl.shape).astype(np.float32)
            down[i] = sl
        slabs[name] = np.clip(down, 0.0, 1.0)
    return slabs


def split_volume_spatially(
    hr_volume: np.ndarray,
    lr_volume: np.ndarray | None = None,
    scale_factor: float = 2,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    split_axis: int = 0,
    guard_slices_lr: int = 0,
) -> dict[str, tuple[np.ndarray | None, np.ndarray]]:
    """Partition a 3D HR (and optional 3D LR) volume into disjoint spatial sub-volumes."""
    hr = sanitize_xct_volume(hr_volume)
    if hr.ndim != 3:
        raise ValueError(f"Expected 3D HR volume, got shape {hr.shape}.")
    s_float = float(scale_factor)
    total_hr = hr.shape[split_axis]
    total_lr = (
        int(lr_volume.shape[split_axis])
        if lr_volume is not None
        else int(round(total_hr / s_float))
    )
    if total_lr < 3:
        raise ValueError(
            f"Volume dimension {total_hr} along axis {split_axis} is too small for 3-way split "
            f"at scale_factor={scale_factor}."
        )

    usable_lr = total_lr - 2 * guard_slices_lr
    if usable_lr < 3:
        raise ValueError("guard_slices_lr leaves fewer than 3 LR slices across splits.")

    n_train_lr = max(1, int(round(usable_lr * train_frac)))
    n_val_lr = max(1, int(round(usable_lr * val_frac)))
    if n_train_lr + n_val_lr >= usable_lr:
        n_train_lr = max(1, usable_lr - 2)
        n_val_lr = 1
    n_test_lr = usable_lr - n_train_lr - n_val_lr

    t0_lr, t1_lr = 0, n_train_lr
    v0_lr = t1_lr + guard_slices_lr
    v1_lr = v0_lr + n_val_lr
    s0_lr = v1_lr + guard_slices_lr
    s1_lr = s0_lr + n_test_lr

    def _slice_axis(arr: np.ndarray, start: int, stop: int) -> np.ndarray:
        idx = [slice(None), slice(None), slice(None)]
        idx[split_axis] = slice(start, stop)
        return np.ascontiguousarray(arr[tuple(idx)], dtype=np.float32)

    splits: dict[str, tuple[np.ndarray | None, np.ndarray]] = {}
    lr_clean = sanitize_xct_volume(lr_volume) if lr_volume is not None else None
    for name, (a_lr, b_lr) in (
        ("train", (t0_lr, t1_lr)),
        ("val", (v0_lr, v1_lr)),
        ("test", (s0_lr, s1_lr)),
    ):
        n_len_hr = max(1, int(round((b_lr - a_lr) * s_float)))
        a_hr = int(round(a_lr * s_float))
        b_hr = min(total_hr, a_hr + n_len_hr)
        a_hr = max(0, b_hr - n_len_hr)
        hr_sub = _slice_axis(hr, a_hr, b_hr)
        lr_sub = _slice_axis(lr_clean, a_lr, b_lr) if lr_clean is not None else None
        splits[name] = (lr_sub, hr_sub)
    return splits


def _safe_pad_spatial(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Symmetric half-sample pad trailing ``(H, W)`` dimensions up to ``(target_h, target_w)``.

    Half-sample ``mode="symmetric"`` reflects across the outer voxel face ``N - 0.5`` rather than
    voxel center ``N - 1``, preserving exact ``s``-voxel cell-centered alignment between padded
    LR and HR patches for any integer super-resolution factor ``s``.
    """
    cur_h, cur_w = int(arr.shape[-2]), int(arr.shape[-1])
    pad_h = max(0, target_h - cur_h)
    pad_w = max(0, target_w - cur_w)
    if pad_h == 0 and pad_w == 0:
        return arr
    pad_spec = [(0, 0)] * (arr.ndim - 2) + [
        (0, pad_h),
        (0, pad_w),
    ]
    return np.pad(arr, pad_spec, mode="symmetric").astype(np.float32, copy=False)


def build_strided_context_windows(
    slices: np.ndarray,
    context_slices: int,
    context_stride: int = 1,
) -> np.ndarray:
    """Build ``(N, context_slices, h, w)`` windows separated by ``context_stride`` slices.

    When an LR slice stack has been zoomed along the stacking axis by ``s = scale_factor``,
    setting ``context_stride = round(s)`` samples neighbors separated by one true physical LR
    voxel ``+-Delta z_LR`` rather than redundant cubic sub-slice blends.
    """
    if context_slices < 1 or context_slices % 2 == 0:
        raise ValueError(
            f"context_slices must be a positive odd integer (1, 3, 5, ...), got {context_slices}."
        )
    if context_slices == 1:
        return slices[:, np.newaxis, :, :]
    stride = max(1, int(context_stride))
    half_span = (context_slices // 2) * stride
    pad_mode: Literal["reflect", "edge"] = "reflect" if slices.shape[0] > half_span else "edge"
    padded = np.pad(slices, ((half_span, half_span), (0, 0), (0, 0)), mode=pad_mode)
    n = int(slices.shape[0])
    channels = [padded[m * stride : m * stride + n] for m in range(context_slices)]
    return np.ascontiguousarray(np.stack(channels, axis=1), dtype=np.float32)


def extract_paired_slices_from_subvolume(
    lr_sub: np.ndarray,
    hr_sub: np.ndarray,
    scale_factor: float = 2,
    axes: tuple[str, ...] = ("yz",),
    context_slices: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract cell-centered aligned ``(low_slices, high_slices)`` from paired 3D ``(lr_sub, hr_sub)``."""
    if context_slices < 1 or context_slices % 2 == 0:
        raise ValueError(
            f"context_slices must be a positive odd integer (1, 3, 5, ...), got {context_slices}."
        )
    axis_map = {"yz": 0, "xz": 1, "xy": 2}
    low_list: list[np.ndarray] = []
    high_list: list[np.ndarray] = []

    for ax in axes:
        dim = axis_map[ax]
        lr_ord = np.moveaxis(lr_sub, dim, 0)
        hr_ord = np.moveaxis(hr_sub, dim, 0)
        ctx_stride = 1
        if lr_ord.shape[0] != hr_ord.shape[0]:
            zoom_f = hr_ord.shape[0] / float(lr_ord.shape[0])
            lr_ord = zoom_cell_centered(lr_ord, zoom_factors=(zoom_f, 1.0, 1.0), order=3)
            lr_ord = np.clip(lr_ord, 0.0, 1.0)
            ctx_stride = max(1, int(round(zoom_f)))
        if context_slices > 1:
            lr_ctx = build_strided_context_windows(
                lr_ord, context_slices=context_slices, context_stride=ctx_stride
            )
            low_list.append(lr_ctx)
        else:
            low_list.append(lr_ord)
        high_list.append(hr_ord)

    if len(low_list) > 1:
        max_lh = max(int(a.shape[-2]) for a in low_list)
        max_lw = max(int(a.shape[-1]) for a in low_list)
        max_hh = max(int(a.shape[-2]) for a in high_list)
        max_hw = max(int(a.shape[-1]) for a in high_list)
        low_list = [_safe_pad_spatial(a, max_lh, max_lw) for a in low_list]
        high_list = [_safe_pad_spatial(a, max_hh, max_hw) for a in high_list]

    return np.concatenate(low_list, axis=0), np.concatenate(high_list, axis=0)


class PairedSliceDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch Dataset of paired ``(low_res, high_res)`` 2D or 2.5D XCT slices.

    Supports integer and fractional ``scale_factor`` values, non-square slices (applying
    shape-preserving ``D_2`` flips/180-deg rotations when ``H != W`` and full ``D_4`` symmetry
    when ``H == W``), and ``O(3)`` through-plane ``z``-stack reversal when ``context_slices > 1``.
    """

    def __init__(
        self,
        low_res: Tensor | np.ndarray,
        high_res: Tensor | np.ndarray | None = None,
        patch_size_lr: int | None = None,
        scale_factor: float = 2,
        augment: bool = False,
        context_slices: int = 1,
        context_stride: int = 1,
    ) -> None:
        super().__init__()
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError(
                f"context_slices must be a positive odd integer (1, 3, 5, ...), got {context_slices}."
            )
        self.context_slices = int(context_slices)
        self.context_stride = max(1, int(context_stride))

        lr_arr = sanitize_xct_volume(np.asarray(low_res, dtype=np.float32))
        if lr_arr.ndim == 3 and self.context_slices > 1:
            lr_arr = build_strided_context_windows(
                lr_arr,
                context_slices=self.context_slices,
                context_stride=self.context_stride,
            )

        self.low_res = torch.as_tensor(lr_arr, dtype=torch.float32)
        self.high_res = (
            torch.as_tensor(
                sanitize_xct_volume(np.asarray(high_res, dtype=np.float32)), dtype=torch.float32
            )
            if high_res is not None
            else None
        )
        if self.low_res.ndim == 4 and self.low_res.shape[1] == 1 and self.context_slices == 1:
            self.low_res = self.low_res.squeeze(1)
        if self.high_res is not None and self.high_res.ndim == 4 and self.high_res.shape[1] == 1:
            self.high_res = self.high_res.squeeze(1)

        if self.high_res is not None and self.low_res.shape[0] != self.high_res.shape[0]:
            raise ValueError(
                f"Mismatched slice counts: low_res={self.low_res.shape[0]} vs high_res={self.high_res.shape[0]}."
            )

        self.patch_size_lr = int(patch_size_lr) if patch_size_lr is not None else None
        self.scale_factor = float(scale_factor)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return int(self.low_res.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        lr = self.low_res[idx]
        hr = (
            self.high_res[idx]
            if self.high_res is not None
            else (lr[lr.shape[0] // 2] if lr.ndim == 3 else lr)
        )

        if self.patch_size_lr is not None and self.high_res is not None:
            p_lr = self.patch_size_lr
            p_hr = max(1, int(round(p_lr * self.scale_factor)))
            h_lr, w_lr = int(lr.shape[-2]), int(lr.shape[-1])
            if h_lr < p_lr or w_lr < p_lr:
                lr_np = _safe_pad_spatial(lr.numpy(), max(h_lr, p_lr), max(w_lr, p_lr))
                hr_np = _safe_pad_spatial(
                    hr.numpy(),
                    max(int(hr.shape[-2]), p_hr),
                    max(int(hr.shape[-1]), p_hr),
                )
                lr = torch.from_numpy(lr_np)
                hr = torch.from_numpy(hr_np)
                h_lr, w_lr = int(lr.shape[-2]), int(lr.shape[-1])

            top = int(torch.randint(0, h_lr - p_lr + 1, (1,)).item())
            left = int(torch.randint(0, w_lr - p_lr + 1, (1,)).item())
            lr = lr[..., top : top + p_lr, left : left + p_lr]

            h_hr, w_hr = int(hr.shape[-2]), int(hr.shape[-1])
            top_hr = min(max(0, int(round(top * self.scale_factor))), max(0, h_hr - p_hr))
            left_hr = min(max(0, int(round(left * self.scale_factor))), max(0, w_hr - p_hr))
            hr = hr[..., top_hr : top_hr + p_hr, left_hr : left_hr + p_hr]

        if self.augment and self.high_res is not None:
            is_square = int(lr.shape[-2]) == int(lr.shape[-1]) and int(hr.shape[-2]) == int(
                hr.shape[-1]
            )
            if is_square:
                k_rot = int(torch.randint(0, 4, (1,)).item())
            else:
                k_rot = int(torch.randint(0, 2, (1,)).item()) * 2
            if k_rot > 0:
                lr = torch.rot90(lr, k=k_rot, dims=(-2, -1))
                hr = torch.rot90(hr, k=k_rot, dims=(-2, -1))
            if bool(torch.randint(0, 2, (1,)).item()):
                lr = torch.flip(lr, dims=(-1,))
                hr = torch.flip(hr, dims=(-1,))
            if not is_square and bool(torch.randint(0, 2, (1,)).item()):
                lr = torch.flip(lr, dims=(-2,))
                hr = torch.flip(hr, dims=(-2,))
            if lr.ndim == 3 and lr.shape[0] > 1 and bool(torch.randint(0, 2, (1,)).item()):
                lr = torch.flip(lr, dims=(0,))

        return lr.contiguous(), hr.contiguous()


class ConcatSubdatasetBatchSampler(Sampler[list[int]]):
    """Yield batches strictly within each sub-dataset of a ``ConcatDataset``.

    Prevents ``DataLoader`` collation errors when orthogonal slice views of a non-cubic slab
    (e.g. ``yz`` of ``(32, 32)`` vs ``xz`` of ``(5, 32)``) have distinct spatial shapes without
    corrupting thin-slab validation/test metrics with artificial border padding.
    """

    def __init__(
        self,
        concat_dataset: ConcatDataset[tuple[Tensor, Tensor]],
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        self.concat_dataset = concat_dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self._ranges: list[tuple[int, int]] = []
        start = 0
        for end in concat_dataset.cumulative_sizes:
            if end > start:
                self._ranges.append((start, int(end)))
            start = int(end)

    def __len__(self) -> int:
        return sum(math.ceil((end - start) / self.batch_size) for start, end in self._ranges)

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        for start, end in self._ranges:
            if self.shuffle:
                perm = torch.randperm(end - start).tolist()
                idxs = [start + i for i in perm]
            else:
                idxs = list(range(start, end))
            for k in range(0, len(idxs), self.batch_size):
                batches.append(idxs[k : k + self.batch_size])
        if self.shuffle and len(batches) > 1:
            b_perm = torch.randperm(len(batches)).tolist()
            batches = [batches[i] for i in b_perm]
        yield from batches


def make_slice_dataloader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Construct a ``DataLoader`` that safely batches heterogeneous-shape ``ConcatDataset`` axes."""
    if isinstance(dataset, ConcatDataset) and len(dataset.datasets) > 1:
        shapes: set[tuple[int, int]] = set()
        for sub_ds in dataset.datasets:
            if isinstance(sub_ds, PairedSliceDataset) and sub_ds.patch_size_lr is None:
                shapes.add((int(sub_ds.low_res.shape[-2]), int(sub_ds.low_res.shape[-1])))
        if len(shapes) > 1:
            sampler = ConcatSubdatasetBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def build_multi_axis_slice_dataset(
    lr_sub: np.ndarray,
    hr_sub: np.ndarray,
    scale_factor: float = 2,
    axes: tuple[str, ...] = ("yz",),
    patch_size_lr: int | None = None,
    augment: bool = False,
    context_slices: int = 1,
) -> Dataset[tuple[Tensor, Tensor]]:
    """Build a single- or multi-axis slice dataset with zero artificial border padding."""
    if len(axes) <= 1:
        lo, hi = extract_paired_slices_from_subvolume(
            lr_sub,
            hr_sub,
            scale_factor=scale_factor,
            axes=axes,
            context_slices=context_slices,
        )
        return PairedSliceDataset(
            lo,
            hi,
            patch_size_lr=patch_size_lr,
            scale_factor=scale_factor,
            augment=augment,
            context_slices=context_slices,
        )

    per_axis: list[Dataset[tuple[Tensor, Tensor]]] = []
    for ax in axes:
        lo_ax, hi_ax = extract_paired_slices_from_subvolume(
            lr_sub,
            hr_sub,
            scale_factor=scale_factor,
            axes=(ax,),
            context_slices=context_slices,
        )
        per_axis.append(
            PairedSliceDataset(
                lo_ax,
                hi_ax,
                patch_size_lr=patch_size_lr,
                scale_factor=scale_factor,
                augment=augment,
                context_slices=context_slices,
            )
        )
    return ConcatDataset(per_axis)


def minimum_psf_guard_slices_lr(psf_sigma: float, scale_factor: float) -> int:
    """Minimum LR guard slices needed to prevent 3-sigma HR Gaussian PSF bleed across splits."""
    if psf_sigma <= 0.0:
        return 0
    return int(math.ceil(3.0 * float(psf_sigma) / max(float(scale_factor), 1e-6)))


# Alias preserving clean familiarity with original OpenLSR-X naming
DatasetObject = PairedSliceDataset
