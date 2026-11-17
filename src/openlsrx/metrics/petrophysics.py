"""Petrophysical, topological, and morphological XCT metrics for porous media."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import wasserstein_distance
from skimage.filters import threshold_otsu, threshold_yen
from skimage.measure import euler_number, marching_cubes, mesh_surface_area, perimeter

ThresholdMethod = Literal["otsu", "yen", "fixed"]


def compute_threshold(
    volume: np.ndarray,
    method: ThresholdMethod = "otsu",
    fixed_value: float | None = None,
) -> float:
    """Compute a greyscale binarization threshold on a 2D or 3D XCT array."""
    arr = np.asarray(volume, dtype=np.float32)
    if method == "fixed":
        return 0.5 if fixed_value is None else float(fixed_value)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if np.isclose(vmin, vmax):
        return vmin
    if method == "otsu":
        return float(threshold_otsu(arr))
    if method == "yen":
        return float(threshold_yen(arr))
    raise ValueError(
        f"Unsupported threshold method={method!r}. Expected 'otsu', 'yen', or 'fixed'."
    )


def segment_pore_space(
    volume: np.ndarray,
    method: ThresholdMethod = "otsu",
    fixed_value: float | None = None,
    pore_is_low: bool = True,
) -> tuple[np.ndarray, float]:
    """Segment a greyscale XCT volume into a boolean pore mask and return ``(mask, threshold)``.

    Args:
        volume: 2D or 3D greyscale array.
        method: Thresholding algorithm (``'otsu'``, ``'yen'``, or ``'fixed'``).
        fixed_value: Threshold value when ``method == 'fixed'``.
        pore_is_low: ``True`` when pores correspond to lower X-ray attenuation
            (standard absorption XCT); ``False`` when pores have higher intensity
            (e.g., PoreSpy local-thickness maps or inverted greyscale).
    """
    arr = np.asarray(volume, dtype=np.float32)
    thr = compute_threshold(arr, method=method, fixed_value=fixed_value)
    mask = (arr < thr) if pore_is_low else (arr > thr)
    return mask, thr


def compute_porosity(pore_mask: np.ndarray) -> float:
    """Volume/area fraction of the pore phase."""
    return float(np.asarray(pore_mask, dtype=bool).mean())


def compute_dice_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> tuple[float, float]:
    """Compute Dice similarity coefficient and IoU (Jaccard) on binary pore masks."""
    p = np.asarray(pred_mask, dtype=bool)
    t = np.asarray(true_mask, dtype=bool)
    intersection = float(np.logical_and(p, t).sum())
    p_sum = float(p.sum())
    t_sum = float(t.sum())
    union = float(np.logical_or(p, t).sum())
    dice = 1.0 if (p_sum + t_sum) == 0.0 else (2.0 * intersection) / (p_sum + t_sum)
    iou = 1.0 if union == 0.0 else intersection / union
    return dice, iou


def compute_specific_surface_area(pore_mask: np.ndarray, voxel_size: float = 1.0) -> float:
    """Specific surface area ``S_v`` (interfacial area per bulk volume, or perimeter per area in 2D)."""
    mask = np.asarray(pore_mask, dtype=bool)
    if mask.all() or (~mask).all():
        return 0.0
    if mask.ndim == 2:
        perim = float(perimeter(mask)) * voxel_size
        area = float(mask.size) * (voxel_size**2)
        return perim / area
    if mask.ndim == 3:
        verts, faces, _, _ = marching_cubes(
            mask.astype(np.uint8),
            level=0.5,
            spacing=(voxel_size, voxel_size, voxel_size),
        )
        surf_area = float(mesh_surface_area(verts, faces))
        bulk_vol = float(mask.size) * (voxel_size**3)
        return surf_area / bulk_vol
    raise ValueError(f"Expected 2D or 3D binary mask, got shape {mask.shape}.")


def compute_euler_density(pore_mask: np.ndarray, voxel_size: float = 1.0) -> float:
    """Euler characteristic per unit volume/area (measure of pore-network connectivity)."""
    mask = np.asarray(pore_mask, dtype=bool)
    chi = float(euler_number(mask))
    bulk = float(mask.size) * (voxel_size**mask.ndim)
    return chi / bulk


def compute_pore_radius_wasserstein(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    voxel_size: float = 1.0,
) -> float:
    """Wasserstein-1 distance between inscribed pore-radius distributions (via exact EDT)."""
    p_mask = np.asarray(pred_mask, dtype=bool)
    t_mask = np.asarray(true_mask, dtype=bool)
    p_dt = distance_transform_edt(p_mask) * voxel_size
    t_dt = distance_transform_edt(t_mask) * voxel_size

    p_vals = p_dt[p_mask]
    t_vals = t_dt[t_mask]
    if p_vals.size == 0 and t_vals.size == 0:
        return 0.0
    if p_vals.size == 0:
        return float(t_vals.mean())
    if t_vals.size == 0:
        return float(p_vals.mean())
    return float(wasserstein_distance(p_vals, t_vals))


def two_point_correlation_radial(pore_mask: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """FFT-accelerated radially averaged two-point correlation function ``S_2(r)``."""
    f = np.asarray(pore_mask, dtype=np.float64)
    fft = np.fft.fftn(f)
    power = np.abs(fft) ** 2
    autocorr = np.fft.ifftn(power).real / float(f.size)

    # Shift zero-lag to origin and compute integer radial bins up to max_lag.
    min_dim = min(f.shape)
    limit = min(min_dim // 2, 64 if max_lag is None else int(max_lag))
    slices = tuple(slice(0, limit) for _ in range(f.ndim))
    octant = autocorr[slices]

    grids = np.ogrid[slices]
    dist_sq = sum(g * g for g in grids)
    radii = np.rint(np.sqrt(dist_sq)).astype(np.int32)

    radial_sum = np.bincount(radii.ravel(), weights=octant.ravel(), minlength=limit)
    radial_cnt = np.bincount(radii.ravel(), minlength=limit)
    valid = radial_cnt[:limit] > 0
    s2 = np.zeros(limit, dtype=np.float64)
    s2[valid] = radial_sum[:limit][valid] / radial_cnt[:limit][valid]
    return s2


def compute_s2_rmse(pred_mask: np.ndarray, true_mask: np.ndarray, max_lag: int = 48) -> float:
    """Root-mean-square discrepancy between radial two-point correlation curves ``S_2(r)``."""
    s2_pred = two_point_correlation_radial(pred_mask, max_lag=max_lag)
    s2_true = two_point_correlation_radial(true_mask, max_lag=max_lag)
    n = min(len(s2_pred), len(s2_true))
    return float(np.sqrt(np.mean((s2_pred[:n] - s2_true[:n]) ** 2)))


def compute_through_plane_anisotropy_error(
    pred_vol: np.ndarray,
    true_vol: np.ndarray,
    stack_axis: int = 0,
    eps: float = 1e-8,
) -> float:
    """Relative error in through-plane vs. in-plane gradient energy for 3D reconstructions.

    Directly measures slice-stacking ("venetian blind") discontinuities along ``stack_axis``
    normalized against the natural structural anisotropy of the ground-truth volume.
    """
    p = np.asarray(pred_vol, dtype=np.float32)
    t = np.asarray(true_vol, dtype=np.float32)
    if p.ndim != 3 or t.ndim != 3:
        raise ValueError("Through-plane anisotropy error requires 3D volumes.")

    def _grad_ratio(vol: np.ndarray) -> float:
        grads = [float(np.mean(np.abs(np.diff(vol, axis=ax)))) for ax in range(3)]
        through = grads[stack_axis]
        in_plane = np.mean([grads[ax] for ax in range(3) if ax != stack_axis])
        return float(through / max(float(in_plane), eps))

    ratio_pred = _grad_ratio(p)
    ratio_true = _grad_ratio(t)
    if ratio_true < eps:
        return abs(ratio_pred - ratio_true)
    return abs(ratio_pred / ratio_true - 1.0)


def evaluate_petrophysical_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold_method: ThresholdMethod = "otsu",
    fixed_threshold: float | None = None,
    pore_is_low: bool = True,
    voxel_size: float = 1.0,
    stack_axis: int = 0,
) -> dict[str, float]:
    """Compute a comprehensive dictionary of porous-media XCT metrics."""
    pred_arr = np.asarray(pred, dtype=np.float32)
    true_arr = np.asarray(target, dtype=np.float32)

    true_mask, hr_thr = segment_pore_space(
        true_arr,
        method=threshold_method,
        fixed_value=fixed_threshold,
        pore_is_low=pore_is_low,
    )
    pred_mask_auto, sr_thr = segment_pore_space(
        pred_arr,
        method=threshold_method,
        fixed_value=fixed_threshold,
        pore_is_low=pore_is_low,
    )
    pred_mask_locked, _ = segment_pore_space(
        pred_arr,
        method="fixed",
        fixed_value=hr_thr,
        pore_is_low=pore_is_low,
    )

    phi_true = compute_porosity(true_mask)
    phi_pred = compute_porosity(pred_mask_auto)
    phi_pred_locked = compute_porosity(pred_mask_locked)

    dice, iou = compute_dice_iou(pred_mask_auto, true_mask)
    dice_locked, _ = compute_dice_iou(pred_mask_locked, true_mask)

    sv_true = compute_specific_surface_area(true_mask, voxel_size=voxel_size)
    sv_pred = compute_specific_surface_area(pred_mask_auto, voxel_size=voxel_size)
    sv_rel_err = abs(sv_pred - sv_true) / max(sv_true, 1e-8)

    euler_true = compute_euler_density(true_mask, voxel_size=voxel_size)
    euler_pred = compute_euler_density(pred_mask_auto, voxel_size=voxel_size)

    psd_w1 = compute_pore_radius_wasserstein(pred_mask_auto, true_mask, voxel_size=voxel_size)
    s2_rmse = compute_s2_rmse(pred_mask_auto, true_mask)

    metrics: dict[str, float] = {
        "threshold_true": hr_thr,
        "threshold_pred": sr_thr,
        "porosity_true": phi_true,
        "porosity_pred": phi_pred,
        "porosity_abs_error": abs(phi_pred - phi_true),
        "porosity_rel_error": abs(phi_pred - phi_true) / max(phi_true, 1e-8),
        "porosity_locked_abs_error": abs(phi_pred_locked - phi_true),
        "pore_dice": dice,
        "pore_iou": iou,
        "pore_dice_locked": dice_locked,
        "specific_surface_true": sv_true,
        "specific_surface_pred": sv_pred,
        "specific_surface_rel_error": sv_rel_err,
        "euler_density_true": euler_true,
        "euler_density_pred": euler_pred,
        "euler_density_abs_error": abs(euler_pred - euler_true),
        "pore_radius_wasserstein": psd_w1,
        "two_point_corr_rmse": s2_rmse,
    }
    if pred_arr.ndim == 3 and true_arr.ndim == 3:
        metrics["through_plane_anisotropy_error"] = compute_through_plane_anisotropy_error(
            pred_arr, true_arr, stack_axis=stack_axis
        )
    return metrics
