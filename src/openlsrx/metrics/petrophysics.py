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
    """Segment a greyscale XCT volume into a boolean pore mask and return ``(mask, threshold)``."""
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
    if mask.ndim == 3 and min(mask.shape) < 2:
        sq_ax = next(ax for ax, sz in enumerate(mask.shape) if sz < 2)
        mask = np.squeeze(mask, axis=sq_ax)
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
    if mask.ndim == 3 and 1 in mask.shape:
        mask = np.squeeze(mask, axis=mask.shape.index(1))
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
    """Unbiased, zero-padded, full-sphere/circle radially averaged two-point correlation ``S_2(r)``.

    Zero-pads by ``limit`` along every spatial dimension before ``rfftn`` autocorrelation to
    eliminate periodic wrap-around, normalizes each lag vector ``r`` by the exact count of
    in-bounds overlapping voxel pairs ``prod(L_d - |r_d|)``, and radially averages over all
    ``2^d`` signed octants/quadrants for full ``O_h`` rotational invariance.
    """
    mask = np.asarray(pore_mask, dtype=bool)
    if mask.ndim == 3 and 1 in mask.shape:
        mask = np.squeeze(mask, axis=mask.shape.index(1))
    f = mask.astype(np.float64)
    min_dim = min(f.shape)
    limit = max(1, min(min_dim // 2, 64 if max_lag is None else int(max_lag)))

    pad_shape = tuple(int(dim + limit) for dim in f.shape)
    fft_axes = tuple(range(f.ndim))
    fft = np.fft.rfftn(f, s=pad_shape, axes=fft_axes)
    autocorr_full = np.fft.irfftn(np.abs(fft) ** 2, s=pad_shape, axes=fft_axes)

    # Extract centered signed lags in [-(limit - 1), ..., +(limit - 1)]^ndim across all 2^d octants
    lag_1d = np.arange(-(limit - 1), limit, dtype=np.int64)
    idx_per_dim = [np.mod(lag_1d, p_dim) for p_dim in pad_shape]
    sub_corr = autocorr_full[np.ix_(*idx_per_dim)]

    # Exact count of valid non-padded voxel pairs for each signed displacement vector
    overlap_1d = [(float(dim) - np.abs(lag_1d).astype(np.float64)) for dim in f.shape]
    overlap_counts = overlap_1d[0]
    for ov in overlap_1d[1:]:
        overlap_counts = overlap_counts[..., np.newaxis] * ov
    unbiased_corr = sub_corr / np.maximum(overlap_counts, 1.0)

    sq_1d = (lag_1d * lag_1d).astype(np.float64)
    dist_sq = sq_1d
    for sq in sq_1d[np.newaxis, :] if f.ndim == 1 else [sq_1d for _ in range(f.ndim - 1)]:
        dist_sq = dist_sq[..., np.newaxis] + sq
    radii = np.rint(np.sqrt(dist_sq)).astype(np.int32)

    in_shell = radii < limit
    r_valid = radii[in_shell]
    c_valid = unbiased_corr[in_shell]

    radial_sum = np.bincount(r_valid, weights=c_valid, minlength=limit)
    radial_cnt = np.bincount(r_valid, minlength=limit)
    valid = radial_cnt[:limit] > 0
    s2 = np.zeros(limit, dtype=np.float64)
    s2[valid] = radial_sum[:limit][valid] / radial_cnt[:limit][valid]
    return np.clip(s2, 0.0, 1.0)


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
    """Relative error in through-plane vs. in-plane gradient energy for 3D reconstructions."""
    p = np.asarray(pred_vol, dtype=np.float32)
    t = np.asarray(true_vol, dtype=np.float32)
    if p.ndim != 3 or t.ndim != 3 or min(p.shape) < 2 or min(t.shape) < 2:
        raise ValueError("Through-plane anisotropy error requires 3D volumes with dims >= 2.")

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
    """Compute a comprehensive dictionary of porous-media XCT metrics.

    Headline morphological metrics (``porosity_abs_error``, ``pore_dice``, ``pore_iou``,
    ``specific_surface_rel_error``, ``euler_density_abs_error``, ``pore_radius_wasserstein``,
    ``two_point_corr_rmse``) apply the physical ground-truth HR threshold ``threshold_true``
    consistently to both ``target`` and ``pred`` (matching ``evaluate_transport_metrics``), so
    greyscale drift or attenuation bias is rigorously penalized. Secondary contrast-invariant
    diagnostics under independent Otsu thresholding are reported with ``*_auto_otsu`` keys.
    """
    pred_arr = np.asarray(pred, dtype=np.float32)
    true_arr = np.asarray(target, dtype=np.float32)

    if (
        pred_arr.ndim == 3
        and true_arr.ndim == 3
        and pred_arr.shape == true_arr.shape
        and 1 in pred_arr.shape
    ):
        sq_ax = pred_arr.shape.index(1)
        pred_arr = np.squeeze(pred_arr, axis=sq_ax)
        true_arr = np.squeeze(true_arr, axis=sq_ax)

    true_mask, hr_thr = segment_pore_space(
        true_arr,
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
    pred_mask_auto, sr_thr = segment_pore_space(
        pred_arr,
        method=threshold_method,
        fixed_value=fixed_threshold,
        pore_is_low=pore_is_low,
    )

    phi_true = compute_porosity(true_mask)
    phi_pred = compute_porosity(pred_mask_locked)
    phi_pred_auto = compute_porosity(pred_mask_auto)

    dice_locked, iou_locked = compute_dice_iou(pred_mask_locked, true_mask)
    dice_auto, _ = compute_dice_iou(pred_mask_auto, true_mask)

    sv_true = compute_specific_surface_area(true_mask, voxel_size=voxel_size)
    sv_pred = compute_specific_surface_area(pred_mask_locked, voxel_size=voxel_size)
    sv_rel_err = abs(sv_pred - sv_true) / max(sv_true, 1e-8)

    euler_true = compute_euler_density(true_mask, voxel_size=voxel_size)
    euler_pred = compute_euler_density(pred_mask_locked, voxel_size=voxel_size)

    psd_w1 = compute_pore_radius_wasserstein(pred_mask_locked, true_mask, voxel_size=voxel_size)
    s2_rmse = compute_s2_rmse(pred_mask_locked, true_mask)

    metrics: dict[str, float] = {
        "threshold_true": hr_thr,
        "threshold_pred": hr_thr,
        "threshold_pred_auto_otsu": sr_thr,
        "porosity_true": phi_true,
        "porosity_pred": phi_pred,
        "porosity_abs_error": abs(phi_pred - phi_true),
        "porosity_rel_error": abs(phi_pred - phi_true) / max(phi_true, 1e-8),
        "porosity_locked_abs_error": abs(phi_pred - phi_true),
        "porosity_auto_otsu_pred": phi_pred_auto,
        "porosity_auto_otsu_abs_error": abs(phi_pred_auto - phi_true),
        "pore_dice": dice_locked,
        "pore_iou": iou_locked,
        "pore_dice_locked": dice_locked,
        "pore_dice_auto_otsu": dice_auto,
        "specific_surface_true": sv_true,
        "specific_surface_pred": sv_pred,
        "specific_surface_rel_error": sv_rel_err,
        "euler_density_true": euler_true,
        "euler_density_pred": euler_pred,
        "euler_density_abs_error": abs(euler_pred - euler_true),
        "pore_radius_wasserstein": psd_w1,
        "two_point_corr_rmse": s2_rmse,
    }
    if pred_arr.ndim == 3 and true_arr.ndim == 3 and min(pred_arr.shape) >= 2:
        metrics["through_plane_anisotropy_error"] = compute_through_plane_anisotropy_error(
            pred_arr, true_arr, stack_axis=stack_axis
        )
    return metrics


def evaluate_triphase_zn13x_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    thresholds: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Quantify three-phase reactive microstructure in Zn-doped Zeolite 13X XCT volumes.

    Locks ``(t_low, t_high)`` on ``target`` (ground-truth HR) and applies those exact physical
    attenuation thresholds to ``pred`` so greyscale shift across Pore / ``Na-13X`` / ``Zn-13X``
    phases is faithfully measured.
    """
    from skimage.filters import threshold_multiotsu

    pred_arr = np.asarray(pred, dtype=np.float32)
    true_arr = np.asarray(target, dtype=np.float32)
    if pred_arr.shape != true_arr.shape:
        raise ValueError(f"Shape mismatch: {pred_arr.shape} vs {true_arr.shape}.")

    if thresholds is not None:
        t_lo, t_hi = float(thresholds[0]), float(thresholds[1])
    elif float(np.ptp(true_arr)) < 1e-4:
        t_lo, t_hi = 0.40, 0.78
    else:
        try:
            thrs = threshold_multiotsu(true_arr, classes=3)
            t_lo, t_hi = float(thrs[0]), float(thrs[1])
        except ValueError:
            p33, p67 = np.percentile(true_arr, [33.3, 66.7])
            t_lo, t_hi = float(p33), float(p67)
        if np.isclose(t_lo, t_hi, atol=1e-5):
            t_lo, t_hi = 0.40, 0.78

    true_pore = true_arr < t_lo
    true_na = (true_arr >= t_lo) & (true_arr < t_hi)
    true_zn = true_arr >= t_hi

    pred_pore = pred_arr < t_lo
    pred_na = (pred_arr >= t_lo) & (pred_arr < t_hi)
    pred_zn = pred_arr >= t_hi

    pore_dice, _ = compute_dice_iou(pred_pore, true_pore)
    na_dice, _ = compute_dice_iou(pred_na, true_na)
    zn_dice, _ = compute_dice_iou(pred_zn, true_zn)
    macro_dice = (pore_dice + na_dice + zn_dice) / 3.0

    f_na_true = float(true_na.mean())
    f_na_pred = float(pred_na.mean())
    f_zn_true = float(true_zn.mean())
    f_zn_pred = float(pred_zn.mean())

    zn_ratio_true = f_zn_true / max(f_zn_true + f_na_true, 1e-8)
    zn_ratio_pred = f_zn_pred / max(f_zn_pred + f_na_pred, 1e-8)

    return {
        "triphase_threshold_low": t_lo,
        "triphase_threshold_high": t_hi,
        "triphase_pore_dice": pore_dice,
        "triphase_na13x_core_dice": na_dice,
        "triphase_zn13x_rim_dice": zn_dice,
        "triphase_macro_dice": macro_dice,
        "na13x_fraction_true": f_na_true,
        "na13x_fraction_pred": f_na_pred,
        "na13x_fraction_abs_error": abs(f_na_pred - f_na_true),
        "zn13x_fraction_true": f_zn_true,
        "zn13x_fraction_pred": f_zn_pred,
        "zn13x_fraction_abs_error": abs(f_zn_pred - f_zn_true),
        "zn_exchange_ratio_true": zn_ratio_true,
        "zn_exchange_ratio_pred": zn_ratio_pred,
        "zn_exchange_ratio_abs_error": abs(zn_ratio_pred - zn_ratio_true),
    }
