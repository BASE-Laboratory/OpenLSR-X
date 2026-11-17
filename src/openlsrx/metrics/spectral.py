"""2D Fourier Ring Correlation (FRC) and 3D Fourier Shell Correlation (FSC) metrology for XCT.

Provides synchrotron beamline resolution metrology (van Heel & Schatz, J. Struct. Biol. 2005)
to distinguish genuine high-wavenumber spatial recovery from GAN high-frequency hallucination.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _tukey_window_1d(n: int, alpha: float = 0.15) -> np.ndarray:
    if n <= 2 or alpha <= 0.0:
        return np.ones(n, dtype=np.float64)
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    left = x < (alpha / 2.0)
    right = x > (1.0 - alpha / 2.0)
    w[left] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[left] - alpha / 2.0)))
    w[right] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[right] - 1.0 + alpha / 2.0)))
    return w


def _apodize_nd(arr: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    centered = np.asarray(arr, dtype=np.float64) - float(np.mean(arr))
    win = np.ones_like(centered, dtype=np.float64)
    for ax, length in enumerate(centered.shape):
        w1d = _tukey_window_1d(length, alpha=alpha)
        shape = [1] * centered.ndim
        shape[ax] = length
        win = win * w1d.reshape(shape)
    return centered * win


def _half_bit_threshold(shell_counts: np.ndarray) -> np.ndarray:
    """van Heel & Schatz (2005) 1/2-bit information threshold per Fourier ring/shell."""
    n_eff = np.maximum(shell_counts.astype(np.float64), 1.0)
    sqrt_n = np.sqrt(n_eff)
    return (0.2071 + 1.9102 / sqrt_n) / (1.2071 + 0.9102 / sqrt_n)


def _find_crossing_frequency(
    freqs: np.ndarray,
    corr: np.ndarray,
    thresh: np.ndarray | float,
) -> float:
    """Return first frequency (cycles/voxel) where ``corr`` drops below ``thresh`` (or 0.5 Nyquist)."""
    thr_arr = (
        np.full_like(corr, float(thresh))
        if isinstance(thresh, (int, float))
        else np.asarray(thresh, dtype=np.float64)
    )
    below = np.flatnonzero(corr < thr_arr)
    if below.size == 0:
        return 0.5
    idx = int(below[0])
    if idx == 0:
        return float(freqs[0])
    # Linear interpolation between (idx - 1) and idx
    y0 = float(corr[idx - 1] - thr_arr[idx - 1])
    y1 = float(corr[idx] - thr_arr[idx])
    denom = y0 - y1
    frac = np.clip(y0 / denom, 0.0, 1.0) if abs(denom) > 1e-12 else 0.0
    return float(freqs[idx - 1] + frac * (freqs[idx] - freqs[idx - 1]))


def _compute_fourier_correlation_nd(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_size_um: float = 1.0,
    apodize: bool = True,
    eps: float = 1e-12,
) -> dict[str, Any]:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch in Fourier correlation: {p.shape} vs {t.shape}.")
    if min(p.shape) < 4:
        raise ValueError(f"All spatial dimensions must be >= 4, got {p.shape}.")

    w_p = _apodize_nd(p) if apodize else (p - float(np.mean(p)))
    w_t = _apodize_nd(t) if apodize else (t - float(np.mean(t)))

    f_p = np.fft.fftn(w_p)
    f_t = np.fft.fftn(w_t)

    freq_grids = [np.fft.fftfreq(n) for n in p.shape]
    mesh = np.meshgrid(*freq_grids, indexing="ij")
    k_mag = np.sqrt(sum(g * g for g in mesh))  # cycles / voxel in [0, sqrt(ndim)*0.5]

    n_shells = max(2, min(p.shape) // 2)
    edges = np.linspace(0.0, 0.5, n_shells + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_idx = np.digitize(k_mag.ravel(), edges) - 1
    valid_mask = (shell_idx >= 0) & (shell_idx < n_shells)
    bins = shell_idx[valid_mask]

    cross_real = (f_p * np.conj(f_t)).real.ravel()[valid_mask]
    pow_p = (np.abs(f_p) ** 2).ravel()[valid_mask]
    pow_t = (np.abs(f_t) ** 2).ravel()[valid_mask]

    num = np.bincount(bins, weights=cross_real, minlength=n_shells)
    den_p = np.bincount(bins, weights=pow_p, minlength=n_shells)
    den_t = np.bincount(bins, weights=pow_t, minlength=n_shells)
    counts = np.bincount(bins, minlength=n_shells)

    denom = np.sqrt(den_p * den_t)
    corr = np.where(denom > eps, num / np.maximum(denom, eps), 0.0)
    corr = np.clip(corr, -1.0, 1.0)

    half_bit = _half_bit_threshold(counts)
    k_half_bit = _find_crossing_frequency(centers, corr, half_bit)
    k_0143 = _find_crossing_frequency(centers, corr, 0.143)
    k_050 = _find_crossing_frequency(centers, corr, 0.5)

    vox = float(voxel_size_um)
    res_half_bit_um = vox / (2.0 * max(k_half_bit, 1e-6))
    res_0143_um = vox / (2.0 * max(k_0143, 1e-6))

    pos_corr = np.clip(corr, 0.0, 1.0)
    dx = np.diff(centers)
    auc = (
        float(
            np.sum(0.5 * (pos_corr[:-1] + pos_corr[1:]) * dx) / max(centers[-1] - centers[0], 1e-8)
        )
        if centers.size >= 2
        else float(pos_corr.mean())
    )

    return {
        "frequencies_cyc_per_voxel": centers.astype(np.float32),
        "correlation": corr.astype(np.float32),
        "half_bit_threshold": half_bit.astype(np.float32),
        "shell_counts": counts.astype(np.int64),
        "cutoff_freq_half_bit": k_half_bit,
        "cutoff_freq_0143": k_0143,
        "cutoff_freq_050": k_050,
        "nyquist_fraction_half_bit": float(np.clip(k_half_bit / 0.5, 0.0, 1.0)),
        "effective_resolution_half_bit_um": float(res_half_bit_um),
        "effective_resolution_0143_um": float(res_0143_um),
        "spectral_auc": float(np.clip(auc, 0.0, 1.0)),
    }


def compute_frc_2d(
    pred_slice: np.ndarray,
    true_slice: np.ndarray,
    voxel_size_um: float = 1.0,
    apodize: bool = True,
) -> dict[str, Any]:
    """Compute 2D Fourier Ring Correlation (FRC) and effective half-pitch resolution."""
    p = np.asarray(pred_slice, dtype=np.float64)
    t = np.asarray(true_slice, dtype=np.float64)
    if p.ndim != 2 or t.ndim != 2:
        raise ValueError(f"compute_frc_2d expects 2D arrays, got {p.shape} and {t.shape}.")
    return _compute_fourier_correlation_nd(p, t, voxel_size_um=voxel_size_um, apodize=apodize)


def compute_fsc_3d(
    pred_vol: np.ndarray,
    true_vol: np.ndarray,
    voxel_size_um: float = 1.0,
    apodize: bool = True,
) -> dict[str, Any]:
    """Compute 3D Fourier Shell Correlation (FSC) and effective synchrotron resolution cutoff."""
    p = np.asarray(pred_vol, dtype=np.float64)
    t = np.asarray(true_vol, dtype=np.float64)
    if p.ndim != 3 or t.ndim != 3:
        raise ValueError(f"compute_fsc_3d expects 3D volumes, got {p.shape} and {t.shape}.")
    return _compute_fourier_correlation_nd(p, t, voxel_size_um=voxel_size_um, apodize=apodize)


def evaluate_spectral_metrics(
    pred_vol: np.ndarray,
    true_vol: np.ndarray,
    voxel_size_um: float = 1.0,
) -> dict[str, float]:
    """Return JSON-serializable scalar summary of 3D FSC (or 2D FRC) resolution metrology."""
    arr = np.asarray(pred_vol)
    res = (
        compute_fsc_3d(pred_vol, true_vol, voxel_size_um=voxel_size_um)
        if arr.ndim == 3
        else compute_frc_2d(pred_vol, true_vol, voxel_size_um=voxel_size_um)
    )
    return {
        "fsc_auc": float(res["spectral_auc"]),
        "fsc_cutoff_half_bit_cyc_vox": float(res["cutoff_freq_half_bit"]),
        "fsc_cutoff_0143_cyc_vox": float(res["cutoff_freq_0143"]),
        "fsc_nyquist_fraction": float(res["nyquist_fraction_half_bit"]),
        "effective_resolution_half_bit_um": float(res["effective_resolution_half_bit_um"]),
        "effective_resolution_0143_um": float(res["effective_resolution_0143_um"]),
    }
