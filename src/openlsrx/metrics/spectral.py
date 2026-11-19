"""2D Fourier Ring Correlation (FRC) and 3D Fourier Shell Correlation (FSC) metrology for XCT.

Provides synchrotron beamline resolution metrology (van Heel & Schatz, J. Struct. Biol. 2005)
to distinguish genuine high-wavenumber spatial recovery from GAN high-frequency hallucination.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt


def _tukey_window_1d(n: int, alpha: float = 0.15) -> np.ndarray:
    if n <= 2 or alpha <= 0.0:
        return np.ones(n, dtype=np.float64)
    eff_alpha = min(1.0, max(float(alpha), 4.0 / float(n)))
    # Cell-centered coordinates keep boundary slices tapered (> 0) rather than zeroed out
    x = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    w = np.ones(n, dtype=np.float64)
    left = x < (eff_alpha / 2.0)
    right = x > (1.0 - eff_alpha / 2.0)
    w[left] = 0.5 * (1.0 + np.cos(2.0 * np.pi / eff_alpha * (x[left] - eff_alpha / 2.0)))
    w[right] = 0.5 * (1.0 + np.cos(2.0 * np.pi / eff_alpha * (x[right] - 1.0 + eff_alpha / 2.0)))
    return w


def _prepare_windowed_zero_dc(
    arr: np.ndarray,
    apodize: bool = True,
    alpha: float = 0.15,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Subtract exact window-weighted mean so ``F(0) == 0`` identically and return effective DoF factor."""
    data = np.asarray(arr, dtype=np.float64)
    win = np.ones(data.shape, dtype=np.float64)
    if apodize:
        for ax, length in enumerate(data.shape):
            w1d = _tukey_window_1d(length, alpha=alpha)
            shape = [1] * data.ndim
            shape[ax] = length
            win = win * w1d.reshape(shape)

    if mask is not None:
        m_bool = np.asarray(mask, dtype=bool)
        if m_bool.shape != data.shape:
            raise ValueError(f"Mask shape {m_bool.shape} does not match array shape {data.shape}.")
        if bool(m_bool.any()) and not bool(m_bool.all()):
            dt = distance_transform_edt(m_bool)
            taper_width = max(2.0, 0.04 * float(min(data.shape)))
            smooth_mask = 0.5 * (1.0 - np.cos(np.pi * np.clip(dt / taper_width, 0.0, 1.0)))
            win = win * smooth_mask
        elif not bool(m_bool.any()):
            return np.zeros_like(data, dtype=np.float64), 1.0

    w_sum = float(np.sum(win))
    if w_sum <= 1e-12:
        return np.zeros_like(data, dtype=np.float64), 1.0
    w_sq_sum = float(np.sum(win * win))
    f_win = float(np.clip((w_sum * w_sum) / max(float(win.size) * w_sq_sum, 1e-12), 0.05, 1.0))
    mu_w = float(np.sum(data * win) / w_sum)
    return (data - mu_w) * win, f_win


def _half_bit_threshold(shell_counts: np.ndarray, dof_scale: float = 1.0) -> np.ndarray:
    """van Heel & Schatz (2005) 1/2-bit information threshold per Fourier ring/shell."""
    n_eff = np.maximum(shell_counts.astype(np.float64) * float(dof_scale), 2.0)
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
        # Interpolate from DC limit (k=0, corr=1.0) to first shell (freqs[0], corr[0])
        y0 = 1.0 - float(thr_arr[0])
        y1 = float(corr[0] - thr_arr[0])
        if y0 <= 0.0:
            return 0.0
        denom = y0 - y1
        frac = float(np.clip(y0 / denom, 0.0, 1.0)) if abs(denom) > 1e-12 else 0.0
        return float(frac * freqs[0])
    # Linear interpolation between (idx - 1) and idx
    y0 = float(corr[idx - 1] - thr_arr[idx - 1])
    y1 = float(corr[idx] - thr_arr[idx])
    denom = y0 - y1
    frac = float(np.clip(y0 / denom, 0.0, 1.0)) if abs(denom) > 1e-12 else 0.0
    return float(freqs[idx - 1] + frac * (freqs[idx] - freqs[idx - 1]))


def _compute_fourier_correlation_nd(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_size_um: float = 1.0,
    apodize: bool = True,
    mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch in Fourier correlation: {p.shape} vs {t.shape}.")
    if min(p.shape) < 4:
        raise ValueError(f"All spatial dimensions must be >= 4, got {p.shape}.")

    w_p, f_win = _prepare_windowed_zero_dc(p, apodize=apodize, mask=mask)
    w_t, _ = _prepare_windowed_zero_dc(t, apodize=apodize, mask=mask)

    # Real half-spectrum FFT avoids redundant conjugate storage while accounting for exact multiplicity
    f_p = np.fft.rfftn(w_p)
    f_t = np.fft.rfftn(w_t)

    lead_freqs = [np.fft.fftfreq(n) for n in p.shape[:-1]]
    last_n = p.shape[-1]
    last_freq = np.fft.rfftfreq(last_n)
    mesh = np.meshgrid(*lead_freqs, last_freq, indexing="ij")
    k_mag = np.sqrt(sum(g * g for g in mesh))

    # Multiplicity of rfftn entries in the full Hermitian-symmetric DFT:
    # kx == 0 (and kx == 0.5 when last_n is even) appear once; interior kx > 0 appear twice.
    mult = np.full(f_p.shape, 2.0, dtype=np.float64)
    mult[..., 0] = 1.0
    if last_n % 2 == 0 and f_p.shape[-1] > 1:
        mult[..., -1] = 1.0

    # Use max(p.shape) // 2 so anisotropic thin test slabs (e.g. 24 x 96 x 96) retain fine
    # radial wavenumber resolution along the high-resolution axes instead of collapsing shells.
    n_shells = max(4, max(p.shape) // 2)
    edges = np.linspace(0.0, 0.5, n_shells + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Exclude k == 0 (DC origin) and bin (edges[i], edges[i + 1]] with right=True so fundamental
    # modes k = 1/N_max always populate shell 0 (never leaving a single-voxel DC shell!).
    k_flat = k_mag.ravel()
    shell_idx = np.digitize(k_flat, edges, right=True) - 1
    valid_mask = (k_flat > 0.0) & (shell_idx >= 0) & (shell_idx < n_shells)
    bins = shell_idx[valid_mask]
    m_valid = mult.ravel()[valid_mask]

    cross_real = ((f_p * np.conj(f_t)).real.ravel()[valid_mask]) * m_valid
    pow_p = ((np.abs(f_p) ** 2).ravel()[valid_mask]) * m_valid
    pow_t = ((np.abs(f_t) ** 2).ravel()[valid_mask]) * m_valid

    num = np.bincount(bins, weights=cross_real, minlength=n_shells)
    den_p = np.bincount(bins, weights=pow_p, minlength=n_shells)
    den_t = np.bincount(bins, weights=pow_t, minlength=n_shells)
    counts = np.rint(np.bincount(bins, weights=m_valid, minlength=n_shells)).astype(np.int64)

    nonempty = counts > 0
    if not np.all(nonempty) and np.any(nonempty):
        centers = centers[nonempty]
        num = num[nonempty]
        den_p = den_p[nonempty]
        den_t = den_t[nonempty]
        counts = counts[nonempty]

    denom = np.sqrt(den_p * den_t)
    corr = np.where(denom > eps, num / np.maximum(denom, eps), 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    # Reference-normalized effective modulation transfer Re<P, T*> / <|T|^2> captures amplitude roll-off
    # even under noiseless zero-phase Gaussian smoothing (where shell-normalized FSC cancels H(k) > 0).
    mtf_eff = np.where(den_t > eps, num / np.maximum(den_t, eps), 0.0)
    mtf_eff = np.clip(mtf_eff, -1.0, 2.0)

    half_bit = _half_bit_threshold(counts, dof_scale=f_win)
    k_half_bit = _find_crossing_frequency(centers, corr, half_bit)
    k_0143 = _find_crossing_frequency(centers, corr, 0.143)
    k_050 = _find_crossing_frequency(centers, corr, 0.5)
    k_mtf_050 = _find_crossing_frequency(centers, mtf_eff, 0.5)

    vox = float(voxel_size_um)
    max_domain_um = vox * float(max(p.shape))
    res_half_bit_um = (vox / (2.0 * k_half_bit)) if k_half_bit > 1e-6 else max_domain_um
    res_0143_um = (vox / (2.0 * k_0143)) if k_0143 > 1e-6 else max_domain_um
    res_050_um = (vox / (2.0 * k_050)) if k_050 > 1e-6 else max_domain_um
    res_mtf_050_um = (vox / (2.0 * k_mtf_050)) if k_mtf_050 > 1e-6 else max_domain_um

    pos_corr = np.clip(corr, 0.0, 1.0)
    pos_mtf = np.clip(mtf_eff, 0.0, 1.0)
    dx = np.diff(centers)
    auc = (
        float(
            np.sum(0.5 * (pos_corr[:-1] + pos_corr[1:]) * dx) / max(centers[-1] - centers[0], 1e-8)
        )
        if centers.size >= 2
        else float(pos_corr.mean())
    )
    mtf_auc = (
        float(np.sum(0.5 * (pos_mtf[:-1] + pos_mtf[1:]) * dx) / max(centers[-1] - centers[0], 1e-8))
        if centers.size >= 2
        else float(pos_mtf.mean())
    )

    return {
        "frequencies_cyc_per_voxel": centers.astype(np.float32),
        "correlation": corr.astype(np.float32),
        "effective_mtf": mtf_eff.astype(np.float32),
        "half_bit_threshold": half_bit.astype(np.float32),
        "shell_counts": counts,
        "window_dof_scale": f_win,
        "cutoff_freq_half_bit": k_half_bit,
        "cutoff_freq_0143": k_0143,
        "cutoff_freq_050": k_050,
        "cutoff_freq_mtf_050": k_mtf_050,
        "nyquist_fraction_050": float(np.clip(k_050 / 0.5, 0.0, 1.0)),
        "nyquist_fraction_mtf_050": float(np.clip(k_mtf_050 / 0.5, 0.0, 1.0)),
        "nyquist_fraction_half_bit": float(np.clip(k_half_bit / 0.5, 0.0, 1.0)),
        "spatial_half_pitch_050_um": float(res_050_um),
        "spatial_half_pitch_mtf_050_um": float(res_mtf_050_um),
        "effective_resolution_050_um": float(res_050_um),
        "effective_resolution_half_bit_um": float(res_half_bit_um),
        "effective_resolution_0143_um": float(res_0143_um),
        "spectral_auc": float(np.clip(auc, 0.0, 1.0)),
        "mtf_auc": float(np.clip(mtf_auc, 0.0, 1.0)),
    }


def compute_frc_2d(
    pred_slice: np.ndarray,
    true_slice: np.ndarray,
    voxel_size_um: float = 1.0,
    apodize: bool = True,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute 2D Fourier Ring Correlation (FRC) against a reference slice."""
    p = np.asarray(pred_slice, dtype=np.float64)
    t = np.asarray(true_slice, dtype=np.float64)
    if p.ndim != 2 or t.ndim != 2:
        raise ValueError(f"compute_frc_2d expects 2D arrays, got {p.shape} and {t.shape}.")
    return _compute_fourier_correlation_nd(
        p, t, voxel_size_um=voxel_size_um, apodize=apodize, mask=mask
    )


def compute_fsc_3d(
    pred_vol: np.ndarray,
    true_vol: np.ndarray,
    voxel_size_um: float = 1.0,
    apodize: bool = True,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute 3D Fourier Shell Correlation (FSC) against a reference volume."""
    p = np.asarray(pred_vol, dtype=np.float64)
    t = np.asarray(true_vol, dtype=np.float64)
    if p.ndim != 3 or t.ndim != 3:
        raise ValueError(f"compute_fsc_3d expects 3D volumes, got {p.shape} and {t.shape}.")
    return _compute_fourier_correlation_nd(
        p, t, voxel_size_um=voxel_size_um, apodize=apodize, mask=mask
    )


def evaluate_spectral_metrics(
    pred_vol: np.ndarray,
    true_vol: np.ndarray,
    voxel_size_um: float = 1.0,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Return JSON-serializable scalar summary of 3D FSC/MTF (or 2D FRC/MTF) agreement with ground truth.

    For map-vs-reference comparisons (super-resolved volume vs ground-truth HR scan), Rosenthal &
    Henderson (*J. Mol. Biol.* 2003) show that ``FSC_ref = 0.5`` is the exact theoretical counterpart
    of the half-map ``FSC_half = 0.143`` criterion; accordingly ``fsc_nyquist_fraction`` reports
    ``nyquist_fraction_050``, while ``fsc_nyquist_fraction_half_bit`` and ``fsc_cutoff_0143_cyc_vox``
    are also retained for completeness.
    """
    p = np.asarray(pred_vol)
    t = np.asarray(true_vol)
    m = np.asarray(mask) if mask is not None else None
    if p.ndim == 3 and t.ndim == 3 and 1 in p.shape and p.shape == t.shape:
        sq_ax = p.shape.index(1)
        p = np.squeeze(p, axis=sq_ax)
        t = np.squeeze(t, axis=sq_ax)
        if m is not None and m.shape == pred_vol.shape:
            m = np.squeeze(m, axis=sq_ax)

    res = (
        compute_fsc_3d(p, t, voxel_size_um=voxel_size_um, mask=m)
        if p.ndim == 3
        else compute_frc_2d(p, t, voxel_size_um=voxel_size_um, mask=m)
    )
    return {
        "fsc_auc": float(res["spectral_auc"]),
        "mtf_auc": float(res["mtf_auc"]),
        "fsc_cutoff_050_cyc_vox": float(res["cutoff_freq_050"]),
        "fsc_nyquist_fraction": float(res["nyquist_fraction_050"]),
        "fsc_nyquist_fraction_050": float(res["nyquist_fraction_050"]),
        "fsc_half_pitch_050_um": float(res["spatial_half_pitch_050_um"]),
        "effective_resolution_050_um": float(res["effective_resolution_050_um"]),
        "mtf_cutoff_050_cyc_vox": float(res["cutoff_freq_mtf_050"]),
        "mtf_nyquist_fraction_050": float(res["nyquist_fraction_mtf_050"]),
        "mtf_half_pitch_050_um": float(res["spatial_half_pitch_mtf_050_um"]),
        "fsc_cutoff_half_bit_cyc_vox": float(res["cutoff_freq_half_bit"]),
        "fsc_nyquist_fraction_half_bit": float(res["nyquist_fraction_half_bit"]),
        "fsc_cutoff_0143_cyc_vox": float(res["cutoff_freq_0143"]),
        "effective_resolution_half_bit_um": float(res["effective_resolution_half_bit_um"]),
        "effective_resolution_0143_um": float(res["effective_resolution_0143_um"]),
    }
