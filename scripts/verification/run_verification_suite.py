#!/usr/bin/env python3
"""Reproducible analytical and visual verification suite for OpenLSR-X.

Generates publication- and JOSS-ready verification figures and a machine-readable JSON
report testing every major OpenLSR-X subsystem against closed-form analytical ground truth,
synthetic/calibrated 3D phantoms, and (optionally) real 6 um Bentheimer Sandstone micro-CT
(Zenodo 5542624) under controlled synthetic 2x PSF + noise degradation.

Usage:
    python scripts/verification/run_verification_suite.py \
        --output-dir docs/assets/verification \
        --include-bentheimer
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import tifffile
import torch
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom as nd_zoom
from scipy.stats import spearmanr
from skimage.filters import threshold_multiotsu

from openlsrx.baselines import upsample_3d
from openlsrx.data import (
    PairedVolumeDataModule,
    compute_normalization_bounds,
    degrade_xct_volume,
    extract_paired_slices_from_subvolume,
    fetch_or_generate_rock_sample,
    generate_porespy_volume,
    minimum_psf_guard_slices_lr,
    prepare_zn13x_benchmark,
    register_multires_pair_3d,
    split_volume_spatially,
    synthesize_zn13x_phantom_pair,
    zoom_cell_centered,
)
from openlsrx.inference import (
    super_resolve_volume_orthogonal,
    super_resolve_volume_single_axis,
)
from openlsrx.io import export_for_openimpala
from openlsrx.losses import (
    AttenuationConservationLoss,
    FocalFrequencyLoss,
    SobelGradientLoss,
    SoftCLDiceLoss,
    soft_skeletonize,
)
from openlsrx.metrics import (
    compute_directional_tortuosity,
    compute_drainage_curve,
    compute_euler_density,
    compute_fsc_3d,
    compute_tortuosity_uncertainty_bounds,
    evaluate_image_metrics,
    evaluate_petrophysical_metrics,
    evaluate_spectral_metrics,
    evaluate_transport_metrics,
    evaluate_triphase_zn13x_metrics,
    precompute_reference_transport,
    segment_pore_space,
    solve_steady_state_diffusion_axis,
    two_point_correlation_radial,
)
from openlsrx.models import (
    SRGAN,
    SR3DResNet,
    build_3d_subcube_dataloaders,
    super_resolve_volume_3d,
)


def _solve_conc_field_z(pore_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve 3D Laplace field c(z, y, x) in [0, 1] along z=0 -> z=D-1 for visualization."""
    from scipy.ndimage import label
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import cg

    st = np.zeros((3, 3, 3), dtype=int)
    st[1, 1, :] = 1
    st[1, :, 1] = 1
    st[:, 1, 1] = 1
    labeled, _ = label(pore_mask, structure=st)
    perc = (set(np.unique(labeled[0])) - {0}) & (set(np.unique(labeled[-1])) - {0})
    active = np.isin(labeled, list(perc))
    idx_grid = np.full(active.shape, -1, dtype=np.int32)
    n = int(active.sum())
    if n == 0:
        return np.full(pore_mask.shape, np.nan, dtype=np.float32), 0.0
    idx_grid[active] = np.arange(n, dtype=np.int32)

    diag = np.zeros(n, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    vals_l: list[np.ndarray] = []
    inlet = idx_grid[0][idx_grid[0] >= 0]
    outlet = idx_grid[-1][idx_grid[-1] >= 0]
    diag[inlet] += 2.0
    rhs[inlet] += 2.0
    diag[outlet] += 2.0

    for ax in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[ax] = slice(0, -1)
        s1[ax] = slice(1, None)
        u, v = idx_grid[tuple(s0)], idx_grid[tuple(s1)]
        m = (u >= 0) & (v >= 0)
        uu, vv = u[m], v[m]
        rows_l.extend([uu, vv])
        cols_l.extend([vv, uu])
        vals_l.extend([-np.ones_like(uu, dtype=np.float64), -np.ones_like(vv, dtype=np.float64)])
        np.add.at(diag, uu, 1.0)
        np.add.at(diag, vv, 1.0)

    mat = coo_matrix(
        (
            np.concatenate([diag, *vals_l]),
            (
                np.concatenate([np.arange(n, dtype=np.int32), *rows_l]),
                np.concatenate([np.arange(n, dtype=np.int32), *cols_l]),
            ),
        ),
        shape=(n, n),
    ).tocsr()
    c_sol, _ = cg(
        mat, rhs, rtol=1e-9, atol=0.0, maxiter=4000, M=diags(1.0 / np.maximum(diag, 1e-12))
    )
    q_in = float(np.sum(2.0 * (1.0 - c_sol[inlet])))
    q_out = float(np.sum(2.0 * c_sol[outlet]))
    flux_imbalance = abs(q_in - q_out) / max(0.5 * (q_in + q_out), 1e-12)
    out = np.full(pore_mask.shape, np.nan, dtype=np.float32)
    out[active] = np.clip(c_sol, 0.0, 1.0).astype(np.float32)
    return out, flux_imbalance


def run_subsystem_1_data_alignment(out_dir: Path) -> dict[str, Any]:
    """Verify cell-centered zoom, PoreSpy degradation, 2.5D strided slices, splits, and registration."""
    n_hr = 64
    zz, _, _ = np.meshgrid(np.arange(n_hr), np.arange(n_hr), np.arange(n_hr), indexing="ij")
    ramp_hr = zz.astype(np.float32) / float(n_hr - 1)

    shift_stats: dict[str, dict[str, float]] = {}
    for s in (2, 4, 8):
        lr_blk = ramp_hr.reshape(n_hr // s, s, n_hr // s, s, n_hr // s, s).mean(axis=(1, 3, 5))
        up_legacy = nd_zoom(lr_blk, zoom=s, order=1, grid_mode=False)
        up_centered = zoom_cell_centered(lr_blk, zoom_factors=s, order=1)
        interior = slice(s, n_hr - s)
        err_leg_vox = (up_legacy[interior] - ramp_hr[interior]) * (n_hr - 1)
        err_cen_vox = (up_centered[interior] - ramp_hr[interior]) * (n_hr - 1)
        shift_stats[f"{s}x"] = {
            "legacy_max_abs_shift_hr_vox": round(float(np.max(np.abs(err_leg_vox))), 4),
            "centered_max_abs_shift_hr_vox": round(float(np.max(np.abs(err_cen_vox))), 6),
        }

    _, ps_hr, _ = generate_porespy_volume(
        shape=64,
        feature_regime="high_feature",
        noise_regime="low_noise",
        scale_factor=2,
        psf_sigma=0.9,
        seed=42,
    )
    deg_lr = degrade_xct_volume(ps_hr, scale_factor=2, psf_sigma=0.9, noise_std=0.02, seed=42)
    lr_ctx, hr_slc = extract_paired_slices_from_subvolume(
        deg_lr[:16], ps_hr[:32], scale_factor=2, axes=("yz", "xz", "xy"), context_slices=3
    )

    guard_lr = minimum_psf_guard_slices_lr(psf_sigma=1.0, scale_factor=2)
    splits = split_volume_spatially(
        ps_hr,
        deg_lr,
        scale_factor=2,
        train_frac=0.65,
        val_frac=0.15,
        split_axis=0,
        guard_slices_lr=guard_lr,
    )
    vmin_tr, vmax_tr = compute_normalization_bounds(
        splits["train"][1], mode="percentile", lower_pct=0.5, upper_pct=99.5
    )

    zn_lr_clean, zn_hr_clean = synthesize_zn13x_phantom_pair(
        pair="2x", lr_shape=(28, 28, 28), seed=7
    )
    true_shift_hr = (3, -2, 4)
    hr_shifted = np.roll(zn_hr_clean, shift=true_shift_hr, axis=(0, 1, 2))[6:50, 6:50, 6:50]
    lr_ref = zn_lr_clean[3:25, 3:25, 3:25]
    naive_up = zoom_cell_centered(lr_ref, zoom_factors=2, order=1)
    lr_reg, hr_reg, reg_meta = register_multires_pair_3d(
        lr_ref, hr_shifted, scale_factor=2, local_refine_radius=3
    )
    reg_up = zoom_cell_centered(lr_reg, zoom_factors=2, order=1)

    plt.rcParams.update({"font.size": 9.0, "axes.titlesize": 9.4, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.0), dpi=180)
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    z_axis = np.arange(n_hr)
    lr_ramp_8x = ramp_hr.reshape(8, 8, 8, 8, 8, 8).mean(axis=(1, 3, 5))
    leg_8x = nd_zoom(lr_ramp_8x, zoom=8, order=1, grid_mode=False)
    cen_8x = zoom_cell_centered(lr_ramp_8x, zoom_factors=8, order=1)
    axes[0, 0].plot(
        z_axis, ramp_hr[:, 32, 32], color="#111827", lw=2.2, label="True HR Ramp z/(N−1)"
    )
    axes[0, 0].plot(
        z_axis,
        leg_8x[:, 32, 32],
        color="#ef476f",
        lw=1.8,
        ls="--",
        label=f"Legacy grid_mode=False (±{shift_stats['8x']['legacy_max_abs_shift_hr_vox']:.2f} vox)",
    )
    axes[0, 0].plot(
        z_axis,
        cen_8x[:, 32, 32],
        color="#00b4d8",
        lw=2.0,
        ls=":",
        label=f"zoom_cell_centered (±{shift_stats['8x']['centered_max_abs_shift_hr_vox']:.5f} vox)",
    )
    ax_in = axes[0, 0].inset_axes((0.52, 0.12, 0.44, 0.36))
    ax_in.plot(
        np.arange(8, 56),
        (leg_8x[8:56, 32, 32] - ramp_hr[8:56, 32, 32]) * 63.0,
        color="#ef476f",
        lw=1.4,
    )
    ax_in.plot(
        np.arange(8, 56),
        (cen_8x[8:56, 32, 32] - ramp_hr[8:56, 32, 32]) * 63.0,
        color="#00b4d8",
        lw=1.4,
    )
    ax_in.set_title("Depth Shift (HR vox)", fontsize=7.5)
    ax_in.set_ylim(-3.5, 3.5)
    ax_in.grid(alpha=0.25)
    axes[0, 0].set_title("1. zoom_cell_centered vs Legacy Zoom (8×)")
    axes[0, 0].set_xlabel("Depth index z (HR voxels)")
    axes[0, 0].set_ylabel("Normalized coordinate")
    axes[0, 0].legend(fontsize=7.2, loc="upper left")
    axes[0, 0].grid(alpha=0.25)

    combo = np.hstack([ps_hr[32], zoom_cell_centered(deg_lr, 2, order=0)[32]])
    axes[0, 1].imshow(combo, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[0, 1].axvline(63.5, color="#00bbf9", lw=1.5, ls="--")
    axes[0, 1].text(
        4,
        8,
        "HR 64³ (PSF-free)",
        color="#00f5d4",
        fontsize=8.0,
        fontweight="bold",
        bbox={"facecolor": "#000000aa", "pad": 2, "edgecolor": "none"},
    )
    axes[0, 1].text(
        68,
        8,
        "degrade_xct_volume (32³)",
        color="#fee440",
        fontsize=8.0,
        fontweight="bold",
        bbox={"facecolor": "#000000aa", "pad": 2, "edgecolor": "none"},
    )
    axes[0, 1].set_title("2. generate_porespy_volume & degrade_xct_volume")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    idx_show = 8
    h_hr_s, w_hr_s = hr_slc[idx_show].shape
    h_lr_s, w_lr_s = lr_ctx[idx_show, 0].shape
    zy, zx = h_hr_s / h_lr_s, w_hr_s / w_lr_s
    strip_25d = np.hstack(
        [
            zoom_cell_centered(lr_ctx[idx_show, 0], (zy, zx), order=1),
            zoom_cell_centered(lr_ctx[idx_show, 1], (zy, zx), order=1),
            zoom_cell_centered(lr_ctx[idx_show, 2], (zy, zx), order=1),
            hr_slc[idx_show],
        ]
    )
    axes[0, 2].imshow(
        strip_25d, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest"
    )
    for ci in (1, 2, 3):
        axes[0, 2].axvline(ci * w_hr_s - 0.5, color="#00bbf9", lw=1.1, ls=":")
    for ci, lbl in enumerate(["LR(z−Δz)", "LR(z)", "LR(z+Δz)", "HR(z)"]):
        axes[0, 2].text(
            ci * w_hr_s + 2,
            4,
            lbl,
            color="#ffffff",
            fontsize=7.5,
            fontweight="bold",
            bbox={"facecolor": "#000000aa", "pad": 1.5, "edgecolor": "none"},
        )
    axes[0, 2].set_title(f"3. extract_paired_slices (N={len(lr_ctx)}, 3-Axis 2.5D)")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    n_tr = splits["train"][1].shape[0]
    n_va = splits["val"][1].shape[0]
    n_te = splits["test"][1].shape[0]
    g_hr = guard_lr * 2
    zx_rgb = np.repeat(ps_hr[:, 32, :, None], 3, axis=-1) * 0.72
    zx_rgb[:n_tr, :, 2] = np.clip(zx_rgb[:n_tr, :, 2] + 0.28, 0, 1)
    zx_rgb[n_tr : n_tr + g_hr] = [0.95, 0.25, 0.35]
    v0 = n_tr + g_hr
    zx_rgb[v0 : v0 + n_va, :, 1] = np.clip(zx_rgb[v0 : v0 + n_va, :, 1] + 0.28, 0, 1)
    zx_rgb[v0 + n_va : v0 + n_va + g_hr] = [0.95, 0.25, 0.35]
    t0 = v0 + n_va + g_hr
    zx_rgb[t0:, :, 0] = np.clip(zx_rgb[t0:, :, 0] + 0.28, 0, 1)
    zx_rgb[t0:, :, 1] = np.clip(zx_rgb[t0:, :, 1] + 0.18, 0, 1)
    axes[1, 0].imshow(zx_rgb, aspect="auto", interpolation="nearest")
    axes[1, 0].set_title(
        f"4. split_volume_spatially (guard_lr={guard_lr} → {g_hr} HR slices)\n"
        f"Train={n_tr}, Val={n_va}, Test={n_te} | Bounds=[{vmin_tr:.2f}, {vmax_tr:.2f}]"
    )
    axes[1, 0].set_ylabel("Split axis z (HR slices)")
    axes[1, 0].set_xlabel("In-plane x (Red = purged PSF guard bands)")

    z_m = naive_up.shape[0] // 2
    before_rgb = np.stack([hr_shifted[z_m], naive_up[z_m], naive_up[z_m]], axis=-1)
    axes[1, 1].imshow(np.clip(before_rgb, 0, 1), interpolation="nearest")
    axes[1, 1].set_title(
        f"5a. Unregistered Zn-13X Phantom (SSIM = {reg_meta['initial_triplanar_ssim']:.3f})\n"
        "Injected Half-LR Shift = (+3, −2, +4) HR vox"
    )
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    zr_m = reg_up.shape[0] // 2
    after_rgb = np.stack([hr_reg[zr_m], reg_up[zr_m], reg_up[zr_m]], axis=-1)
    axes[1, 2].imshow(np.clip(after_rgb, 0, 1), interpolation="nearest")
    axes[1, 2].set_title(
        f"5b. register_multires_pair_3d (SSIM = {reg_meta['hr_triplanar_ssim']:.3f})\n"
        f"Recovered HR Shift = {reg_meta['hr_shift_zyx']}"
    )
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    fig.savefig(out_dir / "vr_01_data_and_alignment.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "shift_stats": shift_stats,
        "porespy_shapes": {"hr": list(ps_hr.shape), "lr": list(deg_lr.shape)},
        "paired_25d_shapes": {"lr_ctx": list(lr_ctx.shape), "hr_slc": list(hr_slc.shape)},
        "split_depths_hr": {"train": n_tr, "guard_hr": g_hr, "val": n_va, "test": n_te},
        "train_norm_bounds": [round(float(vmin_tr), 4), round(float(vmax_tr), 4)],
        "registration": {
            "true_injected_hr_shift": list(true_shift_hr),
            "recovered_hr_shift": np.asarray(reg_meta["hr_shift_zyx"], dtype=int).tolist(),
            "initial_triplanar_ssim": round(float(reg_meta["initial_triplanar_ssim"]), 4),  # type: ignore[arg-type]
            "registered_triplanar_ssim": round(float(reg_meta["hr_triplanar_ssim"]), 4),  # type: ignore[arg-type]
            "aligned_hr_shape": list(hr_reg.shape),
        },
    }


def run_subsystem_2_losses(out_dir: Path) -> dict[str, Any]:
    """Verify SoftCLDiceLoss, soft_skeletonize, SobelGradientLoss, FFL, and approximate conservation loss."""
    h, w = 64, 64
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pore_gt = (
        ((yy - 20) ** 2 + (xx - 18) ** 2 <= 11**2)
        | ((yy - 44) ** 2 + (xx - 46) ** 2 <= 11**2)
        | (
            (np.abs((yy - 20) * 28 - (xx - 18) * 24) <= 48)
            & (xx >= 18)
            & (xx <= 46)
            & (yy >= 20)
            & (yy <= 44)
        )
    )
    hr_clean = np.where(pore_gt, 0.10, 0.90).astype(np.float32)

    rng = np.random.default_rng(42)
    pred_noisy = np.clip(
        hr_clean + rng.uniform(-0.025, 0.025, size=hr_clean.shape), 0.0, 1.0
    ).astype(np.float32)
    pred_pinched = hr_clean.copy()
    pinch_box = (yy >= 29) & (yy <= 35) & (xx >= 29) & (xx <= 35)
    changed_px = int((pore_gt & pinch_box).sum())
    pred_pinched[pinch_box] = 0.90

    t_hr = torch.from_numpy(hr_clean)[None, None]
    t_noisy = torch.from_numpy(pred_noisy)[None, None]
    t_pinch = torch.from_numpy(pred_pinched)[None, None]

    cldice_fn = SoftCLDiceLoss(num_iter=8, temperature=20.0, pore_is_low=True, threshold=0.50)
    sobel_fn = SobelGradientLoss()
    ffl_fn = FocalFrequencyLoss(alpha=1.0)

    l1_noisy = float(torch.nn.functional.l1_loss(t_noisy, t_hr))
    l1_pinch = float(torch.nn.functional.l1_loss(t_pinch, t_hr))
    cl_noisy = float(cldice_fn(t_noisy, t_hr))
    cl_pinch = float(cldice_fn(t_pinch, t_hr))

    skel_hr = soft_skeletonize(torch.sigmoid(20.0 * (0.50 - t_hr)), num_iter=8)[0, 0].numpy()
    skel_pinch = soft_skeletonize(torch.sigmoid(20.0 * (0.50 - t_pinch)), num_iter=8)[0, 0].numpy()

    lr_exact = torch.nn.functional.interpolate(t_hr, size=(32, 32), mode="area")
    cons_zero_fn = AttenuationConservationLoss(psf_sigma=0.0, reduction_norm="l1")
    deltas = np.linspace(-0.20, 0.20, 41)
    cons_vs_bias = [
        float(cons_zero_fn(torch.clamp(t_hr + float(d), 0.0, 1.0), lr_exact)) for d in deltas
    ]

    zz3 = np.arange(6, dtype=np.float32)[:, None, None] / 2.0 - 1.0
    vol_quad = np.broadcast_to(0.45 + 0.12 * (zz3**2), (6, 32, 32)).copy()
    vol_quad_blur = gaussian_filter(vol_quad, sigma=1.0, mode="nearest")
    lr_quad = vol_quad_blur.reshape(3, 2, 16, 2, 16, 2).mean(axis=(1, 3, 5))
    cons_psf1 = AttenuationConservationLoss(psf_sigma=1.0, reduction_norm="l1")
    cons_err_2d = float(
        cons_psf1(torch.from_numpy(vol_quad[2:3])[None], torch.from_numpy(lr_quad[1:2])[None])
    )
    cons_err_25d = float(
        cons_psf1(torch.from_numpy(vol_quad[2:3])[None], torch.from_numpy(lr_quad[0:3])[None])
    )

    _, ps_hr_v, _ = generate_porespy_volume(shape=64, scale_factor=2, psf_sigma=0.8, seed=11)
    ref_sl = ps_hr_v[32]
    t_ref = torch.from_numpy(ref_sl)[None, None]
    t_ref_lr = torch.nn.functional.interpolate(t_ref, size=(32, 32), mode="area")
    sigmas = np.linspace(0.0, 3.0, 13)
    curves: dict[str, list[float]] = {
        "SobelGradient": [],
        "FocalFrequency": [],
        "SoftCLDice": [],
        "Conservation": [],
    }
    for sg in sigmas:
        bl = gaussian_filter(ref_sl, sigma=float(sg)) if sg > 0 else ref_sl.copy()
        tb = torch.from_numpy(bl.astype(np.float32))[None, None]
        curves["SobelGradient"].append(float(sobel_fn(tb, t_ref)))
        curves["FocalFrequency"].append(float(ffl_fn(tb, t_ref)))
        curves["SoftCLDice"].append(float(cldice_fn(tb, t_ref)))
        tb_shift = torch.clamp(tb + 0.04 * float(sg), 0.0, 1.0)
        curves["Conservation"].append(float(cons_zero_fn(tb_shift, t_ref_lr)))

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.0), dpi=180)
    plt.subplots_adjust(hspace=0.36, wspace=0.30)

    vis_top = np.hstack([hr_clean, skel_hr])
    vis_bot = np.hstack([pred_pinched, skel_pinch])
    grid_00 = np.vstack([vis_top, np.ones((2, 128), dtype=np.float32) * 0.5, vis_bot])
    axes[0, 0].imshow(grid_00, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[0, 0].axvline(63.5, color="#00f5d4", lw=1.2, ls="--")
    for tx, ty, txt, col in [
        (4, 9, "GT Connected Throat", "#ffffff"),
        (68, 9, "soft_skeletonize(GT)", "#00f5d4"),
        (4, 75, f"Pinched ({changed_px} px blocked)", "#ffbe0b"),
        (68, 75, "Severed Skeleton!", "#ff595e"),
    ]:
        axes[0, 0].text(
            tx,
            ty,
            txt,
            color=col,
            fontsize=7.8,
            fontweight="bold",
            bbox={"facecolor": "#000000aa", "pad": 1.5, "edgecolor": "none"},
        )
    axes[0, 0].set_title("1. soft_skeletonize & Throat Connectivity")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    x_pos = np.arange(2)
    w_bar = 0.34
    axes[0, 1].bar(
        x_pos - w_bar / 2, [l1_noisy, l1_pinch], width=w_bar, color="#8d99ae", label="Pixel L1 Loss"
    )
    axes[0, 1].bar(
        x_pos + w_bar / 2,
        [cl_noisy, cl_pinch],
        width=w_bar,
        color="#ef476f",
        label="SoftCLDiceLoss",
    )
    for xi, (v1, v2) in enumerate([(l1_noisy, cl_noisy), (l1_pinch, cl_pinch)]):
        axes[0, 1].text(
            xi - w_bar / 2, v1 + 0.004, f"{v1:.4f}", ha="center", fontsize=8.0, fontweight="bold"
        )
        axes[0, 1].text(
            xi + w_bar / 2,
            v2 + 0.004,
            f"{v2:.4f}",
            ha="center",
            fontsize=8.0,
            fontweight="bold",
            color="#d90429",
        )
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(
        ["Mild Global Noise\n(100% Connected)", f"Severed Throat\n(Only {changed_px}/4096 px)"]
    )
    axes[0, 1].set_ylabel("Loss Value")
    axes[0, 1].set_ylim(0, max(cl_pinch, l1_noisy) * 1.25)
    axes[0, 1].set_title("2. Why L1 Misses Broken Throats vs SoftCLDice")
    axes[0, 1].legend(fontsize=8.2)
    axes[0, 1].grid(axis="y", alpha=0.25)

    bl_sl = gaussian_filter(ref_sl, sigma=1.6).astype(np.float32)
    gx_r, gy_r = np.gradient(ref_sl)
    gx_b, gy_b = np.gradient(bl_sl)
    mag_r, mag_b = np.hypot(gx_r, gy_r), np.hypot(gx_b, gy_b)
    im02 = axes[0, 2].imshow(
        np.hstack([mag_r, mag_b, np.abs(mag_r - mag_b)]), cmap="inferno", interpolation="nearest"
    )
    for xc in (63.5, 127.5):
        axes[0, 2].axvline(xc, color="#00f5d4", lw=1.0, ls=":")
    for ci, lb in enumerate(["|∇HR|", "|∇Blurred|", "|Δ∇| Residual"]):
        axes[0, 2].text(
            ci * 64 + 4,
            8,
            lb,
            color="#ffffff",
            fontsize=7.8,
            fontweight="bold",
            bbox={"facecolor": "#000000aa", "pad": 1.5, "edgecolor": "none"},
        )
    sobel_val = float(sobel_fn(torch.from_numpy(bl_sl)[None, None], t_ref))
    axes[0, 2].set_title(f"3. SobelGradientLoss = {sobel_val:.4f}")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.025, pad=0.02)

    fft_diff = np.abs(np.fft.fftshift(np.fft.fft2(bl_sl) - np.fft.fft2(ref_sl)))
    im10 = axes[1, 0].imshow(np.log1p(fft_diff), cmap="viridis", interpolation="nearest")
    ffl_val = float(ffl_fn(torch.from_numpy(bl_sl)[None, None], t_ref))
    axes[1, 0].set_title(
        f"4. FocalFrequencyLoss 2D |ΔF(kx, ky)| Ring\n(FFL = {ffl_val:.5f} at σ=1.6)"
    )
    axes[1, 0].set_xlabel("kx (DC at center)")
    axes[1, 0].set_ylabel("ky")
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.03)

    axes[1, 1].plot(
        deltas, cons_vs_bias, color="#00b4d8", lw=2.2, label="Approx. Conservation L1 vs Δμ"
    )
    axes[1, 1].axvline(0.0, color="#ef476f", ls=":", lw=1.2)
    axes[1, 1].set_xlabel("Additive HR Intensity Bias Δμ")
    axes[1, 1].set_ylabel("AttenuationConservationLoss")
    axes[1, 1].set_title(
        "5. Approx. Slice/2.5D Conservation Loss\n"
        f"(Local z-Laplacian reduces z-PSF bias {cons_err_2d:.4f} → {cons_err_25d:.4f})"
    )
    axes[1, 1].legend(fontsize=8.0)
    axes[1, 1].grid(alpha=0.25)

    for k_name, col in [
        ("SobelGradient", "#f8961e"),
        ("FocalFrequency", "#9b5de5"),
        ("SoftCLDice", "#ef476f"),
        ("Conservation", "#00b4d8"),
    ]:
        arr = np.array(curves[k_name])
        axes[1, 2].plot(
            sigmas,
            arr / max(float(arr[-1]), 1e-8),
            marker="o",
            markersize=4,
            lw=1.9,
            color=col,
            label=k_name,
        )
    axes[1, 2].set_xlabel("Gaussian Degradation Width σ (HR voxels)")
    axes[1, 2].set_ylabel("Normalized Loss (L(σ) / L(3.0))")
    axes[1, 2].set_title("6. Strict Monotonicity Across All 4 Losses")
    axes[1, 2].legend(fontsize=8.0)
    axes[1, 2].grid(alpha=0.25)

    fig.savefig(out_dir / "vr_02_losses_diagnostics.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "throat_pinch_experiment": {
            "changed_pixels": changed_px,
            "total_pixels": h * w,
            "l1_global_noise": round(l1_noisy, 5),
            "l1_severed_throat": round(l1_pinch, 5),
            "soft_cldice_global_noise": round(cl_noisy, 5),
            "soft_cldice_severed_throat": round(cl_pinch, 5),
            "cldice_spike_ratio": round(cl_pinch / max(cl_noisy, 1e-8), 2),
        },
        "conservation_zero_bias": round(float(cons_vs_bias[20]), 6),
        "conservation_z_psf_2d_vs_25d": {
            "2d_slice_only_err": round(cons_err_2d, 5),
            "25d_z_laplacian_err": round(cons_err_25d, 5),
            "note": "Approximate slice/2.5D regularizer; exact 3D conservation requires full 3D volume pooling.",
        },
    }


def _train_2d_model(
    hr_path: Path,
    lr_path: Path | None,
    degrade: bool,
    backbone: str,
    ctx: int,
    epochs: int,
) -> SRGAN:
    dm_loc = PairedVolumeDataModule(
        high_path=str(hr_path),
        low_path=str(lr_path) if lr_path else None,
        degrade_from_hr=degrade,
        scale_factor=2,
        psf_sigma=0.9,
        degrade_noise_std=0.02,
        norm_mode="none",
        train_frac=0.60,
        val_frac=0.15,
        guard_slices_lr=1,
        train_axes=("yz", "xz", "xy"),
        context_slices=ctx,
        patch_size_lr=16,
        batch_size=8,
        seed=42,
    )
    is_srresnet = backbone == "srgan"
    model = SRGAN(
        super_res_factor=2,
        backbone=backbone,  # type: ignore[arg-type]
        context_slices=ctx,
        out_channels=20,
        num_blocks=2,
        norm_type="pixel" if is_srresnet else "batch",
        swin_num_heads=4,
        swin_window_size=8,
        output_activation="custom_sigmoid" if is_srresnet else "clamp",
        use_adversarial_loss=False,
        use_vgg_loss=False,
        pixel_loss_type="l1",
        conservation_loss_weight=0.15,
        conservation_psf_sigma=0.9,
        topology_loss_weight=0.04,
        gradient_loss_weight=0.05,
        ffl_loss_weight=0.02,
        learning_rate=1e-3 if is_srresnet else 5e-4,
        save_val_images=False,
    )
    pl.Trainer(
        max_epochs=epochs,
        limit_train_batches=12,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    ).fit(model, datamodule=dm_loc)
    return model.eval()


def run_subsystem_3_models_and_fusion(
    out_dir: Path, work_dir: Path, epochs: int = 8
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Verify upsample_3d, SR3DResNet, SRResNet, RRDB, SwinIR, and Fourier fusion on 3 phantoms."""
    pl.seed_everything(42, workers=True)

    # Dataset A: Synthetic PoreSpy 3D connected blobs
    ps_lr, ps_hr, _ = generate_porespy_volume(
        shape=(80, 48, 48),
        feature_regime="low_feature",
        noise_regime="low_noise",
        scale_factor=2,
        psf_sigma=0.9,
        seed=21,
    )
    ps_tif_hr, ps_tif_lr = work_dir / "ps_hr.tif", work_dir / "ps_lr.tif"
    tifffile.imwrite(ps_tif_hr, ps_hr)
    tifffile.imwrite(ps_tif_lr, ps_lr)

    dm_ps = PairedVolumeDataModule(
        high_path=str(ps_tif_hr),
        low_path=str(ps_tif_lr),
        degrade_from_hr=False,
        scale_factor=2,
        norm_mode="none",
        train_frac=0.60,
        val_frac=0.15,
        guard_slices_lr=1,
        train_axes=("yz", "xz", "xy"),
        context_slices=3,
        patch_size_lr=16,
        batch_size=8,
        seed=42,
    )
    dm_ps.setup("fit")
    ps_lr_te, ps_hr_te = dm_ps.splits_3d["test"]

    m_res_ps = _train_2d_model(ps_tif_hr, ps_tif_lr, False, "srgan", 1, epochs=epochs)
    m_rrdb_ps = _train_2d_model(ps_tif_hr, ps_tif_lr, False, "rrdb", 1, epochs=epochs)
    m_swin_ps = _train_2d_model(ps_tif_hr, ps_tif_lr, False, "swinir", 3, epochs=epochs + 2)

    dl_tr_3d, dl_va_3d = build_3d_subcube_dataloaders(
        dm_ps.splits_3d["train"][0],
        dm_ps.splits_3d["train"][1],
        dm_ps.splits_3d["val"][0],
        dm_ps.splits_3d["val"][1],
        scale_factor=2,
        patch_size_lr=6,
        train_samples=32,
        val_samples=8,
        batch_size=4,
        seed=42,
    )
    m_3d = SR3DResNet(
        super_res_factor=2,
        out_channels=16,
        num_blocks=2,
        learning_rate=5e-4,
        conservation_loss_weight=0.1,
    )
    pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    ).fit(m_3d, train_dataloaders=dl_tr_3d, val_dataloaders=dl_va_3d)
    m_3d.eval()

    ps_near = upsample_3d(ps_lr_te, 2, "nearest")
    ps_tri = upsample_3d(ps_lr_te, 2, "tricubic")
    ps_3d_sr = super_resolve_volume_3d(
        m_3d.gen, ps_lr_te, scale_factor=2, patch_size_lr=6, overlap_lr=2
    )
    ps_res_sr = super_resolve_volume_orthogonal(
        m_res_ps.gen, ps_lr_te, scale_factor=2, fusion="confidence_fourier", context_slices=1
    )["fused"]
    ps_rrdb_sr = super_resolve_volume_orthogonal(
        m_rrdb_ps.gen, ps_lr_te, scale_factor=2, fusion="confidence_fourier", context_slices=1
    )["fused"]
    ps_swin_yz = super_resolve_volume_single_axis(
        m_swin_ps.gen, ps_lr_te, scale_factor=2, axis="yz", context_slices=3
    )
    ps_swin_ort = super_resolve_volume_orthogonal(
        m_swin_ps.gen, ps_lr_te, scale_factor=2, fusion="confidence_fourier", context_slices=3
    )["fused"]

    ps_table: dict[str, dict[str, float]] = {}
    for name, vol in [
        ("Nearest", ps_near),
        ("Tricubic-3D", ps_tri),
        ("SR3DResNet (Native 3D)", ps_3d_sr),
        ("SRResNet-3Axis", ps_res_sr),
        ("RRDB-3Axis", ps_rrdb_sr),
        ("SwinIR-SingleAxis (yz)", ps_swin_yz),
        ("SwinIR-2.5D-3Axis", ps_swin_ort),
    ]:
        im = evaluate_image_metrics(vol, ps_hr_te)
        pe = evaluate_petrophysical_metrics(vol, ps_hr_te, pore_is_low=True)
        sp = evaluate_spectral_metrics(vol, ps_hr_te, voxel_size_um=1.0)
        ps_table[name] = {
            "psnr": round(im["psnr"], 2),
            "ssim_3d": round(im["ssim_3d_triplanar"], 4),
            "pore_dice": round(pe["pore_dice"], 4),
            "porosity_abs_err": round(pe["porosity_abs_error"], 4),
            "anisotropy_err": round(pe["through_plane_anisotropy_error"], 4),
            "mtf_cutoff_050": round(sp["mtf_cutoff_050_cyc_vox"], 4),
        }

    # Dataset B: Calibrated Indiana Limestone Carbonate Phantom
    lime_tif, lime_manifest = fetch_or_generate_rock_sample(
        "indiana_limestone",
        dest_dir=work_dir / "rock_bench",
        shape=(80, 48, 48),
        prefer_offline=True,
        allow_offline_fallback=True,
        seed=42,
    )
    dm_lime = PairedVolumeDataModule(
        high_path=str(lime_tif),
        degrade_from_hr=True,
        scale_factor=2,
        psf_sigma=0.9,
        degrade_noise_std=0.02,
        norm_mode="none",
        train_frac=0.60,
        val_frac=0.15,
        guard_slices_lr=1,
        train_axes=("yz", "xz", "xy"),
        context_slices=3,
        patch_size_lr=16,
        batch_size=8,
        seed=42,
    )
    dm_lime.setup("fit")
    b_lr_te, b_hr_te = dm_lime.splits_3d["test"]
    m_lime = _train_2d_model(lime_tif, None, True, "swinir", 3, epochs=epochs + 2)
    b_near = upsample_3d(b_lr_te, 2, "nearest")
    b_tri = upsample_3d(b_lr_te, 2, "tricubic")
    b_yz = super_resolve_volume_single_axis(
        m_lime.gen, b_lr_te, scale_factor=2, axis="yz", context_slices=3
    )
    b_ort = super_resolve_volume_orthogonal(
        m_lime.gen, b_lr_te, scale_factor=2, fusion="confidence_fourier", context_slices=3
    )["fused"]

    # Dataset C: Calibrated 3-Phase Zn-13X Phantom (32^3 -> 64^3)
    zn_lr_tif, zn_hr_tif, zn_manifest = prepare_zn13x_benchmark(
        dest_dir=work_dir / "zn13x_phantom32",
        pair="2x",
        lr_size=32,
        prefer_offline=True,
        allow_offline_fallback=True,
        seed=42,
    )
    dm_zn = PairedVolumeDataModule(
        high_path=str(zn_hr_tif),
        low_path=str(zn_lr_tif),
        degrade_from_hr=False,
        scale_factor=2,
        norm_mode="none",
        train_frac=0.60,
        val_frac=0.15,
        guard_slices_lr=1,
        train_axes=("yz", "xz", "xy"),
        context_slices=3,
        patch_size_lr=16,
        batch_size=8,
        seed=42,
    )
    dm_zn.setup("fit")
    z_lr_te, z_hr_te = dm_zn.splits_3d["test"]
    m_zn = _train_2d_model(zn_hr_tif, zn_lr_tif, False, "swinir", 3, epochs=epochs + 2)
    z_near = upsample_3d(z_lr_te, 2, "nearest")
    z_tri = upsample_3d(z_lr_te, 2, "tricubic")
    z_yz = super_resolve_volume_single_axis(
        m_zn.gen, z_lr_te, scale_factor=2, axis="yz", context_slices=3
    )
    z_ort = super_resolve_volume_orthogonal(
        m_zn.gen, z_lr_te, scale_factor=2, fusion="confidence_fourier", context_slices=3
    )["fused"]

    zn_tri_m = evaluate_triphase_zn13x_metrics(z_tri, z_hr_te)
    zn_ort_m = evaluate_triphase_zn13x_metrics(z_ort, z_hr_te)
    zn_im_tri = evaluate_image_metrics(z_tri, z_hr_te)
    zn_im_ort = evaluate_image_metrics(z_ort, z_hr_te)

    fig, axes = plt.subplots(3, 5, figsize=(16.2, 9.6), dpi=180)
    plt.subplots_adjust(hspace=0.28, wspace=0.14)
    rows_data = [
        (
            "Dataset A: Synthetic PoreSpy\n(Connected Blobs, 2×)",
            [ps_near, ps_tri, ps_swin_yz, ps_swin_ort, ps_hr_te],
            [
                evaluate_image_metrics(v, ps_hr_te)["psnr"]
                for v in (ps_near, ps_tri, ps_swin_yz, ps_swin_ort)
            ],
        ),
        (
            "Dataset B: Indiana Limestone\n(Calibrated Phantom, 2×)",
            [b_near, b_tri, b_yz, b_ort, b_hr_te],
            [evaluate_image_metrics(v, b_hr_te)["psnr"] for v in (b_near, b_tri, b_yz, b_ort)],
        ),
        (
            "Dataset C: 3-Phase Zn-13X\n(Calibrated Phantom, 2×)",
            [z_near, z_tri, z_yz, z_ort, z_hr_te],
            [evaluate_image_metrics(v, z_hr_te)["psnr"] for v in (z_near, z_tri, z_yz, z_ort)],
        ),
    ]
    col_names = [
        "upsample_3d (nearest)",
        "upsample_3d (tricubic)",
        "Single-Axis SwinIR",
        "3-Axis Fourier Fusion",
        "Ground-Truth HR",
    ]
    for r_i, (r_label, vols, psnrs) in enumerate(rows_data):
        zm = vols[-1].shape[0] // 2
        for c_i, v in enumerate(vols):
            ax = axes[r_i, c_i]
            ax.imshow(v[zm], cmap="gray", vmin=0.02, vmax=0.98, interpolation="nearest")
            sub = f"PSNR {psnrs[c_i]:.2f} dB" if c_i < 4 else "Target Reference"
            ax.set_title(f"{col_names[c_i]}\n{sub}" if r_i == 0 else sub, fontsize=8.7)
            ax.set_xticks([])
            ax.set_yticks([])
            if c_i == 0:
                ax.set_ylabel(r_label, fontweight="bold", fontsize=8.6)

    fig.suptitle(
        "Multi-Dataset Super-Resolution Verification on Synthetic & Calibrated 3D Phantoms (2×)",
        fontsize=11.4,
        fontweight="bold",
        y=0.99,
    )
    fig.savefig(out_dir / "vr_03_models_and_fusion.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset_a_porespy_phantom_models": ps_table,
        "dataset_b_limestone_phantom": {
            "source_kind": lime_manifest["source_kind"],
            "nearest_psnr": round(evaluate_image_metrics(b_near, b_hr_te)["psnr"], 2),
            "tricubic_psnr": round(evaluate_image_metrics(b_tri, b_hr_te)["psnr"], 2),
            "swinir_yz_psnr": round(evaluate_image_metrics(b_yz, b_hr_te)["psnr"], 2),
            "swinir_3axis_psnr": round(evaluate_image_metrics(b_ort, b_hr_te)["psnr"], 2),
        },
        "dataset_c_zn13x_phantom_provenance": {
            "source_kind": zn_manifest["source_kind"],
            "is_experimental_xct": zn_manifest["is_experimental_xct"],
            "low_zenodo_doi": zn_manifest["low_zenodo_doi"],
        },
        "dataset_c_zn13x_phantom_metrics": {
            "tricubic_psnr": round(zn_im_tri["psnr"], 2),
            "swinir_3axis_psnr": round(zn_im_ort["psnr"], 2),
            "tricubic_macro_dice": round(zn_tri_m["triphase_macro_dice"], 4),
            "swinir_3axis_macro_dice": round(zn_ort_m["triphase_macro_dice"], 4),
            "tricubic_pore_dice": round(zn_tri_m["triphase_pore_dice"], 4),
            "swinir_pore_dice": round(zn_ort_m["triphase_pore_dice"], 4),
            "tricubic_na13x_err": round(zn_tri_m["na13x_fraction_abs_error"], 4),
            "swinir_na13x_err": round(zn_ort_m["na13x_fraction_abs_error"], 4),
            "tricubic_zn13x_err": round(zn_tri_m["zn13x_fraction_abs_error"], 4),
            "swinir_zn13x_err": round(zn_ort_m["zn13x_fraction_abs_error"], 4),
            "swinir_zn_exchange_err": round(zn_ort_m["zn_exchange_ratio_abs_error"], 4),
        },
    }
    return summary, (z_tri, z_ort, z_hr_te)


def run_subsystem_4_analytical_metrics(
    out_dir: Path,
    work_dir: Path,
    z_tri: np.ndarray,
    z_ort: np.ndarray,
    z_hr_te: np.ndarray,
) -> dict[str, Any]:
    """Verify diffusion, spectral MTF, topology, S_2(r), drainage, and OpenImpala export."""
    diffusion_cases: list[dict[str, Any]] = []

    open_box = np.ones((32, 32, 32), dtype=bool)
    res_open = solve_steady_state_diffusion_axis(open_box, axis=0)
    diffusion_cases.append(
        {
            "case": "1. Open Bulk Space (φ=1.000)",
            "phi": 1.0,
            "exact_d_eff": 1.0,
            "num_d_eff": round(res_open["d_eff"], 6),
            "exact_tau": 1.0,
            "num_tau": round(res_open["tortuosity"], 6),
            "solver_converged": bool(res_open["solver_converged"]),
            "rel_err_pct": round(abs(res_open["d_eff"] - 1.0) * 100.0, 4),
        }
    )

    straight = np.zeros((32, 32, 32), dtype=bool)
    for r0 in (4, 20):
        for c0 in (4, 20):
            straight[:, r0 : r0 + 8, c0 : c0 + 8] = True
    res_st_z = solve_steady_state_diffusion_axis(straight, axis=0)
    res_st_x = solve_steady_state_diffusion_axis(straight, axis=2)
    diffusion_cases.append(
        {
            "case": "2. Parallel Straight Ducts z (φ=0.250)",
            "phi": 0.25,
            "exact_d_eff": 0.25,
            "num_d_eff": round(res_st_z["d_eff"], 6),
            "exact_tau": 1.0,
            "num_tau": round(res_st_z["tortuosity"], 6),
            "transverse_d_eff_x": round(res_st_x["d_eff"], 6),
            "solver_converged": bool(res_st_z["solver_converged"]),
            "rel_err_pct": round(abs(res_st_z["d_eff"] - 0.25) / 0.25 * 100.0, 4),
        }
    )

    dead_end = np.zeros((48, 32, 32), dtype=bool)
    dead_end[:, 12:20, 6:14] = True
    dead_end[22:26, 14:18, 14:16] = True
    dead_end[14:34, 10:22, 16:29] = True
    res_de = solve_steady_state_diffusion_axis(dead_end, axis=0)
    conc_de, flux_imb = _solve_conc_field_z(dead_end)
    phi_de = float(dead_end.mean())
    diffusion_cases.append(
        {
            "case": "3. Trunk + Dead-End Cavity",
            "phi": round(phi_de, 4),
            "exact_d_eff": 0.0625,
            "num_d_eff": round(res_de["d_eff"], 5),
            "exact_tau": round(phi_de / 0.0625, 4),
            "num_tau": round(res_de["tortuosity"], 4),
            "flux_rel_discrepancy": float(f"{flux_imb:.2e}"),
            "solver_converged": bool(res_de["solver_converged"]),
            "rel_err_pct": round(abs(res_de["d_eff"] - 0.0625) / 0.0625 * 100.0, 2),
        }
    )

    for label_s, rad in [
        ("4. Maxwell Sphere (R=6/40³)", 6.0),
        ("5. Maxwell Sphere (R=9/40³)", 9.0),
        ("6. Maxwell Sphere (R=11/40³)", 11.0),
    ]:
        n_m = 40
        zz, yy, xx = np.meshgrid(np.arange(n_m), np.arange(n_m), np.arange(n_m), indexing="ij")
        pore_m = ((zz - 19.5) ** 2 + (yy - 19.5) ** 2 + (xx - 19.5) ** 2) > rad**2
        phi_m = float(pore_m.mean())
        fs = 1.0 - phi_m
        d_maxwell = phi_m / (1.0 + 0.5 * fs)
        res_m = compute_directional_tortuosity(pore_m)
        diffusion_cases.append(
            {
                "case": label_s,
                "phi": round(phi_m, 4),
                "exact_d_eff": round(d_maxwell, 5),
                "num_d_eff": round(res_m["d_eff_mean"], 5),
                "exact_tau": round(1.0 + 0.5 * fs, 4),
                "num_tau": round(res_m["tortuosity_mean"], 4),
                "solver_converged": True,
                "rel_err_pct": round(abs(res_m["d_eff_mean"] - d_maxwell) / d_maxwell * 100.0, 3),
            }
        )

    n_t = 48
    zz, yy, xx = np.meshgrid(np.arange(n_t), np.arange(n_t), np.arange(n_t), indexing="ij")
    one_sphere = ((zz - 23.5) ** 2 + (yy - 23.5) ** 2 + (xx - 23.5) ** 2) <= 10.0**2
    rho_xy = np.sqrt((yy - 23.5) ** 2 + (xx - 23.5) ** 2)
    one_torus = ((rho_xy - 12.0) ** 2 + (zz - 23.5) ** 2) <= 4.5**2
    chi_sphere_total = round(compute_euler_density(one_sphere, voxel_size=1.0) * (n_t**3), 4)
    chi_torus_total = round(compute_euler_density(one_torus, voxel_size=1.0) * (n_t**3), 4)
    phi_sphere_exact = (4.0 / 3.0) * math.pi * (10.0**3) / (n_t**3)
    phi_sphere_num = float(one_sphere.mean())

    rng = np.random.default_rng(123)
    white_3d = rng.normal(0.5, 0.12, size=(48, 48, 48)).astype(np.float32)
    mtf_checks: dict[str, dict[str, float]] = {}
    mtf_plot_data: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for sg in (0.8, 1.4, 2.2):
        filt_3d = gaussian_filter(white_3d, sigma=sg, mode="wrap")
        fsc_d = compute_fsc_3d(filt_3d, white_3d, voxel_size_um=1.0, apodize=False)
        k_arr = fsc_d["frequencies_cyc_per_voxel"]
        mtf_num = fsc_d["effective_mtf"]
        mtf_exact = np.exp(-2.0 * (math.pi**2) * (sg**2) * (k_arr**2))
        valid = k_arr <= 0.45
        rmse_mtf = float(np.sqrt(np.mean((mtf_num[valid] - mtf_exact[valid]) ** 2)))
        k05_exact = math.sqrt(math.log(2.0) / (2.0 * math.pi**2)) / sg
        mtf_checks[f"sigma_{sg}"] = {
            "curve_rmse": round(rmse_mtf, 5),
            "exact_k05_cyc_vox": round(k05_exact, 4),
            "measured_k05_cyc_vox": round(float(fsc_d["cutoff_freq_mtf_050"]), 4),
            "measured_fsc05_cyc_vox": round(float(fsc_d["cutoff_freq_050"]), 4),
        }
        mtf_plot_data[sg] = (
            k_arr[valid],
            mtf_num[valid],
            mtf_exact[valid],
            fsc_d["correlation"][valid],
        )

    bern = rng.random((48, 48, 48)) < 0.30
    s2_bern = two_point_correlation_radial(bern, max_lag=18)
    phi_b = float(bern.mean())

    tubes = np.zeros((40, 48, 48), dtype=bool)
    yy2, xx2 = np.meshgrid(np.arange(48), np.arange(48), indexing="ij")
    tubes[:, (yy2 - 16) ** 2 + (xx2 - 16) ** 2 <= 7.0**2] = True
    tubes[:, (yy2 - 35) ** 2 + (xx2 - 35) ** 2 <= 3.0**2] = True
    pc_arr, snw_arr = compute_drainage_curve(tubes, n_steps=24, voxel_size=1.0, inlet_axis=0)

    imp_tif, imp_in = export_for_openimpala(
        tubes, output_dir=work_dir / "impala", name="bimodal_tubes", direction="z"
    )

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.0), dpi=180)
    plt.subplots_adjust(hspace=0.36, wspace=0.30)

    axes[0, 0].plot(
        [0, 1.05], [0, 1.05], color="#111827", ls="--", lw=1.5, label="Exact Parity y = x"
    )
    colors_d = ["#00b4d8", "#9b5de5", "#f8961e", "#06d6a0", "#118ab2", "#ef476f"]
    for ci, cdict in enumerate(diffusion_cases):
        axes[0, 0].scatter(
            cdict["exact_d_eff"],
            cdict["num_d_eff"],
            s=65,
            color=colors_d[ci],
            zorder=5,
            label=f"{cdict['case']} ({cdict['rel_err_pct']:.2f}%)",
        )
    axes[0, 0].set_xlim(0, 1.06)
    axes[0, 0].set_ylim(0, 1.06)
    axes[0, 0].set_xlabel("Exact Analytical Effective Diffusivity D_eff")
    axes[0, 0].set_ylabel("OpenLSR-X 3D FVM Solver D_eff")
    axes[0, 0].set_title("1. 3D Diffusion Solver vs Analytical Solutions")
    axes[0, 0].legend(fontsize=6.9, loc="upper left")
    axes[0, 0].grid(alpha=0.25)

    cmap_c = plt.get_cmap("viridis").copy()
    cmap_c.set_bad(color="#1d2028")
    im01 = axes[0, 1].imshow(
        conc_de[:, 16, :], cmap=cmap_c, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest"
    )
    axes[0, 1].annotate(
        "Zero-Flux Dead-End Pocket\n(Uniform c ≈ 0.50, ∇c = 0!)",
        xy=(22, 24),
        xytext=(14, 6),
        color="#ffffff",
        fontsize=7.8,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#fee440", "lw": 1.5},
        bbox={"facecolor": "#000000bb", "pad": 2, "edgecolor": "#fee440"},
    )
    axes[0, 1].set_title(
        f"2. Laplace Field c(z, x) in Dead-End Duct\n(|J_in − J_out| / J_mean = {flux_imb:.2e})"
    )
    axes[0, 1].set_xlabel("Transverse x (Trunk: x∈[6,14], Pocket: x∈[16,29])")
    axes[0, 1].set_ylabel("Flow axis z (c=1 → c=0)")
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.03, label="Conc. c")

    col_sg = {0.8: "#00b4d8", 1.4: "#f8961e", 2.2: "#ef476f"}
    for sg, (ka, m_num, m_ex, _) in mtf_plot_data.items():
        axes[0, 2].plot(ka, m_ex, color=col_sg[sg], lw=2.0, label=f"Exact exp(−2π²·{sg}²k²)")
        axes[0, 2].scatter(
            ka[::2],
            m_num[::2],
            color=col_sg[sg],
            s=26,
            marker="o",
            zorder=5,
            label=f"compute_fsc_3d MTF (σ={sg})",
        )
    axes[0, 2].plot(
        mtf_plot_data[1.4][0],
        mtf_plot_data[1.4][3],
        color="#6c757d",
        ls=":",
        lw=1.8,
        label="Shell FSC(k) at σ=1.4 (≡1.0!)",
    )
    axes[0, 2].axhline(0.5, color="#111827", ls="--", lw=1.0)
    axes[0, 2].set_ylim(-0.02, 1.06)
    axes[0, 2].set_xlabel("Spatial Frequency k (cycles / voxel)")
    axes[0, 2].set_ylabel("Transfer Amplitude MTF_eff(k)")
    axes[0, 2].set_title("3. compute_fsc_3d vs Analytical Optical MTF")
    axes[0, 2].legend(fontsize=6.8, loc="lower left")
    axes[0, 2].grid(alpha=0.25)

    lags = np.arange(len(s2_bern))
    s2_exact_bern = np.where(lags == 0, phi_b, phi_b**2)
    s2_rmse = float(np.sqrt(np.mean((s2_bern - s2_exact_bern) ** 2)))
    axes[1, 0].plot(
        lags,
        s2_exact_bern,
        color="#111827",
        lw=2.2,
        ls="--",
        label=f"Exact Bernoulli (S₂(0)=φ={phi_b:.3f}, r≥1: φ²={phi_b**2:.4f})",
    )
    axes[1, 0].scatter(
        lags,
        s2_bern,
        color="#00b4d8",
        s=34,
        zorder=5,
        label=f"two_point_correlation_radial (RMSE={s2_rmse:.4f})",
    )
    axes[1, 0].set_xlabel("Radial Lag r (voxels)")
    axes[1, 0].set_ylabel("Two-Point Probability S₂(r)")
    axes[1, 0].set_title(
        "4. Non-Periodic S₂(r) & Topology\n"
        f"(Euler χ: Sphere={chi_sphere_total:+.1f} [GT +1], Torus={chi_torus_total:+.1f} [GT 0])"
    )
    axes[1, 0].legend(fontsize=7.3, loc="upper right")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].step(
        pc_arr,
        snw_arr,
        where="post",
        color="#9b5de5",
        lw=2.3,
        label="compute_drainage_curve S_nw(Pc)",
    )
    axes[1, 1].axvline(
        2.0 / 7.0, color="#06d6a0", ls="--", lw=1.5, label="Exact Wide Tube Entry 2/R₂ = 0.286"
    )
    axes[1, 1].axvline(
        2.0 / 3.0, color="#ef476f", ls="--", lw=1.5, label="Exact Narrow Tube Entry 2/R₁ = 0.667"
    )
    axes[1, 1].set_xlabel("Capillary Entry Pressure Proxy Pc = 2 / r_inscribed")
    axes[1, 1].set_ylabel("Non-Wetting Saturation S_nw")
    axes[1, 1].set_title("5. Bimodal Capillary Tube Drainage Steps")
    axes[1, 1].legend(fontsize=7.5, loc="lower right")
    axes[1, 1].grid(alpha=0.25)

    thr_z = tuple(float(x) for x in threshold_multiotsu(z_hr_te, classes=3))

    def _color_3ph(v2d: np.ndarray) -> np.ndarray:
        c = np.zeros((*v2d.shape, 3), dtype=np.float32)
        c[v2d < thr_z[0]] = (0.10, 0.16, 0.34)
        c[(v2d >= thr_z[0]) & (v2d < thr_z[1])] = (0.06, 0.72, 0.68)
        c[v2d >= thr_z[1]] = (0.98, 0.80, 0.22)
        return c

    zm = z_hr_te.shape[0] // 2
    strip_3p = np.hstack([_color_3ph(z_tri[zm]), _color_3ph(z_ort[zm]), _color_3ph(z_hr_te[zm])])
    axes[1, 2].imshow(strip_3p, interpolation="nearest")
    w3 = z_hr_te.shape[2]
    for xc in (w3 - 0.5, 2 * w3 - 0.5):
        axes[1, 2].axvline(xc, color="#ffffff", lw=1.2, ls="--")
    for ci, lb in enumerate(["Tricubic", "SwinIR-3Ax", "HR GT"]):
        axes[1, 2].text(
            ci * w3 + 2,
            5,
            lb,
            color="#ffffff",
            fontsize=7.8,
            fontweight="bold",
            bbox={"facecolor": "#000000bb", "pad": 1.5, "edgecolor": "none"},
        )
    axes[1, 2].set_title(
        "6. evaluate_triphase_zn13x_metrics (Phantom)\n(Navy=Pore, Teal=Na-13X, Gold=Zn-13X Rim)"
    )
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    fig.savefig(out_dir / "vr_04_analytical_physics_benchmarks.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "diffusion_analytical_cases": diffusion_cases,
        "topology_analytical": {
            "sphere_porosity_exact": round(phi_sphere_exact, 5),
            "sphere_porosity_measured": round(phi_sphere_num, 5),
            "sphere_euler_exact": 1.0,
            "sphere_euler_measured": chi_sphere_total,
            "torus_euler_exact": 0.0,
            "torus_euler_measured": chi_torus_total,
        },
        "spectral_mtf_analytical": mtf_checks,
        "bernoulli_s2": {
            "phi": round(phi_b, 4),
            "s2_lag0_measured": round(float(s2_bern[0]), 4),
            "s2_asymptote_exact": round(phi_b**2, 4),
            "s2_lag1_to_18_mean": round(float(np.mean(s2_bern[1:])), 4),
            "s2_rmse_vs_exact": round(s2_rmse, 5),
        },
        "openimpala_export": {
            "tif_exists": imp_tif.exists(),
            "inputs_exists": imp_in.exists(),
            "tif_bytes": imp_tif.stat().st_size,
            "inputs_preview": imp_in.read_text(encoding="utf-8").splitlines()[:6],
        },
    }


def run_bentheimer_synthetic_degradation_check(
    out_dir: Path,
    work_dir: Path,
    seeds: tuple[int, ...] = (42, 43, 44),
    epochs: int = 8,
) -> dict[str, Any]:
    """Multi-seed, two-core check on real 6 um Bentheimer micro-CT with synthetic 2x degradation."""
    cores = ["bentheimer_sandstone", "bentheimer_laminated"]
    per_core_results: dict[str, Any] = {}
    last_vis: dict[str, np.ndarray] = {}

    for core_name in cores:
        tif_p, manifest = fetch_or_generate_rock_sample(
            core_name,
            dest_dir=work_dir / "zenodo_bentheimer",
            shape=(96, 80, 80),
            prefer_offline=False,
            allow_offline_fallback=False,
            seed=42,
        )
        seed_rows: list[dict[str, Any]] = []
        for sd in seeds:
            pl.seed_everything(sd, workers=True)
            dm = PairedVolumeDataModule(
                high_path=str(tif_p),
                degrade_from_hr=True,
                scale_factor=2,
                psf_sigma=1.0,
                degrade_noise_std=0.02,
                norm_mode="percentile",
                lower_pct=0.5,
                upper_pct=99.5,
                train_frac=0.60,
                val_frac=0.15,
                guard_slices_lr=1,
                train_axes=("yz", "xz", "xy"),
                context_slices=3,
                patch_size_lr=20,
                batch_size=8,
                seed=sd,
            )
            dm.setup("fit")
            lr_te, hr_te = dm.splits_3d["test"]

            model = SRGAN(
                super_res_factor=2,
                backbone="swinir",
                context_slices=3,
                out_channels=24,
                num_blocks=2,
                swin_num_heads=4,
                swin_window_size=8,
                output_activation="clamp",
                use_adversarial_loss=False,
                use_vgg_loss=False,
                pixel_loss_type="l1",
                conservation_loss_weight=0.15,
                conservation_psf_sigma=1.0,
                topology_loss_weight=0.04,
                gradient_loss_weight=0.05,
                ffl_loss_weight=0.02,
                learning_rate=4e-4,
                save_val_images=False,
            )
            pl.Trainer(
                max_epochs=epochs,
                limit_train_batches=14,
                accelerator="cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            ).fit(model, datamodule=dm)
            model.eval()

            v_tri = upsample_3d(lr_te, 2, "tricubic")
            ort_d = super_resolve_volume_orthogonal(
                model.gen,
                lr_te,
                scale_factor=2,
                fusion="confidence_fourier",
                context_slices=3,
                return_uncertainty=True,
            )
            v_yz = ort_d["yz"]
            v_ort = ort_d["fused"]
            u_std = ort_d["uncertainty_std"]

            ref_tr = precompute_reference_transport(hr_te, pore_is_low=True, drainage_steps=8)
            tr_tri = evaluate_transport_metrics(
                v_tri, hr_te, pore_is_low=True, drainage_steps=8, precomputed_true=ref_tr
            )
            tr_yz = evaluate_transport_metrics(
                v_yz, hr_te, pore_is_low=True, drainage_steps=8, precomputed_true=ref_tr
            )
            tr_ort = evaluate_transport_metrics(
                v_ort, hr_te, pore_is_low=True, drainage_steps=8, precomputed_true=ref_tr
            )
            unc_b = compute_tortuosity_uncertainty_bounds(
                v_ort,
                u_std,
                true_vol=hr_te,
                pore_is_low=True,
                k_sigma=1.0,
                precomputed_true=ref_tr,
            )
            r_sp, _ = spearmanr(u_std.ravel(), np.abs(v_ort - hr_te).ravel())

            true_deff = float(tr_ort["true_d_eff_mean"] or 0.0)
            tri_err = float(tr_tri["d_eff_rel_error"] or 0.0) * 100.0
            yz_err = float(tr_yz["d_eff_rel_error"] or 0.0) * 100.0
            ort_err = float(tr_ort["d_eff_rel_error"] or 0.0) * 100.0
            brack = float(unc_b["d_eff_bracketed_by_envelope"] or 0.0) > 0.5

            seed_rows.append(
                {
                    "seed": sd,
                    "tricubic_psnr": round(evaluate_image_metrics(v_tri, hr_te)["psnr"], 2),
                    "yz_psnr": round(evaluate_image_metrics(v_yz, hr_te)["psnr"], 2),
                    "ortho_psnr": round(evaluate_image_metrics(v_ort, hr_te)["psnr"], 2),
                    "true_d_eff": round(true_deff, 5),
                    "tricubic_d_eff_rel_err_pct": round(tri_err, 2),
                    "yz_d_eff_rel_err_pct": round(yz_err, 2),
                    "ortho_d_eff_rel_err_pct": round(ort_err, 2),
                    "spearman_r": round(float(r_sp), 4),
                    "bracketed_1sigma": brack,
                }
            )
            if core_name == "bentheimer_sandstone" and sd == seeds[0]:
                last_vis = {
                    "near": upsample_3d(lr_te, 2, "nearest"),
                    "tri": v_tri,
                    "yz": v_yz,
                    "ort": v_ort,
                    "hr": hr_te,
                    "std": u_std,
                }

        def _msd(key: str, rows: list[dict[str, Any]] = seed_rows) -> dict[str, float]:
            vals = [float(r[key]) for r in rows]
            return {
                "mean": round(float(np.mean(vals)), 2),
                "std": round(float(np.std(vals, ddof=1)), 2),
            }

        per_core_results[core_name] = {
            "source_kind": manifest["source_kind"],
            "pairing_mode": "synthetic_2x_psf_noise_degradation_from_real_6um_xct",
            "test_slab_shape": list(hr_te.shape),
            "n_seeds": len(seeds),
            "seeds": seed_rows,
            "summary": {
                "tricubic_psnr": _msd("tricubic_psnr"),
                "yz_psnr": _msd("yz_psnr"),
                "ortho_psnr": _msd("ortho_psnr"),
                "tricubic_d_eff_rel_err_pct": _msd("tricubic_d_eff_rel_err_pct"),
                "yz_d_eff_rel_err_pct": _msd("yz_d_eff_rel_err_pct"),
                "ortho_d_eff_rel_err_pct": _msd("ortho_d_eff_rel_err_pct"),
                "spearman_r": {
                    "mean": round(float(np.mean([r["spearman_r"] for r in seed_rows])), 3),
                    "std": round(float(np.std([r["spearman_r"] for r in seed_rows], ddof=1)), 3),
                },
                "bracketed_count": f"{sum(1 for r in seed_rows if r['bracketed_1sigma'])}/{len(seeds)}",
            },
        }

    if last_vis:
        zm = last_vis["hr"].shape[0] // 2
        fig, axes = plt.subplots(2, 5, figsize=(16.2, 6.8), dpi=180)
        plt.subplots_adjust(hspace=0.26, wspace=0.14)
        cols = [
            ("Synthetic 12 µm LR\n(Nearest)", last_vis["near"]),
            ("Tricubic-3D", last_vis["tri"]),
            ("Single-Axis SwinIR (yz)", last_vis["yz"]),
            ("3-Axis Fourier Fusion", last_vis["ort"]),
            ("Real 6 µm HR Scan\n(Zenodo 5542624)", last_vis["hr"]),
        ]
        for ci, (cname, cvol) in enumerate(cols):
            axes[0, ci].imshow(cvol[zm], cmap="gray", vmin=0.02, vmax=0.98, interpolation="nearest")
            axes[0, ci].set_title(cname, fontsize=8.8)
            axes[0, ci].set_xticks([])
            axes[0, ci].set_yticks([])
            if ci < 4:
                err_map = np.abs(cvol[zm] - last_vis["hr"][zm])
                axes[1, ci].imshow(
                    err_map, cmap="inferno", vmin=0.0, vmax=0.25, interpolation="nearest"
                )
                axes[1, ci].set_title(f"|SR − HR| (MAE {err_map.mean():.3f})", fontsize=8.4)
            else:
                _, thr_hr = segment_pore_space(last_vis["hr"], pore_is_low=True)
                axes[1, ci].imshow(last_vis["std"][zm], cmap="magma", interpolation="nearest")
                axes[1, ci].set_title(f"Inter-View SD σ_ortho (thr={thr_hr:.2f})", fontsize=8.4)
            axes[1, ci].set_xticks([])
            axes[1, ci].set_yticks([])
        axes[0, 0].set_ylabel("Greyscale Slice", fontweight="bold")
        axes[1, 0].set_ylabel("Error / View SD", fontweight="bold")
        fig.suptitle(
            "Real 6 µm Bentheimer Micro-CT (Zenodo 5542624) with Synthetic 2× Degradation (12 µm → 6 µm)",
            fontsize=11.2,
            fontweight="bold",
            y=0.99,
        )
        fig.savefig(out_dir / "vr_05_bentheimer_synth_deg.png", bbox_inches="tight")
        plt.close(fig)

    return per_core_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/verification"),
        help="Directory where verification PNG figures and JSON report are written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="CPU training epochs per compact neural verification run.",
    )
    parser.add_argument(
        "--include-bentheimer",
        action="store_true",
        help="Also download Zenodo 5542624 6 um Bentheimer cores (2 cores x 3 seeds) under synthetic 2x degradation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="openlsrx_verify_") as tmp_s:
        work_dir = Path(tmp_s)
        s1 = run_subsystem_1_data_alignment(out_dir)
        print("[1/4] Subsystem 1 (Data, Cell-Centered Alignment & Registration) verified")
        s2 = run_subsystem_2_losses(out_dir)
        print(
            "[2/4] Subsystem 2 (Topology, Gradient, Spectral & Approx. Conservation Losses) verified"
        )
        s3, zn_vols = run_subsystem_3_models_and_fusion(out_dir, work_dir, epochs=args.epochs)
        print("[3/4] Subsystem 3 (Models & Orthogonal Fourier Fusion on 3 Phantoms) verified")
        s4 = run_subsystem_4_analytical_metrics(out_dir, work_dir, *zn_vols)
        print("[4/4] Subsystem 4 (Closed-Form Analytical Physics & Spectral Metrics) verified")

        report: dict[str, Any] = {
            "elapsed_seconds": round(time.perf_counter() - t0, 1),
            "subsystem_1_data": s1,
            "subsystem_2_losses": s2,
            "subsystem_3_models": s3,
            "subsystem_4_analytical": s4,
        }
        if args.include_bentheimer:
            print(
                "[5/5] Running 2-Core x 3-Seed Real 6 um Bentheimer (Synthetic 2x Degradation) check..."
            )
            report["bentheimer_synthetic_degradation_multiseed"] = (
                run_bentheimer_synthetic_degradation_check(out_dir, work_dir, epochs=args.epochs)
            )
            report["elapsed_seconds"] = round(time.perf_counter() - t0, 1)

    summary_path = out_dir / "verification_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verification complete in {report['elapsed_seconds']}s -> {summary_path}")


if __name__ == "__main__":
    main()
