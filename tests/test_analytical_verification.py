"""Closed-form analytical accuracy verification suite for OpenLSR-X subsystems.

Verifies alignment, sub-voxel registration, provenance guards, topological/conservation
losses, 3D Fickian diffusion, optical MTF, 3D Euler characteristic, non-periodic S_2(r),
and bimodal capillary drainage against exact mathematical ground-truth solutions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom as nd_zoom

from openlsrx.cli import main as cli_main
from openlsrx.data import (
    AVAILABLE_DUALSCAN_DATASETS,
    PairedVolumeDataModule,
    apply_cross_scan_calibration,
    degrade_xct_volume,
    fetch_or_generate_rock_sample,
    fit_cross_scan_calibration,
    prepare_bentheimer_dualscan_benchmark,
    prepare_zn13x_benchmark,
    register_multires_pair_3d,
    synthesize_zn13x_phantom_pair,
    zoom_cell_centered,
)
from openlsrx.inference import super_resolve_volume_orthogonal
from openlsrx.losses import (
    AttenuationConservationLoss,
    SoftCLDiceLoss,
)
from openlsrx.metrics import (
    compute_directional_tortuosity,
    compute_drainage_curve,
    compute_euler_density,
    compute_fsc_3d,
    solve_steady_state_diffusion_axis,
    two_point_correlation_radial,
)
from openlsrx.models import RRDBGenerator, SR3DResNet, SwinIRGenerator
from openlsrx.models.rrdb import ArbitraryScaleUpsample


def test_zoom_cell_centered_eliminates_depth_shift_2x_4x_8x() -> None:
    """Cell-centered zoom must keep interior coordinate shift < 1e-4 HR voxels at 2x, 4x, and 8x."""
    n_hr = 64
    zz, _, _ = np.meshgrid(np.arange(n_hr), np.arange(n_hr), np.arange(n_hr), indexing="ij")
    ramp_hr = zz.astype(np.float32) / float(n_hr - 1)

    expected_legacy_min = {2: 0.40, 4: 1.20, 8: 2.50}
    for s, leg_floor in expected_legacy_min.items():
        lr_blk = ramp_hr.reshape(n_hr // s, s, n_hr // s, s, n_hr // s, s).mean(axis=(1, 3, 5))
        up_legacy = nd_zoom(lr_blk, zoom=s, order=1, grid_mode=False)
        up_centered = zoom_cell_centered(lr_blk, zoom_factors=s, order=1)

        interior = slice(s, n_hr - s)
        leg_shift = float(np.max(np.abs(up_legacy[interior] - ramp_hr[interior])) * (n_hr - 1))
        cen_shift = float(np.max(np.abs(up_centered[interior] - ramp_hr[interior])) * (n_hr - 1))

        assert leg_shift > leg_floor
        assert cen_shift < 1e-4


def test_register_multires_recovers_exact_half_lr_voxel_shift() -> None:
    """Sub-voxel HR registration must recover half-LR-voxel 3D offsets (+3, -2, +4) HR voxels at 2x."""
    zn_lr, zn_hr = synthesize_zn13x_phantom_pair(pair="2x", lr_shape=(28, 28, 28), seed=7)
    injected_hr_shift = (3, -2, 4)  # (+1.5, -1.0, +2.0) LR voxels
    hr_shifted = np.roll(zn_hr, shift=injected_hr_shift, axis=(0, 1, 2))[6:50, 6:50, 6:50]
    lr_ref = zn_lr[3:25, 3:25, 3:25]

    _, _, meta = register_multires_pair_3d(
        lr_ref, hr_shifted, scale_factor=2, local_refine_radius=3
    )
    assert np.asarray(meta["hr_shift_zyx"], dtype=int).tolist() == [-3, 2, -4]
    assert float(meta["hr_triplanar_ssim"]) > float(meta["initial_triplanar_ssim"]) + 0.40  # type: ignore[arg-type]


def test_offline_provenance_never_spoofs_experimental_dois(tmp_path: Path) -> None:
    """Offline calibrated phantoms must carry explicit phantom provenance and null Zenodo DOIs."""
    _, _, zn_manifest = prepare_zn13x_benchmark(
        dest_dir=tmp_path / "zn",
        pair="2x",
        lr_size=16,
        prefer_offline=True,
        allow_offline_fallback=True,
        seed=42,
    )
    assert zn_manifest["source_kind"] == "offline_calibrated_zn13x_phantom"
    assert zn_manifest["is_experimental_xct"] is False
    assert zn_manifest["low_zenodo_doi"] is None
    assert zn_manifest["high_zenodo_doi"] is None

    _, rock_manifest = fetch_or_generate_rock_sample(
        "indiana_limestone",
        dest_dir=tmp_path / "rock",
        shape=(32, 32, 32),
        prefer_offline=True,
        allow_offline_fallback=True,
        seed=42,
    )
    assert rock_manifest["source_kind"] == "calibrated_carbonate_phantom"


def test_soft_cldice_detects_severed_throat_missed_by_l1() -> None:
    """Severing a narrow throat (0.56% of pixels) must yield lower L1 than global noise but >30x SoftCLDice."""
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
    pred_pinched[(yy >= 29) & (yy <= 35) & (xx >= 29) & (xx <= 35)] = 0.90

    t_hr = torch.from_numpy(hr_clean)[None, None]
    t_noisy = torch.from_numpy(pred_noisy)[None, None]
    t_pinch = torch.from_numpy(pred_pinched)[None, None]

    cldice_fn = SoftCLDiceLoss(num_iter=8, temperature=20.0, pore_is_low=True, threshold=0.50)
    l1_noisy = float(torch.nn.functional.l1_loss(t_noisy, t_hr))
    l1_pinch = float(torch.nn.functional.l1_loss(t_pinch, t_hr))
    cl_noisy = float(cldice_fn(t_noisy, t_hr))
    cl_pinch = float(cldice_fn(t_pinch, t_hr))

    assert l1_pinch < l1_noisy
    assert cl_pinch > 30.0 * cl_noisy


def test_approximate_conservation_loss_bias_and_25d_curvature_reduction() -> None:
    """Approximate slice/slab conservation loss must vanish at zero bias and reduce 2.5D quadratic z-PSF error."""
    t_hr = torch.full((1, 1, 32, 32), 0.45, dtype=torch.float32)
    t_lr = torch.nn.functional.interpolate(t_hr, size=(16, 16), mode="area")
    cons0 = AttenuationConservationLoss(psf_sigma=0.0, reduction_norm="l1")
    assert float(cons0(t_hr, t_lr)) == 0.0
    assert abs(float(cons0(t_hr + 0.10, t_lr)) - 0.10) < 1e-6

    # Smooth quadratic z-profile: 2.5D discrete z-Laplacian reduces but does not eliminate 3D PSF discrepancy
    zz = np.arange(6, dtype=np.float32)[:, None, None] / 2.0 - 1.0
    vol_quad = np.broadcast_to(0.45 + 0.12 * (zz**2), (6, 32, 32)).copy()
    lr_quad = (
        gaussian_filter(vol_quad, sigma=1.0, mode="nearest")
        .reshape(3, 2, 16, 2, 16, 2)
        .mean(axis=(1, 3, 5))
    )
    cons_psf = AttenuationConservationLoss(psf_sigma=1.0, reduction_norm="l1")
    err_2d = float(
        cons_psf(torch.from_numpy(vol_quad[2:3])[None], torch.from_numpy(lr_quad[1:2])[None])
    )
    err_25d = float(
        cons_psf(torch.from_numpy(vol_quad[2:3])[None], torch.from_numpy(lr_quad[0:3])[None])
    )
    assert 0.0 < err_25d < 0.65 * err_2d


def test_diffusion_solver_exact_analytical_geometries() -> None:
    """3D FVM diffusion solver must match exact solutions across open space, ducts, dead ends, and Maxwell spheres."""
    # 1. Open Bulk Space (phi = 1.0 -> D_eff = 1.0, tau = 1.0)
    res_open = solve_steady_state_diffusion_axis(np.ones((24, 24, 24), dtype=bool), axis=0)
    assert res_open["solver_converged"] == 1.0
    assert abs(res_open["d_eff"] - 1.0) < 1e-4
    assert abs(res_open["tortuosity"] - 1.0) < 1e-4

    # 2. Parallel Straight Ducts along z (phi = 0.25 -> D_eff_z = 0.25, D_eff_x = 0.0)
    ducts = np.zeros((24, 32, 32), dtype=bool)
    for r0 in (4, 20):
        for c0 in (4, 20):
            ducts[:, r0 : r0 + 8, c0 : c0 + 8] = True
    res_z = solve_steady_state_diffusion_axis(ducts, axis=0)
    res_x = solve_steady_state_diffusion_axis(ducts, axis=2)
    assert res_z["solver_converged"] == 1.0
    assert abs(res_z["d_eff"] - 0.25) < 1e-4
    assert res_x["d_eff"] == 0.0

    # 3. Straight Trunk (phi_trunk = 0.0625) + Large Zero-Flux Dead-End Pocket (phi_tot = 0.1266)
    dead_end = np.zeros((48, 32, 32), dtype=bool)
    dead_end[:, 12:20, 6:14] = True
    dead_end[22:26, 14:18, 14:16] = True
    dead_end[14:34, 10:22, 16:29] = True
    res_de = solve_steady_state_diffusion_axis(dead_end, axis=0)
    assert res_de["solver_converged"] == 1.0
    assert abs(res_de["d_eff"] - 0.0625) / 0.0625 < 0.01

    # 4. Maxwell Dilute Insulating Sphere Suspension: D_eff = phi / (1 + 0.5*(1 - phi))
    n_m = 40
    zz, yy, xx = np.meshgrid(np.arange(n_m), np.arange(n_m), np.arange(n_m), indexing="ij")
    for rad in (6.0, 9.0, 11.0):
        pore_m = ((zz - 19.5) ** 2 + (yy - 19.5) ** 2 + (xx - 19.5) ** 2) > rad**2
        phi_m = float(pore_m.mean())
        d_maxwell = phi_m / (1.0 + 0.5 * (1.0 - phi_m))
        res_m = compute_directional_tortuosity(pore_m)
        assert abs(res_m["d_eff_mean"] - d_maxwell) / d_maxwell < 0.01


def test_fsc_3d_effective_mtf_matches_analytical_gaussian_otf() -> None:
    """Effective MTF must recover exact Gaussian k_0.5 = sqrt(ln(2)/2)/(pi*sigma) while shell FSC stays ~1."""
    rng = np.random.default_rng(123)
    white_3d = rng.normal(0.5, 0.12, size=(48, 48, 48)).astype(np.float32)

    for sg in (0.8, 1.4, 2.2):
        filt_3d = gaussian_filter(white_3d, sigma=sg, mode="wrap")
        fsc_d = compute_fsc_3d(filt_3d, white_3d, voxel_size_um=1.0, apodize=False)
        k_arr = fsc_d["frequencies_cyc_per_voxel"]
        mtf_exact = np.exp(-2.0 * (math.pi**2) * (sg**2) * (k_arr**2))
        valid = k_arr <= 0.45
        rmse = float(np.sqrt(np.mean((fsc_d["effective_mtf"][valid] - mtf_exact[valid]) ** 2)))
        k05_exact = math.sqrt(math.log(2.0) / (2.0 * math.pi**2)) / sg
        k05_meas = float(fsc_d["cutoff_freq_mtf_050"])

        assert rmse < 0.015
        assert abs(k05_meas - k05_exact) / k05_exact < 0.015
    # Under noiseless positive scalar blur (sigma=1.4), shell FSC remains near unity at low/mid frequencies
    fsc_14 = compute_fsc_3d(
        gaussian_filter(white_3d, sigma=1.4, mode="wrap"),
        white_3d,
        voxel_size_um=1.0,
        apodize=False,
    )
    assert float(fsc_14["cutoff_freq_050"]) >= 0.45


def test_euler_characteristic_and_nonperiodic_s2_analytical() -> None:
    """3D Euler characteristic must equal +1 (ball) and 0 (torus); Bernoulli S_2(r) must match phi and phi^2."""
    n_t = 48
    zz, yy, xx = np.meshgrid(np.arange(n_t), np.arange(n_t), np.arange(n_t), indexing="ij")
    ball = ((zz - 23.5) ** 2 + (yy - 23.5) ** 2 + (xx - 23.5) ** 2) <= 10.0**2
    rho = np.sqrt((yy - 23.5) ** 2 + (xx - 23.5) ** 2)
    torus = ((rho - 12.0) ** 2 + (zz - 23.5) ** 2) <= 4.5**2

    assert round(compute_euler_density(ball, voxel_size=1.0) * (n_t**3), 4) == 1.0
    assert round(compute_euler_density(torus, voxel_size=1.0) * (n_t**3), 4) == 0.0

    rng = np.random.default_rng(77)
    bern = rng.random((48, 48, 48)) < 0.30
    phi_b = float(bern.mean())
    s2 = two_point_correlation_radial(bern, max_lag=16)
    s2_exact = np.where(np.arange(len(s2)) == 0, phi_b, phi_b**2)
    assert abs(float(s2[0]) - phi_b) < 1e-6
    assert float(np.sqrt(np.mean((s2 - s2_exact) ** 2))) < 2e-3


def test_bimodal_capillary_tube_drainage_steps() -> None:
    """Morphological drainage on parallel tubes R1=3, R2=7 must jump at Young-Laplace 2/R2 and 2/R1."""
    tubes = np.zeros((36, 48, 48), dtype=bool)
    yy, xx = np.meshgrid(np.arange(48), np.arange(48), indexing="ij")
    tubes[:, (yy - 16) ** 2 + (xx - 16) ** 2 <= 7.0**2] = True
    tubes[:, (yy - 35) ** 2 + (xx - 35) ** 2 <= 3.0**2] = True

    pc, snw = compute_drainage_curve(tubes, n_steps=24, voxel_size=1.0, inlet_axis=0)
    # Between 2/7 (~0.286) and 2/3 (~0.667), only the wide tube (~81.5% of pore space) is invaded
    mid_mask = (pc > 0.35) & (pc < 0.60)
    assert np.any(mid_mask)
    assert np.all((snw[mid_mask] > 0.75) & (snw[mid_mask] < 0.90))
    assert float(snw[-1]) > 0.95


def test_zoom_cell_centered_eliminates_depth_shift_3x() -> None:
    """Cell-centered zoom must eliminate coordinate shift at odd 3x scale (18 um -> 6 um)."""
    n_hr = 63
    zz, _, _ = np.meshgrid(np.arange(n_hr), np.arange(n_hr), np.arange(n_hr), indexing="ij")
    ramp_hr = zz.astype(np.float32) / float(n_hr - 1)
    lr_blk = ramp_hr.reshape(21, 3, 21, 3, 21, 3).mean(axis=(1, 3, 5))

    up_legacy = nd_zoom(lr_blk, zoom=3, order=1, grid_mode=False)
    up_centered = zoom_cell_centered(lr_blk, zoom_factors=3, order=1)

    interior = slice(3, n_hr - 3)
    leg_shift = float(np.max(np.abs(up_legacy[interior] - ramp_hr[interior])) * (n_hr - 1))
    cen_shift = float(np.max(np.abs(up_centered[interior] - ramp_hr[interior])) * (n_hr - 1))

    assert leg_shift > 0.80
    assert cen_shift < 1e-4


def test_native_3x_subpixel_swinir_rrdb_sr3d_and_orthogonal_fusion() -> None:
    """3x upsampling must use native PixelShuffle(3) without residual interpolation and fuse non-cubic 3D stacks."""
    up3 = ArbitraryScaleUpsample(channels=16, scale_factor=3)
    up6 = ArbitraryScaleUpsample(channels=16, scale_factor=6)
    up25 = ArbitraryScaleUpsample(channels=16, scale_factor=2.5)
    assert up3.pow2_scale == 3 and up3.has_residual is False
    assert up6.pow2_scale == 6 and up6.has_residual is False
    assert up25.pow2_scale == 2 and up25.has_residual is True

    x2d = torch.rand(2, 3, 12, 10)
    rrdb3 = RRDBGenerator(
        in_channels=3,
        out_channels=16,
        out_img_channels=1,
        growth_channels=8,
        num_blocks=2,
        super_res_factor=3,
    ).eval()
    swin3 = SwinIRGenerator(
        in_channels=3,
        out_channels=16,
        out_img_channels=1,
        num_blocks=2,
        num_heads=4,
        window_size=4,
        super_res_factor=3,
    ).eval()
    with torch.no_grad():
        assert tuple(rrdb3(x2d).shape) == (2, 1, 36, 30)
        assert tuple(swin3(x2d).shape) == (2, 1, 36, 30)

    lr_noncubic = np.random.default_rng(5).uniform(0.1, 0.9, size=(8, 12, 10)).astype(np.float32)
    ortho = super_resolve_volume_orthogonal(
        swin3,
        lr_noncubic,
        scale_factor=3,
        fusion="confidence_fourier",
        context_slices=3,
        return_uncertainty=True,
    )
    assert ortho["fused"].shape == (24, 36, 30)
    assert ortho["uncertainty_std"].shape == (24, 36, 30)

    sr3d = SR3DResNet(out_channels=12, num_blocks=2, super_res_factor=3).eval()
    with torch.no_grad():
        out_3d = sr3d.gen(torch.rand(1, 1, 8, 8, 8))
    assert tuple(out_3d.shape) == (1, 1, 24, 24, 24)


def test_split_isolated_cross_scan_calibration_affine_and_quantile(tmp_path: Path) -> None:
    """Cross-scan calibration must invert affine/gamma radiometric shifts and never leak test slabs."""
    rng = np.random.default_rng(99)
    hr = np.clip(
        gaussian_filter(rng.uniform(0.05, 0.95, size=(54, 36, 36)).astype(np.float32), sigma=1.2),
        0.05,
        0.95,
    )
    lr_clean = degrade_xct_volume(hr, scale_factor=3, psf_sigma=1.0, noise_std=0.0, seed=99)

    # 1. Exact affine recovery
    lr_aff = np.clip(0.62 * lr_clean + 0.19, 0.0, 1.0).astype(np.float32)
    p_aff = fit_cross_scan_calibration(
        lr_aff[:12], hr[:36], scale_factor=3, mode="affine", psf_sigma=1.0
    )
    rec_aff = apply_cross_scan_calibration(lr_aff[12:], p_aff)
    assert float(np.mean(np.abs(rec_aff - lr_clean[12:]))) < 1e-3

    # 2. Monotonic polychromatic gamma recovery on unseen test slab
    lr_gam = np.clip(0.08 + 0.82 * np.power(lr_clean, 1.65), 0.0, 1.0).astype(np.float32)
    p_aff_g = fit_cross_scan_calibration(
        lr_gam[:12], hr[:36], scale_factor=3, mode="affine", psf_sigma=1.0
    )
    p_qnt_g = fit_cross_scan_calibration(
        lr_gam[:12], hr[:36], scale_factor=3, mode="quantile", psf_sigma=1.0
    )
    err_aff_g = float(
        np.mean(np.abs(apply_cross_scan_calibration(lr_gam[12:], p_aff_g) - lr_clean[12:]))
    )
    err_qnt_g = float(
        np.mean(np.abs(apply_cross_scan_calibration(lr_gam[12:], p_qnt_g) - lr_clean[12:]))
    )
    assert err_qnt_g < 0.01
    assert err_qnt_g < err_aff_g

    # 3. Zero test-slab leakage in PairedVolumeDataModule
    lr_p1 = tmp_path / "lr1.tif"
    lr_p2 = tmp_path / "lr2.tif"
    hr_p = tmp_path / "hr.tif"
    tifffile.imwrite(lr_p1, lr_gam)
    lr_corrupted_test = lr_gam.copy()
    lr_corrupted_test[-4:] = 0.99  # Mutate only the held-out test slab
    tifffile.imwrite(lr_p2, lr_corrupted_test)
    tifffile.imwrite(hr_p, hr)

    dm1 = PairedVolumeDataModule(
        low_path=str(lr_p1),
        high_path=str(hr_p),
        degrade_from_hr=False,
        scale_factor=3,
        norm_mode="percentile",
        cross_scan_calibration="quantile",
        calibration_psf_sigma=1.0,
        train_frac=0.60,
        val_frac=0.15,
        guard_slices_lr=1,
    )
    dm2 = PairedVolumeDataModule(
        low_path=str(lr_p2),
        high_path=str(hr_p),
        degrade_from_hr=False,
        scale_factor=3,
        norm_mode="percentile",
        cross_scan_calibration="quantile",
        calibration_psf_sigma=1.0,
        train_frac=0.60,
        val_frac=0.15,
        guard_slices_lr=1,
    )
    dm1.setup("fit")
    dm2.setup("fit")
    assert dm1.norm_bounds_lr == dm2.norm_bounds_lr
    assert dm1.cross_scan_calibration_params is not None
    assert dm2.cross_scan_calibration_params is not None
    assert np.allclose(
        np.asarray(dm1.cross_scan_calibration_params["xp"]),
        np.asarray(dm2.cross_scan_calibration_params["xp"]),
        atol=0.0,
    )
    assert np.allclose(
        np.asarray(dm1.cross_scan_calibration_params["fp"]),
        np.asarray(dm2.cross_scan_calibration_params["fp"]),
        atol=0.0,
    )


def test_bentheimer_dualscan_offline_and_local_raw_provenance(tmp_path: Path) -> None:
    """Dual-scan Bentheimer loader must enforce honest offline phantom provenance and coaxial local .raw loading."""
    lr_off, hr_off, man_off = prepare_bentheimer_dualscan_benchmark(
        dataset_name="bentheimer_dualscan_core1",
        dest_dir=tmp_path / "offline_ds",
        lr_shape=(12, 12, 12),
        prefer_offline=True,
        allow_offline_fallback=True,
        seed=42,
    )
    assert lr_off.is_file() and hr_off.is_file()
    assert man_off["source_kind"] == "offline_calibrated_dualscan_phantom"
    assert man_off["is_experimental_xct"] is False
    assert man_off["zenodo_doi"] is None
    assert man_off["low_source_url"] is None
    assert man_off["high_source_url"] is None
    assert man_off["lr_shape"] == [12, 12, 12]
    assert man_off["hr_shape"] == [36, 36, 36]

    # Local 16-bit little-endian .raw roundtrip with coaxial odd-margin crop (75 - 16 = 59)
    spec = AVAILABLE_DUALSCAN_DATASETS["bentheimer_dualscan_core1"]
    raw_dir = tmp_path / "raw_u16"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)
    hr_u16 = np.clip(
        gaussian_filter(rng.uniform(6000, 14000, size=spec.high_raw_shape), sigma=1.5),
        4000,
        16000,
    ).astype("<u2")
    lr_u16 = hr_u16.reshape(75, 3, 75, 3, 75, 3).mean(axis=(1, 3, 5)).astype("<u2")
    lr_u16.tofile(raw_dir / spec.low_filename)
    hr_u16.tofile(raw_dir / spec.high_filename)

    lr_real, hr_real, man_real = prepare_bentheimer_dualscan_benchmark(
        dataset_name="bentheimer_dualscan_core1",
        dest_dir=tmp_path / "local_ds",
        lr_shape=(18, 16, 16),
        raw_dir=raw_dir,
        verify_checksum=False,
        allow_offline_fallback=False,
    )
    assert lr_real.is_file() and hr_real.is_file()
    assert man_real["source_kind"] == "local_zenodo_5542624_dualscan_raw"
    assert man_real["is_experimental_xct"] is True
    assert man_real["zenodo_doi"] == "10.5281/zenodo.5542624"
    assert man_real["lr_shape"] == [18, 16, 16]
    assert man_real["hr_shape"] == [54, 48, 48]
    reg = man_real["registration"]
    assert isinstance(reg, dict)
    assert list(reg["hr_shift_zyx"]) == [0, 0, 0]

    with pytest.raises(SystemExit) as exc_cli:
        cli_main(
            [
                "download-benchmarks",
                "--datasets",
                "bentheimer_dualscan_core1",
                "--dest",
                str(tmp_path / "cli_ds"),
                "--size",
                "15",
                "--offline",
            ]
        )
    assert exc_cli.value.code == 0
    assert (tmp_path / "cli_ds" / "bentheimer_dualscan_core1_lr.tif").is_file()
    assert (tmp_path / "cli_ds" / "bentheimer_dualscan_core1_hr.tif").is_file()
