"""Comprehensive regression tests for Stage 4 Red-Team Audit (P0-1..P3-5) & external cross-review.

Explicitly locks in mathematical, physical, numerical, and provenance fixes across:
- Cell-centered half-voxel coordinate alignment (`grid_mode=True`) at 2x and 8x (`P0-1`)
- Split-isolated train-only normalization & per-split PSF degradation (`P0-2`)
- Coaxial Zenodo HDF5 cropping, joint attenuation calibration, cache-key validation,
  fail-loud `allow_offline_fallback=False`, and `None` DOIs on synthetic manifests (`P0-3`, `P0-4`)
- Window-weighted zero-DC FSC/FRC, `rfftn` multiplicity, and blur-sensitive `FSC = 0.5` (`P0-5`)
- CG convergence verification (`info == 0`, residual & flux balance) + variational bounds (`P0-6`)
- Streaming `float32`/`complex64` `rfftn` Fourier fusion with zero `NaN` (`P0-7`)
- 2.5D `z`-PSF curvature-corrected `AttenuationConservationLoss` & bimodal `SoftCLDiceLoss` (`P0-8`)
- HR-grid sub-voxel 3D registration + low-correlation quality warning (`P1-1`)
- Native-LR-strided 2.5D slice context & 3-axis orthogonal `predict_step` / `predict-volume` (`P1-2`, `P1-3`)
- Rectangular `PairedSliceDataset` `D_2` & `SwinIRGenerator` mask caching (`P1-4`, `P2-1`)
- Reference-locked petrophysical thresholding, non-periodic 8-octant `S_2(r)`, circular FOV masks,
  unimodal triphase guard, straight-through clamp, lazy discriminator, `O_h` 3D symmetry,
  tiny-patch padding, OpenImpala path-injection guard, and `ddof=1` provenance aggregation (`P1-5`..`P3-5`).
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import pytorch_lightning as pl
import tifffile
import torch
from scipy.ndimage import gaussian_filter

from openlsrx.baselines import upsample_3d
from openlsrx.cli import main as cli_main
from openlsrx.data import (
    AVAILABLE_PUBLIC_DATASETS,
    PairedSliceDataset,
    PairedVolumeDataModule,
    SyntheticPoreSpyDataModule,
    build_circular_fov_mask,
    degrade_xct_volume,
    extract_aligned_slab_from_h5,
    fetch_or_generate_rock_sample,
    prepare_zn13x_benchmark,
    register_multires_pair_3d,
    synthesize_zn13x_phantom_pair,
    zoom_cell_centered,
)
from openlsrx.data.transforms import extract_paired_slices_from_subvolume
from openlsrx.inference import fuse_orthogonal_volumes, super_resolve_volume_single_axis
from openlsrx.io import export_for_openimpala
from openlsrx.losses import AttenuationConservationLoss, SoftCLDiceLoss
from openlsrx.metrics import (
    compute_fsc_3d,
    compute_tortuosity_uncertainty_bounds,
    evaluate_image_metrics,
    evaluate_petrophysical_metrics,
    evaluate_spectral_metrics,
    evaluate_triphase_zn13x_metrics,
    solve_steady_state_diffusion_axis,
    two_point_correlation_radial,
)
from openlsrx.models import (
    SRGAN,
    Generator,
    RRDBGenerator,
    SR3DResNet,
    SwinIRGenerator,
    apply_bounded_output,
    build_3d_subcube_dataloaders,
)


@pytest.mark.parametrize("scale", [2, 8])
def test_p0_1_cell_centered_depth_alignment_zero_shift_on_linear_ramp(scale: int) -> None:
    """Verify cell-centered `grid_mode=True` eliminates +/-0.47 (2x) and +/-3.0 (8x) HR-voxel shifts."""
    n_lr = 12
    n_hr = n_lr * scale
    # Analytical linear ramp along z in [0.1, 0.9] evaluated at HR voxel centers (i + 0.5) / n_hr
    z_hr = (np.arange(n_hr, dtype=np.float32) + 0.5) / float(n_hr)
    hr_ramp = np.broadcast_to((0.1 + 0.8 * z_hr)[:, None, None], (n_hr, 16, 16)).copy()

    # Exact block-averaged LR ramp (no PSF blur, no noise) -> LR centers sit at (k + 0.5) / n_lr
    lr_ramp = degrade_xct_volume(hr_ramp, scale_factor=scale, psf_sigma=0.0, noise_std=0.0)

    # 1) 3D trilinear upsampling must match interior HR ramp exactly (zero shift)
    up_trilin = upsample_3d(lr_ramp, scale_factors=scale, mode="trilinear")
    margin = scale
    max_err_3d = float(np.max(np.abs(up_trilin[margin:-margin] - hr_ramp[margin:-margin])))
    assert max_err_3d < 1e-5, f"Expected zero depth shift at {scale}x, got max error {max_err_3d}"

    # 2) Paired slice extraction along yz (cubic order=3) has zero systematic shift and exact interior ramp
    lo_slices, hi_slices = extract_paired_slices_from_subvolume(
        lr_ramp, hr_ramp, scale_factor=scale, axes=("yz",), context_slices=1
    )
    cubic_margin = 3 * scale  # Outside 2-LR-voxel cubic B-spline boundary decay zone
    mean_signed_shift = float(
        np.mean(lo_slices[margin:-margin, 0, 0] - hi_slices[margin:-margin, 0, 0])
    )
    max_err_slices = float(
        np.max(
            np.abs(
                lo_slices[cubic_margin:-cubic_margin, 0, 0]
                - hi_slices[cubic_margin:-cubic_margin, 0, 0]
            )
        )
    )
    assert abs(mean_signed_shift) < 1e-6
    assert max_err_slices < 5e-4

    # 3) zoom_cell_centered preserves constant fields and matches single-axis stack interpolation
    identity_gen = torch.nn.Upsample(scale_factor=scale, mode="nearest")
    identity_gen.in_channels = 1  # type: ignore[assignment]
    sr_yz = super_resolve_volume_single_axis(identity_gen, lr_ramp, scale_factor=scale, axis="yz")
    assert (
        float(
            np.max(np.abs(sr_yz[cubic_margin:-cubic_margin] - hr_ramp[cubic_margin:-cubic_margin]))
        )
        < 5e-4
    )
    assert np.allclose(zoom_cell_centered(np.ones((4, 4), dtype=np.float32), 2.0), 1.0)


def test_p0_2_split_isolated_normalization_and_psf_degradation(tmp_path: Path) -> None:
    """Outlier spikes in `test` slab must NEVER leak into `train` slab normalization or PSF blur."""
    rng = np.random.default_rng(12)
    base_hr = rng.uniform(10.0, 20.0, size=(32, 16, 16)).astype(np.float32)
    spiked_hr = base_hr.copy()
    spiked_hr[-4:, :, :] = 5000.0  # Massive outlier strictly inside the test slab

    tif_clean = tmp_path / "clean.tif"
    tif_spiked = tmp_path / "spiked.tif"
    tifffile.imwrite(tif_clean, base_hr)
    tifffile.imwrite(tif_spiked, spiked_hr)

    dm_clean = PairedVolumeDataModule(
        high_path=str(tif_clean),
        degrade_from_hr=True,
        scale_factor=2,
        psf_sigma=0.8,
        degrade_noise_std=0.0,
        norm_mode="minmax",
        guard_slices_lr=1,
        seed=42,
    )
    dm_spiked = PairedVolumeDataModule(
        high_path=str(tif_spiked),
        degrade_from_hr=True,
        scale_factor=2,
        psf_sigma=0.8,
        degrade_noise_std=0.0,
        norm_mode="minmax",
        guard_slices_lr=1,
        seed=42,
    )
    dm_clean.setup("fit")
    dm_spiked.setup("fit")

    assert np.allclose(dm_clean.splits_3d["train"][0], dm_spiked.splits_3d["train"][0], atol=1e-6)
    assert np.allclose(dm_clean.splits_3d["train"][1], dm_spiked.splits_3d["train"][1], atol=1e-6)

    # SyntheticPoreSpyDataModule must also keep train bounds isolated
    ps_dm = SyntheticPoreSpyDataModule(volume_size=32, scale_factor=2, guard_slices_lr=1, seed=5)
    ps_dm.setup("fit")
    assert float(ps_dm.splits_3d["train"][1].max()) <= 1.0


def test_p0_3_and_p0_4_provenance_cache_keys_fail_loud_and_coaxial_h5(tmp_path: Path) -> None:
    """Verify Zn-13X & rock cache invalidation, fail-loud defaults, None DOIs, and coaxial H5 crop."""
    zn_dir = tmp_path / "zn_cache"

    # 1) Offline synthetic manifest must set Zenodo DOIs to None and is_experimental_xct=False
    _, hr_12, man_12 = prepare_zn13x_benchmark(
        pair="2x", dest_dir=zn_dir, lr_size=12, prefer_offline=True, seed=1
    )
    assert tifffile.imread(hr_12).shape == (24, 24, 24)
    assert man_12["source_kind"] == "offline_calibrated_zn13x_phantom"
    assert man_12["is_experimental_xct"] is False
    assert man_12["low_zenodo_doi"] is None
    assert man_12["high_zenodo_doi"] is None

    # 2) Requesting a larger lr_size=16 MUST invalidate the 12^3 cache!
    _, hr_16, man_16 = prepare_zn13x_benchmark(
        pair="2x", dest_dir=zn_dir, lr_size=16, prefer_offline=True, seed=1
    )
    assert tifffile.imread(hr_16).shape == (32, 32, 32)
    assert man_16["requested_lr_size"] == 16

    # 3) Failing loudly when allow_offline_fallback=False and real files are missing
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        prepare_zn13x_benchmark(
            pair="2x",
            dest_dir=tmp_path / "zn_strict",
            lr_size=12,
            raw_h5_dir=tmp_path / "nonexistent_h5",
            allow_offline_fallback=False,
            prefer_offline=False,
            timeout_s=0.001,
        )

    # 4) Coaxial Zenodo 14245751 8-bit cropped HDF5 loader + cache upgrade from synthetic -> real!
    preview_dir = tmp_path / "preview_h5"
    preview_dir.mkdir()
    _, hr_ref = synthesize_zn13x_phantom_pair(pair="2x", lr_shape=(16, 16, 16), seed=3)
    lr_ref = degrade_xct_volume(hr_ref, scale_factor=2, psf_sigma=0.5, noise_std=0.0)
    # Uneven coaxial FOVs around center (simulating 169067_8bit_cropped.h5 vs 169066_8bit_cropped.h5)
    lo_pad = np.pad((lr_ref * 255.0).astype(np.uint8), ((4, 4), (6, 6), (2, 2)), mode="edge")
    hi_pad = np.pad((hr_ref * 255.0).astype(np.uint8), ((2, 2), (4, 4), (6, 6)), mode="edge")
    with h5py.File(preview_dir / "169067_8bit_cropped.h5", "w") as f_lo:
        f_lo.create_dataset("data", data=lo_pad)
    with h5py.File(preview_dir / "169066_8bit_cropped.h5", "w") as f_hi:
        f_hi.create_dataset("data", data=hi_pad)

    # Passing raw_h5_dir into zn_dir (which currently holds offline synthetic!) must NOT return synthetic!
    _, hr_real, man_real = prepare_zn13x_benchmark(
        pair="2x",
        dest_dir=zn_dir,
        lr_size=14,
        raw_h5_dir=preview_dir,
        allow_offline_fallback=False,
        prefer_offline=False,
    )
    assert man_real["source_kind"] == "local_i13_preview_h5_registered"
    assert man_real["is_experimental_xct"] is True
    assert man_real["low_zenodo_doi"] is not None
    assert tifffile.imread(hr_real).shape == (28, 28, 28)

    # 5) Joint physical [vmin, vmax] attenuation window preserves identical mu across LR and HR H5s
    savu_lo = tmp_path / "169067_recon.h5"
    savu_hi = tmp_path / "169066_recon.h5"
    with h5py.File(savu_lo, "w") as fl:
        ds = fl.create_dataset("4-TomopyRecon-tomo/data", shape=(1600, 920, 1700), dtype="f4")
        ds[858:874, 886:902, 1010:1026] = 2.0e-4
    with h5py.File(savu_hi, "w") as fh:
        ds = fh.create_dataset("4-TomopyRecon-tomo/data", shape=(1800, 840, 2100), dtype="f4")
        ds[466:498, 788:820, 763:795] = 2.0e-4
    slab_lo, slab_hi, _ = extract_aligned_slab_from_h5(
        savu_lo,
        savu_hi,
        pair="2x",
        lr_slab_depth=16,
        lr_crop_yx=16,
        consistent_attenuation_window=True,
    )
    assert np.allclose(slab_lo.mean(), slab_hi.mean(), atol=1e-5)

    # 6) Public rock benchmarks provenance & cache shape check
    assert "zenodo.org/records/5542624" in str(
        AVAILABLE_PUBLIC_DATASETS["bentheimer_sandstone"].url
    )
    assert AVAILABLE_PUBLIC_DATASETS["indiana_limestone"].url is None

    rock_dir = tmp_path / "rock_cache"
    t16, m16 = fetch_or_generate_rock_sample(
        "bentheimer_sandstone", dest_dir=rock_dir, shape=(16, 16, 16), prefer_offline=True
    )
    assert m16["source_url"] is None and m16["is_experimental_xct"] is False
    t20, _ = fetch_or_generate_rock_sample(
        "bentheimer_sandstone", dest_dir=rock_dir, shape=(20, 20, 20), prefer_offline=True
    )
    assert tifffile.imread(t20).shape == (20, 20, 20)


def test_p0_5_fsc_window_zero_dc_and_050_blur_sensitivity() -> None:
    """Verify MTF_eff=0.5 catches noiseless Gaussian blur and FSC=0.5 catches downsampling/noisy blur."""
    rng = np.random.default_rng(99)
    # Strictly positive mean [0.4, 0.9] tests Hann-apodized DC leakage suppression
    hr = rng.uniform(0.4, 0.9, size=(32, 32, 32)).astype(np.float32)
    blurred_1vox = gaussian_filter(hr, sigma=1.0).astype(np.float32)
    lr_deg = degrade_xct_volume(hr, scale_factor=2, psf_sigma=1.0, noise_std=0.015, seed=99)
    sr_interp = upsample_3d(lr_deg, scale_factors=2, mode="tricubic")

    fsc_ident = compute_fsc_3d(hr, hr, voxel_size_um=0.8125)
    spec_blur = evaluate_spectral_metrics(blurred_1vox, hr, voxel_size_um=0.8125)
    spec_sr = evaluate_spectral_metrics(sr_interp, hr, voxel_size_um=0.8125)

    assert fsc_ident["frequencies_cyc_per_voxel"][0] > 0.0  # k == 0 excluded
    assert np.isclose(fsc_ident["nyquist_fraction_050"], 1.0, atol=1e-5)
    assert np.isclose(fsc_ident["nyquist_fraction_mtf_050"], 1.0, atol=1e-5)

    # 1) Noiseless zero-phase sigma=1.0 blur: reference-normalized MTF_eff=0.5 drops to ~0.37 of Nyquist!
    assert spec_blur["mtf_nyquist_fraction_050"] < 0.55
    assert spec_blur["mtf_half_pitch_050_um"] > 0.8125 * 1.6
    assert spec_blur["mtf_auc"] < 0.50

    # 2) Downsampled + upsampled noisy XCT: FSC=0.5 drops well below Nyquist while 0.143 stays higher!
    assert spec_sr["fsc_nyquist_fraction_050"] < 0.70
    assert spec_sr["fsc_half_pitch_050_um"] > 0.8125 * 1.35
    assert spec_sr["fsc_cutoff_050_cyc_vox"] < spec_sr["fsc_cutoff_0143_cyc_vox"]


def test_p0_6_and_p1_8_diffusion_cg_convergence_fallback_and_monotonic_bounds() -> None:
    """Capping CG at maxiter=2 must NOT silently return unphysical D_eff > porosity."""
    duct = np.zeros((16, 10, 10), dtype=bool)
    duct[:, 2:8, 2:8] = True
    duct[5, 2:7, 2:8] = False
    duct[11, 3:8, 2:8] = False

    ref = solve_steady_state_diffusion_axis(duct, axis=0, tol=1e-10, maxiter=4000)
    capped = solve_steady_state_diffusion_axis(duct, axis=0, tol=1e-10, maxiter=2)

    assert ref["solver_converged"] == 1.0
    assert capped["solver_converged"] == 1.0
    assert 0.0 < capped["d_eff"] <= capped["porosity"]
    assert capped["tortuosity"] >= 1.0
    assert capped["d_eff"] == pytest.approx(ref["d_eff"], rel=1e-4)

    # Uncertainty bounds must satisfy lower <= upper even when 1 of 3 axes pinches off
    vol = np.where(duct, 0.2, 0.8).astype(np.float32)
    unc = np.full_like(vol, 0.08)
    bounds = compute_tortuosity_uncertainty_bounds(vol, unc, true_vol=vol, pore_is_low=True)
    assert float(bounds["d_eff_lower"]) <= float(bounds["d_eff_upper"]) + 1e-8  # type: ignore[arg-type]
    if bounds["tortuosity_lower"] is not None and bounds["tortuosity_upper"] is not None:
        assert float(bounds["tortuosity_lower"]) <= float(bounds["tortuosity_upper"]) + 1e-8


def test_p0_7_streaming_rfftn_fourier_fusion_zero_nan_with_psf_cutoff() -> None:
    """Streaming float32/complex64 rfftn fusion must never emit NaN even with tight psf_cutoff."""
    rng = np.random.default_rng(4)
    v_yz = rng.random((16, 16, 16), dtype=np.float32)
    v_xz = rng.random((16, 16, 16), dtype=np.float32)
    v_xy = rng.random((16, 16, 16), dtype=np.float32)

    for mode in ("fourier", "confidence", "confidence_fourier"):
        fused = fuse_orthogonal_volumes(
            {"yz": v_yz, "xz": v_xz, "xy": v_xy},
            mode=mode,  # type: ignore[arg-type]
            psf_cutoff=0.05,
        )
        assert fused.dtype == np.float32
        assert np.isfinite(fused).all()
        assert 0.0 <= float(fused.min()) <= float(fused.max()) <= 1.0


def test_p0_8_25d_z_laplacian_conservation_and_homogeneous_soft_cldice() -> None:
    """Verify 2.5D z-PSF curvature term in AttenuationConservationLoss and homogeneous SoftCLDice."""
    # Quadratic z profile: mu(z_LR) = 0.4 + 0.04 * z_LR^2 -> d^2 mu / dz_LR^2 = 0.08
    # Under sigma_HR = 1.0 at 2x (sigma_LR = 0.5), 3D Gaussian blur shifts LR by +0.5 * 0.25 * 0.08 = +0.01
    z_lr = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).view(1, 3, 1, 1)
    lr_25d = (0.4 + 0.04 * (z_lr**2) + 0.01).expand(2, 3, 8, 8).contiguous()
    hr_mid = torch.full((2, 1, 16, 16), 0.4, dtype=torch.float32)

    loss_25d = AttenuationConservationLoss(psf_sigma=1.0)
    loss_no_z = AttenuationConservationLoss(psf_sigma=0.0)
    assert float(loss_25d(hr_mid, lr_25d)) < 1e-6
    assert float(loss_no_z(hr_mid, lr_25d)) == pytest.approx(0.01, abs=1e-5)

    # Single-phase all-solid or all-pore patch must yield exact 0.0 SoftCLDiceLoss with finite grads
    cl = SoftCLDiceLoss(threshold=None)
    solid_patch = torch.full((2, 1, 16, 16), 0.85, dtype=torch.float32)
    pred_solid = (solid_patch + 0.005 * torch.randn_like(solid_patch)).requires_grad_(True)
    val = cl(pred_solid, solid_patch)
    assert float(val.detach()) == 0.0
    val.backward()
    assert pred_solid.grad is not None and torch.isfinite(pred_solid.grad).all()


def test_p1_1_subvoxel_hr_grid_registration_and_quality_warning() -> None:
    """Verify sub-LR-voxel (+1 HR voxel at 2x) registration and low-correlation warning."""
    _, hr = synthesize_zn13x_phantom_pair(pair="2x", lr_shape=(20, 20, 20), seed=31)
    # Shift HR by +1 HR voxel (= 0.5 LR voxel!) before degrading to LR
    hr_shifted = np.pad(hr, ((1, 0), (1, 0), (0, 0)), mode="edge")[:40, :40, :40]
    lr_half_shift = degrade_xct_volume(hr_shifted, scale_factor=2, psf_sigma=0.6, noise_std=0.0)

    al_lr, al_hr, info = register_multires_pair_3d(
        lr_half_shift, hr, scale_factor=2, local_refine_radius=2
    )
    assert al_hr.shape == tuple(s * 2 for s in al_lr.shape)
    assert info["registration_quality_ok"] is True
    assert float(info["coarse_mid_slice_ssim"]) > 0.85  # type: ignore[arg-type]

    # Uncorrelated pure noise volumes must trigger a low-SSIM UserWarning
    rng = np.random.default_rng(77)
    noise_lr = rng.random((16, 16, 16), dtype=np.float32)
    noise_hr = rng.random((32, 32, 32), dtype=np.float32)
    with pytest.warns(UserWarning, match="Low 3D registration tri-planar SSIM"):
        _, _, bad_info = register_multires_pair_3d(noise_lr, noise_hr, scale_factor=2)
    assert bad_info["registration_quality_ok"] is False


def test_p1_2_and_p1_3_native_lr_25d_stride_orthogonal_predict_and_2d_cli(tmp_path: Path) -> None:
    """Verify native-LR 2.5D stride, 3-axis orthogonal `predict_step`, `predict-volume`, and 2D CLI."""
    # 1) At 2x scale, 2.5D context_slices=3 on zoomed stack must step by stride=2 HR slices (= 1 LR slice)
    lr_ramp = np.broadcast_to(
        np.arange(6, dtype=np.float32)[:, None, None] / 10.0, (6, 8, 8)
    ).copy()
    hr_dummy = np.zeros((12, 16, 16), dtype=np.float32)
    lo_ctx, _ = extract_paired_slices_from_subvolume(
        lr_ramp, hr_dummy, scale_factor=2, axes=("yz",), context_slices=3
    )
    # Interior slice z_hr = 4: neighbors are z_hr = 2 and z_hr = 6 -> difference = 2 * (0.1 / 2) = 0.1 (1 full LR step!)
    step_diff = float(lo_ctx[4, 2, 0, 0] - lo_ctx[4, 1, 0, 0])
    assert step_diff == pytest.approx(0.1, abs=5e-3)

    # 2) PairedVolumeDataModule + SRGAN trainer.predict -> 3-axis orthogonal BigTIFF + uncertainty
    lr_tif = tmp_path / "lr.tif"
    hr_tif = tmp_path / "hr.tif"
    rng = np.random.default_rng(5)
    tifffile.imwrite(lr_tif, rng.random((10, 12, 14), dtype=np.float32))
    tifffile.imwrite(hr_tif, rng.random((20, 24, 28), dtype=np.float32))

    dm = PairedVolumeDataModule(
        low_path=str(lr_tif),
        high_path=str(hr_tif),
        scale_factor=2,
        context_slices=3,
        predict_mode="orthogonal_3axis",
        predict_fusion="confidence_fourier",
        batch_size=2,
        eval_batch_size=2,
    )
    out_dir = tmp_path / "ortho_pred_out"
    model = SRGAN(
        super_res_factor=2,
        num_blocks=1,
        out_channels=8,
        context_slices=3,
        backbone="swinir",
        swin_window_size=4,
        swin_num_heads=2,
        use_adversarial_loss=False,
        use_vgg_loss=False,
        sample_id="ortho",
        output_dir=str(out_dir),
    )
    assert model.disc is None  # P1-10 lazy discriminator instantiation!

    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.predict(model, datamodule=dm)
    pred_stack = tifffile.imread(out_dir / "predicted_orthogonal" / "ortho_predicted_stack.tiff")
    unc_stack = tifffile.imread(
        out_dir / "predicted_orthogonal" / "ortho_uncertainty_std_stack.tiff"
    )
    assert pred_stack.shape == unc_stack.shape

    # 3) Standalone `openlsrx predict-volume` CLI + 2D slice `openlsrx evaluate` (`preserve_2d=True`)
    ckpt_path = tmp_path / "swin.ckpt"
    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(ckpt_path)

    cli_sr_out = tmp_path / "cli_sr" / "lr_sr_fourier.tif"
    cli_unc_out = tmp_path / "cli_sr" / "lr_uncertainty_std.tif"
    with pytest.raises(SystemExit) as exc_pv:
        cli_main(
            [
                "predict-volume",
                "--ckpt",
                str(ckpt_path),
                "--input",
                str(lr_tif),
                "--output",
                str(cli_sr_out),
                "--fusion",
                "fourier",
                "--uncertainty-output",
                str(cli_unc_out),
            ]
        )
    assert exc_pv.value.code == 0
    assert cli_sr_out.is_file()
    assert cli_unc_out.is_file()

    slice_pred = tmp_path / "slice_pred.tif"
    slice_gt = tmp_path / "slice_gt.tif"
    eval_2d_json = tmp_path / "eval_2d.json"
    tifffile.imwrite(slice_pred, rng.random((24, 28), dtype=np.float32))
    tifffile.imwrite(slice_gt, rng.random((24, 28), dtype=np.float32))
    with pytest.raises(SystemExit) as exc_2d:
        cli_main(
            [
                "evaluate",
                "--pred",
                str(slice_pred),
                "--target",
                str(slice_gt),
                "--include-spectral",
                "--output",
                str(eval_2d_json),
            ]
        )
    assert exc_2d.value.code == 0
    rep_2d = json.loads(eval_2d_json.read_text(encoding="utf-8"))
    assert "fsc_auc" in rep_2d["spectral_metrics"]
    assert "fsc_cutoff_050_cyc_vox" in rep_2d["spectral_metrics"]


def test_p1_4_to_p3_5_models_metrics_fov_straight_through_and_security(tmp_path: Path) -> None:
    """Verify rectangular D_2/SwinIR, HR-locked thresholding, non-periodic S_2(r), STE, O_h, and OpenImpala."""
    from openlsrx.models import ResidualSwinBlock, SwinTransformerLayer

    # 1) Non-square PairedSliceDataset D_2 augmentation & rectangular SwinIR mask cache
    lo_rect = np.random.default_rng(1).random((8, 3, 10, 16), dtype=np.float32)
    hi_rect = np.random.default_rng(2).random((8, 20, 32), dtype=np.float32)
    ds_rect = PairedSliceDataset(lo_rect, hi_rect, scale_factor=2, augment=True, context_slices=3)
    for idx in range(len(ds_rect)):
        l_i, h_i = ds_rect[idx]
        assert l_i.shape == (3, 10, 16)
        assert h_i.shape == (20, 32)

    swin = SwinIRGenerator(
        in_channels=3, out_channels=16, num_blocks=1, depth_per_block=2, num_heads=4, window_size=8
    )
    out_rect = swin(torch.rand(1, 3, 10, 18))
    assert out_rect.shape == (1, 1, 20, 36)
    blk0 = swin.trunk[0]
    assert isinstance(blk0, ResidualSwinBlock)
    lyr1 = blk0.layers[1]
    assert isinstance(lyr1, SwinTransformerLayer)
    assert len(lyr1._mask_cache) == 1

    # 2) Tiny-patch (1x1 / 2x2) boundary-safe convs + straight-through clamp gradient flow
    tiny_gen = Generator(in_channels=1, out_channels=8, num_blocks=1, super_res_factor=2)
    tiny_rrdb = RRDBGenerator(in_channels=1, out_channels=8, growth_channels=4, num_blocks=1)
    assert tiny_gen(torch.rand(1, 1, 2, 2)).shape == (1, 1, 4, 4)
    assert tiny_rrdb(torch.rand(1, 1, 2, 2)).shape == (1, 1, 4, 4)

    raw_overshoot = torch.tensor([-0.4, 0.5, 1.6], requires_grad=True)
    clamped = apply_bounded_output(raw_overshoot, activation="clamp", training=True)
    assert torch.allclose(clamped, torch.tensor([0.0, 0.5, 1.0]))
    clamped.sum().backward()
    assert raw_overshoot.grad is not None and torch.allclose(raw_overshoot.grad, torch.ones(3))

    # 3) Reference-locked petrophysical thresholding catches global greyscale offset bias!
    gt = np.zeros((24, 24, 24), dtype=np.float32)
    gt[:, :, :12] = 0.2
    gt[:, :, 12:] = 0.8
    # Prediction shifted upward by +0.35 everywhere -> auto-Otsu would hide the error, HR-locked catches it!
    biased_pred = np.clip(gt + 0.35, 0.0, 1.0)
    petro = evaluate_petrophysical_metrics(biased_pred, gt, pore_is_low=True)
    assert petro["porosity_abs_error"] > 0.45
    assert petro["porosity_auto_otsu_abs_error"] == pytest.approx(0.0, abs=1e-6)

    # 4) Non-periodic 8-octant two_point_correlation_radial: opposite-face slabs do NOT wrap around!
    non_periodic = np.zeros((16, 16, 16), dtype=bool)
    non_periodic[0, :, :] = True
    non_periodic[-1, :, :] = True
    s2 = two_point_correlation_radial(non_periodic, max_lag=4)
    # At lag r=1 along z, non-periodic pairs have zero overlap between z=0 and z=15!
    assert s2[0] == pytest.approx(float(non_periodic.mean()), rel=1e-4)
    assert s2[1] < s2[0] * 0.75

    # 5) Circular FOV mask, (1, H, W) single-slice safety, and unimodal triphase guard
    fov = build_circular_fov_mask((8, 16, 16))
    assert fov.shape == (8, 16, 16) and not bool(fov[0, 0, 0]) and bool(fov[4, 8, 8])
    thin_1hw = np.random.default_rng(9).random((1, 24, 24), dtype=np.float32)
    assert np.isfinite(evaluate_image_metrics(thin_1hw, thin_1hw)["ssim"])
    const_vol = np.full((12, 12, 12), 0.5, dtype=np.float32)
    tri_const = evaluate_triphase_zn13x_metrics(const_vol, const_vol)
    assert np.isfinite(tri_const["triphase_macro_dice"])

    # 6) 48-element O_h 3D subcube augmentation & tiny 3D conv safety
    lr_3d = np.random.default_rng(6).random((8, 8, 8), dtype=np.float32)
    hr_3d = np.random.default_rng(7).random((16, 16, 16), dtype=np.float32)
    tr_3d, _ = build_3d_subcube_dataloaders(
        lr_3d, hr_3d, lr_3d, hr_3d, scale_factor=2, patch_size_lr=4, train_samples=6, val_samples=2
    )
    b_lo, b_hi = next(iter(tr_3d))
    assert b_lo.shape[-3:] == (4, 4, 4) and b_hi.shape[-3:] == (8, 8, 8)
    sr3d = SR3DResNet(super_res_factor=2, out_channels=8, num_blocks=1)
    assert sr3d(torch.rand(1, 1, 2, 2, 2)).shape == (1, 1, 4, 4, 4)

    # 7) OpenImpala strict name validation blocks path traversal / quote injection
    for bad_name in ("../escape", "bad/name", 'quote"inject', "space name"):
        with pytest.raises(ValueError, match="Invalid OpenImpala sample name"):
            export_for_openimpala(gt, output_dir=tmp_path / "imp", name=bad_name)

    # 8) Campaign aggregator ddof=1 sample std, n_percolating_seeds, & source_kinds provenance
    spec = importlib.util.spec_from_file_location(
        "aggregate_campaign",
        Path(__file__).resolve().parents[1] / "scripts" / "benchmarks" / "aggregate_campaign.py",
    )
    assert spec is not None and spec.loader is not None
    agg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg_mod)

    csv_p = tmp_path / "sample_results.csv"
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "source_kind", "seed", "method", "psnr", "pred_tortuosity_mean"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "zn13x_2x",
                "source_kind": "zenodo_14245751_8bit_cropped_h5_registered",
                "seed": 42,
                "method": "SwinIR",
                "psnr": 20.0,
                "pred_tortuosity_mean": 1.8,
            }
        )
        writer.writerow(
            {
                "dataset": "zn13x_2x",
                "source_kind": "zenodo_14245751_8bit_cropped_h5_registered",
                "seed": 43,
                "method": "SwinIR",
                "psnr": 22.0,
                "pred_tortuosity_mean": "inf",  # non-percolating seed!
            }
        )
    out_agg = tmp_path / "agg_out"
    agg_mod.aggregate_csv_files([csv_p], out_agg)
    summary = json.loads((out_agg / "campaign_summary.json").read_text(encoding="utf-8"))
    rec = summary[0]
    # Sample std (ddof=1) of [20.0, 22.0] is sqrt(2) = 1.41421356, NOT population std 1.0!
    assert rec["psnr_std"] == pytest.approx(np.sqrt(2.0), rel=1e-5)
    assert rec["n_seeds"] == 2
    assert rec["n_percolating_seeds"] == 1
    assert rec["source_kinds"] == ["zenodo_14245751_8bit_cropped_h5_registered"]


def test_stage5_dualscan_z_shift_symmetric_pad_and_heterogeneous_eval() -> None:
    """Verify Stage 5 P0-1 & P0-2 fixes: exact HR-grid Z-shift slicing, symmetric pad, and unpadded eval."""
    from openlsrx.data import (
        apply_hr_grid_shift_pair_3d,
        build_multi_axis_slice_dataset,
        make_slice_dataloader,
    )
    from openlsrx.data.transforms import _safe_pad_spatial

    # 1) Exact integer HR-grid shift slicing resolves a -1 HR-voxel Z-offset with zero interpolation
    _, hr_base = synthesize_zn13x_phantom_pair(pair="2x", lr_shape=(24, 24, 24), seed=19)
    # Create a 3x pair (18^3 LR, 54^3 HR) where LR is degraded from z=[1:55] and HR is z=[0:54] (hr_shift_zyx == [-1, 0, 0])
    hr_full = zoom_cell_centered(hr_base, zoom_factors=57.0 / 48.0, order=1).astype(np.float32)
    lr_from_z1 = degrade_xct_volume(
        hr_full[1:55, :54, :54], scale_factor=3, psf_sigma=0.8, noise_std=0.0
    )
    hr_z0 = hr_full[:54, :54, :54]
    _, _, info_before = register_multires_pair_3d(lr_from_z1, hr_z0, scale_factor=3)
    detected = info_before["hr_shift_zyx"]
    assert isinstance(detected, list) and detected[0] == -1

    sliced = apply_hr_grid_shift_pair_3d(lr_from_z1, hr_z0, shift_hr_zyx=(-1, 0, 0), scale_factor=3)
    assert sliced is not None
    al_lr, al_hr = sliced
    _, _, info_after = register_multires_pair_3d(al_lr, al_hr, scale_factor=3)
    assert info_after["hr_shift_zyx"] == [0, 0, 0]

    # 2) Half-sample symmetric spatial padding preserves exact cell-centered alignment at 3x & 8x
    for s in (3, 8):
        n_lr = 10
        n_hr = n_lr * s
        ramp_hr = np.linspace(0.0, 1.0, n_hr, dtype=np.float32)[np.newaxis, :, np.newaxis]
        ramp_hr = np.broadcast_to(ramp_hr, (2, n_hr, n_hr)).copy()
        ramp_lr = ramp_hr.reshape(2, n_lr, s, n_lr, s).mean(axis=(2, 4)).astype(np.float32)

        pad_lr = _safe_pad_spatial(ramp_lr, n_lr + 6, n_lr + 6)
        pad_hr = _safe_pad_spatial(ramp_hr, (n_lr + 6) * s, (n_lr + 6) * s)
        up_pad_lr = zoom_cell_centered(pad_lr, zoom_factors=(1.0, float(s), float(s)), order=1)
        # Inside the reflected linear pad [n_hr + s : n_hr + 4 * s], half-sample symmetric pad
        # has exact zero (< 1e-5) phase error, whereas whole-sample reflect had a 1-HR-voxel bias!
        refl_err = np.max(
            np.abs(
                up_pad_lr[:, n_hr + s : n_hr + 4 * s, s : 3 * s]
                - pad_hr[:, n_hr + s : n_hr + 4 * s, s : 3 * s]
            )
        )
        assert float(refl_err) < 1e-5

    # 3) Non-cubic multi-axis evaluation keeps thin slabs unpadded and batches purely per subdataset
    rng = np.random.default_rng(12)
    lr_slab = rng.random((5, 16, 16), dtype=np.float32)
    hr_slab = rng.random((15, 48, 48), dtype=np.float32)
    eval_ds = build_multi_axis_slice_dataset(
        lr_slab,
        hr_slab,
        scale_factor=3,
        axes=("yz", "xz", "xy"),
        patch_size_lr=None,
        augment=False,
        context_slices=3,
    )
    dl = make_slice_dataloader(eval_ds, batch_size=6, shuffle=False)
    seen_shapes = {tuple(hi.shape[-2:]) for _, hi in dl}
    assert (48, 48) in seen_shapes
    assert (15, 48) in seen_shapes


def test_stage5_dark_cap_slice_loader_and_checkpoint_radiometric_restoration(
    tmp_path: Path,
) -> None:
    """Verify Stage 5 P0-3, P0-4, & P0-5 fixes: dark uint16 slice dirs, checkpoint calibration, & SR3D CLI."""
    from openlsrx.data import load_volume_any
    from openlsrx.models import SRGAN, SR3DResNet

    # 1) Dark uint16 cap slice (max == 1) in a 2D slice directory must scale by 1/65535, NOT stay 1.0!
    slice_dir = tmp_path / "uint16_slices"
    slice_dir.mkdir()
    dark_cap = np.ones((16, 16), dtype=np.uint16)  # max == 1!
    mid_slice = np.full((16, 16), 32768, dtype=np.uint16)
    tifffile.imwrite(slice_dir / "s000.tif", dark_cap)
    tifffile.imwrite(slice_dir / "s001.tif", mid_slice)
    loaded = load_volume_any(slice_dir)
    assert loaded.shape == (2, 16, 16)
    assert float(loaded[0].max()) == pytest.approx(1.0 / 65535.0, rel=1e-4)
    assert float(loaded[1].mean()) == pytest.approx(32768.0 / 65535.0, rel=1e-4)

    # 2) Train-fitted quantile calibration persists in checkpoint and auto-applies in predict-volume
    rng = np.random.default_rng(55)
    hr_vol = np.clip(0.2 + 0.6 * rng.random((24, 24, 24), dtype=np.float32), 0.0, 1.0)
    # Non-linear gamma-distorted LR scan (12x12x12) simulating uncalibrated cross-scan contrast
    lr_raw = np.power(degrade_xct_volume(hr_vol, scale_factor=2, psf_sigma=0.5, noise_std=0.0), 2.2)
    lr_tif = tmp_path / "lr_gamma.tif"
    hr_tif = tmp_path / "hr_lin.tif"
    tifffile.imwrite(lr_tif, lr_raw.astype(np.float32))
    tifffile.imwrite(hr_tif, hr_vol.astype(np.float32))

    dm = PairedVolumeDataModule(
        low_path=str(lr_tif),
        high_path=str(hr_tif),
        degrade_from_hr=False,
        scale_factor=2,
        norm_mode="percentile",
        cross_scan_calibration="quantile",
        train_frac=0.60,
        val_frac=0.20,
        guard_slices_lr=0,
        patch_size_lr=8,
        batch_size=2,
    )
    model = SRGAN(
        backbone="swinir",
        context_slices=1,
        super_res_factor=2,
        out_channels=16,
        num_blocks=1,
        swin_num_heads=4,
        swin_window_size=4,
        use_adversarial_loss=False,
        use_vgg_loss=False,
        save_val_images=False,
    )
    ckpt_path = tmp_path / "calibrated_swinir.ckpt"
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(ckpt_path)

    loaded_model = SRGAN.load_from_checkpoint(ckpt_path, map_location="cpu")
    assert isinstance(loaded_model._saved_radiometric_state, dict)
    cal_state = loaded_model._saved_radiometric_state["cross_scan_calibration_params"]
    assert isinstance(cal_state, dict) and cal_state["mode"] == "quantile"

    out_dir = tmp_path / "cli_pred_out"
    with pytest.raises(SystemExit) as exc_pv:
        cli_main(
            [
                "predict-volume",
                "--ckpt",
                str(ckpt_path),
                "--input",
                str(lr_tif),
                "--output-dir",
                str(out_dir),
                "--save-uncertainty",
                "--device",
                "cpu",
            ]
        )
    assert exc_pv.value.code == 0
    assert (out_dir / "predicted_stack.tiff").is_file()
    assert (out_dir / "uncertainty_std_stack.tiff").is_file()

    # 3) SR3DResNet checkpoint auto-detection in openlsrx predict-volume (12^3 -> 24^3 at 2x)
    m3d = SR3DResNet(super_res_factor=2, out_channels=8, num_blocks=1)
    ckpt_3d = tmp_path / "sr3d.ckpt"
    t3d = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu", devices=1)
    t3d.strategy.connect(m3d)
    t3d.save_checkpoint(ckpt_3d)
    out_3d = tmp_path / "pred_3d.tiff"
    with pytest.raises(SystemExit) as exc_3d:
        cli_main(
            [
                "predict-volume",
                "--ckpt",
                str(ckpt_3d),
                "--input",
                str(lr_tif),
                "--output",
                str(out_3d),
                "--device",
                "cpu",
            ]
        )
    assert exc_3d.value.code == 0
    assert tifffile.imread(out_3d).shape == (24, 24, 24)


def test_stage5_fourier_diagonal_edge_thin_fsc_ms_ssim_and_openimpala_blocking(
    tmp_path: Path,
) -> None:
    """Verify Stage 5 P1-1..P2 fixes: sharp diagonal Fourier fusion, thin-slab FSC, MS-SSIM, & AMReX blocking."""
    from openlsrx.losses import SobelGradientLoss
    from openlsrx.metrics.image import compute_ms_ssim, compute_ssim_3d_isotropic

    # 1) Diagonal (kz=0.36, ky=0.36) edge resolved sharply by xy sweep is NOT clamped onto blurry yz/xz sweeps
    n = 36
    zz, yy, _ = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    sharp_diag = (0.5 + 0.35 * np.cos(2.0 * np.pi * 0.36 * zz + 2.0 * np.pi * 0.36 * yy)).astype(
        np.float32
    )
    blur_z = gaussian_filter(sharp_diag, sigma=(2.2, 0.0, 0.0)).astype(np.float32)
    blur_y = gaussian_filter(sharp_diag, sigma=(0.0, 2.2, 0.0)).astype(np.float32)
    fused_cf = fuse_orthogonal_volumes(
        {"yz": blur_z, "xz": blur_y, "xy": sharp_diag},
        mode="confidence_fourier",
        scale_factor=3,
    )
    interior = (slice(4, -4), slice(4, -4), slice(4, -4))
    rmse_cf = float(np.sqrt(np.mean((fused_cf[interior] - sharp_diag[interior]) ** 2)))
    rmse_mean = float(
        np.sqrt(
            np.mean((((blur_z + blur_y + sharp_diag) / 3.0)[interior] - sharp_diag[interior]) ** 2)
        )
    )
    assert rmse_cf < 0.03 and rmse_cf < 0.25 * rmse_mean

    # 2) Thin-slab (8, 64, 64) 3D FSC preserves 32 radial shells and uses FSC=0.5 for fsc_nyquist_fraction
    rng = np.random.default_rng(88)
    slab_true = rng.random((8, 64, 64), dtype=np.float32)
    slab_pred = gaussian_filter(slab_true, sigma=0.8).astype(np.float32)
    fsc = compute_fsc_3d(slab_pred, slab_true)
    assert len(fsc["frequencies_cyc_per_voxel"]) == 32
    spec_m = evaluate_spectral_metrics(slab_pred, slab_true)
    assert spec_m["fsc_nyquist_fraction"] == pytest.approx(fsc["nyquist_fraction_050"], abs=1e-6)

    # 3) compute_ms_ssim returns None cleanly on 168x168 (< 176) and float on 176x176; D=2 3D SSIM works
    a168 = torch.rand(1, 1, 168, 168)
    assert compute_ms_ssim(a168, a168) is None
    a176 = torch.rand(1, 1, 176, 176)
    ms176 = compute_ms_ssim(a176, a176)
    assert ms176 is not None and ms176 == pytest.approx(1.0, abs=1e-4)
    d2_vol = rng.random((2, 24, 24), dtype=np.float32)
    assert compute_ssim_3d_isotropic(d2_vol, d2_vol) == pytest.approx(1.0, abs=1e-5)

    # 4) Constant all-pore (0.0) volume segments to porosity == 1.0, and SobelGradientLoss works in 2D & 3D
    all_pore = np.zeros((12, 12, 12), dtype=np.float32)
    petro_pore = evaluate_petrophysical_metrics(all_pore, all_pore, pore_is_low=True)
    assert petro_pore["porosity_true"] == pytest.approx(1.0)
    assert petro_pore["porosity_pred"] == pytest.approx(1.0)
    sobel = SobelGradientLoss()
    assert float(sobel(torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8))) == 0.0
    assert float(sobel(torch.ones(1, 1, 6, 6, 6), torch.ones(1, 1, 6, 6, 6))) == 0.0

    # 5) OpenImpala writes amr.blocking_factor = gcd(nx, ny, nz, 8) so 27^3 / 225^3 never crash AMReX
    odd_vol = (rng.random((27, 27, 27)) > 0.5).astype(np.float32)
    _, inp_odd = export_for_openimpala(odd_vol, output_dir=tmp_path / "imp_odd", name="core225")
    inp_txt = inp_odd.read_text(encoding="utf-8")
    assert "amr.blocking_factor = 1" in inp_txt


def test_zn13x_7dof_similarity_resampling_chained_8x_and_certified_regions_cli(
    tmp_path: Path,
) -> None:
    """Verify MultimodalBenchmark 7-DOF similarity registration, chained 8x composition, edge-drift fix, & CLI."""
    import json

    import h5py
    from scipy.ndimage import affine_transform

    from openlsrx.cli import main as cli_main
    from openlsrx.data import (
        DEFAULT_CERTIFIED_REGIONS_PATH,
        SimilarityTransform3DSpec,
        compose_similarity_transforms_3d,
        load_zn13x_certified_regions,
        resample_lr_to_hr_grid_7dof,
        resolve_zn13x_pair_7dof,
    )

    # 1) Canonical certified_regions.yaml exists and chained 3-rung 8x matches sequential composition to <1e-11 um
    assert DEFAULT_CERTIFIED_REGIONS_PATH.is_file()
    cert_doc, _, cert_sha = load_zn13x_certified_regions()
    assert len(cert_sha) == 64

    tx_8x_prev, hr_center_8x, meta_8x_prev = resolve_zn13x_pair_7dof(
        "8x", tier_section="zenodo_14245751_preview_8bit"
    )
    assert hr_center_8x is not None
    assert meta_8x_prev["transform_chain_hr_to_lr"] == [
        "169066_from_169065",
        "169067_from_169066",
        "169068_from_169067",
    ]
    assert tx_8x_prev.scale == pytest.approx(0.987968, abs=1e-5)

    rungs_raw = cert_doc["zenodo_14245751_preview_8bit"]["adjacent_rungs"]
    seq_specs = [
        SimilarityTransform3DSpec.from_dict(rungs_raw[k]["similarity_transform_hr_to_lr"])
        for k in ("169066_from_169065", "169067_from_169066", "169068_from_169067")
    ]
    composed_manual = compose_similarity_transforms_3d(seq_specs)
    rng = np.random.default_rng(2026)
    pts_hr = rng.uniform(0.0, 325.0, size=(64, 3))
    pts_seq = pts_hr.copy()
    for sp in seq_specs:
        pts_seq = sp.transform_points_xyz_um(pts_seq)
    pts_comp = composed_manual.transform_points_xyz_um(pts_hr)
    assert float(np.max(np.abs(pts_seq - pts_comp))) < 1e-11

    # 2) ~1.36% objective magnification drift (scale=0.9864) + sub-voxel shift:
    #    nominal integer slicing loses outer-shell correlation, whereas 7-DOF keeps HR bit-exact & fixes edges!
    _, hr_full = synthesize_zn13x_phantom_pair(pair="2x", lr_shape=(48, 48, 48), seed=19)
    center_um = (38.998, 38.998, 19.499)
    trans_um = (3.10, -2.45, 1.85)
    sim_spec = SimilarityTransform3DSpec(
        matrix_3x3=((0.9864, 0.0, 0.0), (0.0, 0.9864, 0.0), (0.0, 0.0, 0.9864)),
        translation_xyz_um=trans_um,
        center_xyz_um=center_um,
        scale=0.9864,
        versor_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    hr_smooth = gaussian_filter(hr_full, sigma=0.75).astype(np.float32)
    lr_shape_raw = (54, 54, 54)
    c_zyx = np.array([center_um[2], center_um[1], center_um[0]], dtype=np.float64)
    t_zyx = np.array([trans_um[2], trans_um[1], trans_um[0]], dtype=np.float64)
    inv_scale = 1.0 / 0.9864
    a_fwd = np.diag([2.0 * inv_scale, 2.0 * inv_scale, 2.0 * inv_scale])
    b_fwd = ((-t_zyx - c_zyx) * inv_scale + c_zyx) / 0.8125
    lr_drifted = affine_transform(
        hr_smooth, a_fwd, offset=b_fwd, output_shape=lr_shape_raw, order=1, mode="nearest"
    ).astype(np.float32)

    nom_lr, nom_hr, _ = register_multires_pair_3d(
        lr_drifted[:48, :48, :48], hr_full, scale_factor=2
    )
    nom_hr_ds = nom_hr.reshape(nom_lr.shape[0], 2, nom_lr.shape[1], 2, nom_lr.shape[2], 2).mean(
        axis=(1, 3, 5)
    )
    edge_mask_nom = np.ones(nom_lr.shape, dtype=bool)
    edge_mask_nom[8:-8, 8:-8, 8:-8] = False
    r_edge_nom = float(np.corrcoef(nom_lr[edge_mask_nom], nom_hr_ds[edge_mask_nom])[0, 1])

    lr_7dof, hr_7dof, info_7dof = resample_lr_to_hr_grid_7dof(
        lr_drifted,
        hr_full,
        lr_shape_zyx=(40, 40, 40),
        hr_center_zyx=None,
        sp_lo_um=1.625,
        sp_hi_um=0.8125,
        scale_factor=2,
        tx_hr_to_lr=sim_spec,
    )
    assert info_7dof["hr_resampled"] is False
    hr_roi = info_7dof["hr_roi_zyx"]
    assert isinstance(hr_roi, list)
    hz0, hz1 = int(hr_roi[0][0]), int(hr_roi[0][1])
    hy0, hy1 = int(hr_roi[1][0]), int(hr_roi[1][1])
    hx0, hx1 = int(hr_roi[2][0]), int(hr_roi[2][1])
    assert np.array_equal(hr_7dof, hr_full[hz0:hz1, hy0:hy1, hx0:hx1])
    hr_7dof_ds = hr_7dof.reshape(40, 2, 40, 2, 40, 2).mean(axis=(1, 3, 5))
    edge_mask_7d = np.ones((40, 40, 40), dtype=bool)
    edge_mask_7d[8:-8, 8:-8, 8:-8] = False
    r_edge_7dof = float(np.corrcoef(lr_7dof[edge_mask_7d], hr_7dof_ds[edge_mask_7d])[0, 1])
    mse_edge_nom = float(np.mean((nom_lr[edge_mask_nom] - nom_hr_ds[edge_mask_nom]) ** 2))
    mse_edge_7dof = float(np.mean((lr_7dof[edge_mask_7d] - hr_7dof_ds[edge_mask_7d]) ** 2))
    assert r_edge_7dof > 0.94
    assert r_edge_7dof > r_edge_nom + 0.03
    assert mse_edge_7dof < 0.70 * mse_edge_nom

    # 3) Standalone MultimodalBenchmark SimilarityTransform3D .json via `openlsrx download-zn13x --certified-regions`
    raw_dir = tmp_path / "mock_h5"
    raw_dir.mkdir()
    with h5py.File(raw_dir / "169067_8bit_cropped.h5", "w") as f_lo:
        f_lo.create_dataset("data", data=np.clip(lr_drifted * 255.0, 0, 255).astype(np.uint8))
    with h5py.File(raw_dir / "169066_8bit_cropped.h5", "w") as f_hi:
        f_hi.create_dataset("data", data=np.clip(hr_full * 255.0, 0, 255).astype(np.uint8))

    tx_json = tmp_path / "mmb_2x_transform.json"
    tx_json.write_text(json.dumps(sim_spec.to_dict(), indent=2), encoding="utf-8")

    bench_dir = tmp_path / "zn_bench"
    bench_dir.mkdir()
    tifffile.imwrite(bench_dir / "zn13x_2x_lr.tif", np.zeros((20, 20, 20), dtype=np.float32))
    tifffile.imwrite(bench_dir / "zn13x_2x_hr.tif", np.zeros((40, 40, 40), dtype=np.float32))
    (bench_dir / "zn13x_2x_manifest.json").write_text(
        json.dumps(
            {
                "source_kind": "local_i13_preview_h5_registered",
                "is_experimental_xct": True,
                "requested_lr_size": 20,
                "lr_shape": [20, 20, 20],
                "hr_shape": [40, 40, 40],
                "registration": {"scale_factor": 2},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_dl:
        cli_main(
            [
                "download-zn13x",
                "--pair",
                "2x",
                "--lr-size",
                "20",
                "--dest",
                str(bench_dir),
                "--raw-h5-dir",
                str(raw_dir),
                "--certified-regions",
                str(tx_json),
                "--no-offline-fallback",
            ]
        )
    assert exc_dl.value.code == 0
    manifest_new = json.loads((bench_dir / "zn13x_2x_manifest.json").read_text(encoding="utf-8"))
    assert manifest_new["provisional_8bit_preview"] is True
    assert manifest_new["registration"]["registration_schema"] == "multimodal_benchmark_7dof_v2"
    assert manifest_new["registration"]["hr_resampled"] is False
    assert manifest_new["registration"]["fitted_scale_factor"] == pytest.approx(0.9864)


def test_mmb_v2_artifact_cleanup_5rung_t1_rational_sr_splits_and_stage0c_t2_gate(
    tmp_path: Path,
) -> None:
    """Verify conservative XCT cleanup, 5-rung T1 rational SR, benchmark_splits_v1, and Stage 0C T2 gate."""
    import yaml

    from openlsrx.data import (
        ZN13X_T1_PAIRS,
        compute_ruby_beam_hardening_clean_z_mask,
        detrend_local_tomo_cupping_3d,
        rational_scale_pq,
        remove_rings_polar_conservative,
        resolve_zn13x_pair_7dof,
        resolve_zn13x_split_block,
        segment_pores_fringe_guarded,
        verify_stage0c_cross_beamline_gate,
    )
    from openlsrx.inference import super_resolve_volume_orthogonal
    from openlsrx.models import SR3DResNet, SwinIRGenerator, super_resolve_volume_3d

    # 1) Conservative synchrotron XCT cleanup: polar de-ringing + 7-term cupping + ruby shadow + fringe guard
    _, hr_zn = synthesize_zn13x_phantom_pair(pair="2x", lr_shape=(64, 64, 64), seed=11)
    clean_vol = hr_zn[:12].copy()  # (12, 128, 128) 3D tri-phase Zn-13X volume
    rng = np.random.default_rng(99)
    ny, nx = clean_vol.shape[1:]
    yy, xx = np.indices((ny, nx), dtype=np.float32)
    cy, cx = 64.0, 64.0
    rr = np.hypot(yy - cy, xx - cx)
    det_lut = rng.normal(0.0, 0.055, size=120).astype(np.float32)
    det_lut[:8] = 0.0
    rings = det_lut[np.clip(np.rint(rr).astype(int), 0, 119)]
    ringed_vol = clean_vol + rings[None, :, :]

    deringed_vol, rep = remove_rings_polar_conservative(ringed_vol, center_yx=(cy, cx))
    assert deringed_vol.shape == ringed_vol.shape
    assert rep.ring_variance_reduction_pct > 45.0
    assert 0.94 <= rep.edge_sharpness_retention_ratio <= 1.04
    assert float(np.mean((deringed_vol - clean_vol) ** 2)) < 0.85 * float(
        np.mean((ringed_vol - clean_vol) ** 2)
    )

    yn = (yy - cy) / (0.5 * ny)
    xn = (xx - cx) / (0.5 * nx)
    cupping = (0.18 * (yn * yn + xn * xn) ** 2 + 0.08 * (yn * yn + xn * xn)).astype(np.float32)
    cupped = clean_vol + cupping[None, :, :]
    detrended = detrend_local_tomo_cupping_3d(cupped)
    assert float(np.std(detrended - clean_vol)) < 0.35 * float(np.std(cupped - clean_vol))

    # 169067 ruby sphere z-mask excludes z < 300 in 500-slice Zenodo 14245751 preview
    z_mask_169067 = compute_ruby_beam_hardening_clean_z_mask(
        num_z_slices=500,
        sphere_center_z_vox=132.0,
        sphere_radius_vox=155.0,
        safety_margin_vox=12,
    )
    assert not bool(np.any(z_mask_169067[:300]))
    assert bool(np.all(z_mask_169067[300:]))

    # Fringe-guarded pore binarization reduces high-k Fresnel overshoot flips without mutating pred_vol
    fringe_noise = (0.12 * (((yy.astype(int) + xx.astype(int)) % 2) * 2 - 1)).astype(np.float32)
    fringed_pred = np.clip(clean_vol + fringe_noise[None, :, :], 0.0, 1.0)
    fringed_copy = fringed_pred.copy()
    mask_guarded, _ = segment_pores_fringe_guarded(fringed_pred, tie_homogeneity_alpha_vox2=0.6)
    petro_raw = evaluate_petrophysical_metrics(fringed_pred, clean_vol, fringe_guard_alpha=0.0)
    petro_guarded = evaluate_petrophysical_metrics(fringed_pred, clean_vol, fringe_guard_alpha=0.6)
    assert np.array_equal(fringed_pred, fringed_copy)
    assert mask_guarded.shape == fringed_pred.shape
    assert petro_guarded["pore_dice"] >= petro_raw["pore_dice"]

    # 2) Full 5-rung Task T1 transforms + rational 1.6x (8/5) and 2.5x (5/2) end-to-end SR
    assert ZN13X_T1_PAIRS == ("1.6x", "2x", "2.5x", "5x", "8x")
    for p_name in ZN13X_T1_PAIRS:
        tx_p, c_p, m_p = resolve_zn13x_pair_7dof(p_name, tier_section="full_savu_32bit")
        assert 0.95 < tx_p.scale < 1.05 and c_p is not None
        assert m_p["registration_schema"] == "multimodal_benchmark_7dof_v2"

    lr_1p6, hr_1p6 = synthesize_zn13x_phantom_pair(pair="1.6x", lr_shape=(20, 20, 20), seed=5)
    assert lr_1p6.shape == (20, 20, 20) and hr_1p6.shape == (32, 32, 32)
    lr_2p5, hr_2p5 = synthesize_zn13x_phantom_pair(pair="2.5x", lr_shape=(16, 16, 16), seed=6)
    assert lr_2p5.shape == (16, 16, 16) and hr_2p5.shape == (40, 40, 40)

    p_1p6_lr = tmp_path / "r1p6_lr.tif"
    p_1p6_hr = tmp_path / "r1p6_hr.tif"
    tifffile.imwrite(p_1p6_lr, lr_1p6)
    tifffile.imwrite(p_1p6_hr, hr_1p6)
    dm_1p6 = PairedVolumeDataModule(
        low_path=str(p_1p6_lr),
        high_path=str(p_1p6_hr),
        scale_factor=1.6,
        context_slices=3,
        train_frac=0.5,
        val_frac=0.25,
        guard_slices_lr=0,
        patch_size_lr=10,
        batch_size=2,
    )
    dm_1p6.setup("fit")
    batch_lr, batch_hr = next(iter(dm_1p6.train_dataloader()))
    assert tuple(batch_lr.shape) == (2, 3, 10, 10)
    assert tuple(batch_hr.shape) in {(2, 16, 16), (2, 1, 16, 16)}

    swin_1p6 = SwinIRGenerator(
        in_channels=3,
        out_channels=16,
        num_blocks=1,
        window_size=4,
        num_heads=2,
        super_res_factor=1.6,
    ).eval()
    ortho_1p6 = super_resolve_volume_orthogonal(
        swin_1p6, lr_1p6[:10, :10, :10], scale_factor=1.6, context_slices=3
    )
    assert ortho_1p6["fused"].shape == (16, 16, 16)

    sr3d_2p5 = SR3DResNet(super_res_factor=2.5, out_channels=8, num_blocks=1).eval()
    pred_3d_2p5 = super_resolve_volume_3d(
        sr3d_2p5.gen, lr_2p5[:8, :8, :8], scale_factor=2.5, patch_size_lr=8, overlap_lr=2
    )
    assert pred_3d_2p5.shape == (20, 20, 20)

    # 3) benchmark_splits_v1.json block resolution across integer & rational scales
    for pair_k, blk_id, exp_lr in [
        ("2x", "test_00", 128),
        ("2.5x", "test_00", 102),
        ("5x", "test_00", 51),
        ("8x", "test_00", 32),
        ("t2_diad_3x", "test_00", 85),
    ]:
        roi_zyx, blk_meta = resolve_zn13x_split_block(pair_k, blk_id)
        num_p, den_q = rational_scale_pq(
            {"2x": 2.0, "2.5x": 2.5, "5x": 5.0, "8x": 8.0, "t2_diad_3x": 3.0}[pair_k]
        )
        for a0, a1 in roi_zyx:
            assert ((a1 - a0) * den_q) // num_p == exp_lr
        assert blk_meta["split"] == "test"

    # 4) Stage 0C cross-beamline gate for Task T2 (I13-2 169067 -> DIAD k11-43334)
    with pytest.raises(ValueError, match="Stage 0C"):
        prepare_zn13x_benchmark(
            pair="t2_diad_3x", dest_dir=tmp_path / "t2_uncert", prefer_offline=True
        )

    bad_stage0c = tmp_path / "stage0c_bad.yaml"
    bad_stage0c.write_text(
        yaml.safe_dump(
            {
                "max_octant_drift_um": 2.4,
                "stage0c_certified": False,
                "global_pellet_transform": {
                    "matrix_3x3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation_xyz_um": [0.0, 0.0, 0.0],
                    "center_xyz_um": [0.0, 0.0, 0.0],
                    "scale": 1.0,
                    "versor_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rejected"):
        verify_stage0c_cross_beamline_gate(bad_stage0c)

    good_stage0c = tmp_path / "stage0c_good.yaml"
    good_stage0c.write_text(
        yaml.safe_dump(
            {
                "max_octant_drift_um": 0.65,
                "pellet_vs_fiducial_slip_um": 0.42,
                "stage0c_certified": True,
                "provenance": {"diad_astra_flip_x_detected": True},
                "global_pellet_transform": {
                    "matrix_3x3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation_xyz_um": [0.0, 0.0, 0.0],
                    "center_xyz_um": [0.0, 0.0, 0.0],
                    "scale": 1.0,
                    "versor_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ),
        encoding="utf-8",
    )
    h5_t2_dir = tmp_path / "t2_h5"
    h5_t2_dir.mkdir()
    # Asymmetric marker along X in I13-2 (left side x=1..4) and mirrored X in raw Astra DIAD (right side)
    i13_arr = np.zeros((12, 12, 12), dtype=np.float32)
    i13_arr[3:9, 3:9, 1:4] = 8.0e-4
    diad_unflipped = np.zeros((36, 36, 36), dtype=np.float32)
    diad_unflipped[9:27, 9:27, 3:12] = 0.04
    diad_raw_astra = diad_unflipped[:, :, ::-1].copy()
    with h5py.File(h5_t2_dir / "169067_recon.h5", "w") as fl:
        fl.create_dataset("4-TomopyRecon-tomo/data", data=i13_arr)
    with h5py.File(h5_t2_dir / "43334_recon.h5", "w") as fh:
        fh.create_dataset("4-AstraReconGpu-tomo/data", data=diad_raw_astra)

    lr_t2, hr_t2, man_t2 = prepare_zn13x_benchmark(
        pair="t2_diad_3x",
        dest_dir=tmp_path / "t2_cert",
        lr_size=12,
        raw_h5_dir=h5_t2_dir,
        stage0c_path=good_stage0c,
        clean_artifacts=False,
        allow_offline_fallback=False,
    )
    reg_t2 = man_t2["registration"]
    assert isinstance(reg_t2, dict)
    assert reg_t2["stage0c_certified"] is True
    assert reg_t2["diad_astra_flip_x_applied"] is True
    hr_t2_vol = tifffile.imread(hr_t2)
    lr_t2_vol = tifffile.imread(lr_t2)
    assert hr_t2_vol.shape == (36, 36, 36) and lr_t2_vol.shape == (12, 12, 12)
    # After Astra flip_x correction, both LR and HR have high attenuation on the LEFT (low x)!
    assert float(hr_t2_vol[:, :, :15].mean()) > float(hr_t2_vol[:, :, 21:].mean())
    assert float(lr_t2_vol[:, :, :5].mean()) > float(lr_t2_vol[:, :, 7:].mean())
