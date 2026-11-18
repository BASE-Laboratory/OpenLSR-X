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
