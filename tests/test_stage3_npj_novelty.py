"""Analytical and integration tests for Stage 3 npj Computational Materials novelty upgrades.

Covers:
1. Orthogonal epistemic uncertainty & outlier-suppressing Confidence-Fourier fusion
2. 2.5D multi-slice context windowing without cross-axis boundary bleeding
3. Physics, topology (Soft-clDice), gradient, and focal frequency losses (2D, 2.5D, and 3D)
4. SwinIRGenerator shifted-window 2D self-attention across odd sizes & fractional scales
5. 2D FRC and 3D FSC synchrotron beamline resolution metrology
6. Tri-phase Zn-13X ion-exchange chemistry & uncertainty-propagated tortuosity bounds
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import tifffile
import torch
from scipy.ndimage import gaussian_filter

from openlsrx.cli import main as cli_main
from openlsrx.data import PairedSliceDataset, PairedVolumeDataModule, synthesize_zn13x_phantom_pair
from openlsrx.data.transforms import extract_paired_slices_from_subvolume
from openlsrx.inference import (
    AxisName,
    compute_orthogonal_uncertainty,
    fuse_orthogonal_volumes,
    super_resolve_volume_orthogonal,
)
from openlsrx.losses import (
    AttenuationConservationLoss,
    FocalFrequencyLoss,
    SobelGradientLoss,
    SoftCLDiceLoss,
)
from openlsrx.metrics import (
    compute_frc_2d,
    compute_fsc_3d,
    compute_tortuosity_uncertainty_bounds,
    evaluate_spectral_metrics,
    evaluate_triphase_zn13x_metrics,
)
from openlsrx.models import (
    SRGAN,
    SR3DResNet,
    SwinIRGenerator,
    build_3d_subcube_dataloaders,
)


@pytest.mark.parametrize("mode", ["confidence", "confidence_fourier"])
def test_orthogonal_uncertainty_and_confidence_fusion_outlier_suppression(mode: str) -> None:
    rng = np.random.default_rng(7)
    clean = np.clip(0.5 + 0.25 * np.sin(np.linspace(0, 6, 16))[:, None, None], 0.0, 1.0).astype(
        np.float32
    )
    clean = np.broadcast_to(clean, (16, 16, 16)).copy()

    # Identity check when all three orthogonal axes agree
    unc_ident = compute_orthogonal_uncertainty({"yz": clean, "xz": clean, "xy": clean})
    assert np.allclose(unc_ident["std"], 0.0, atol=1e-7)
    assert np.allclose(unc_ident["range"], 0.0, atol=1e-7)
    fused_ident = fuse_orthogonal_volumes(
        {"yz": clean, "xz": clean, "xy": clean},
        mode=mode,  # type: ignore[arg-type]
        psf_cutoff=0.25,
    )
    assert np.allclose(fused_ident, clean, atol=1e-5)

    # Corrupt a single axis with localized blur/artifact while two perpendicular axes agree
    corrupted_xy = clean.copy()
    corrupted_xy[6:10, 6:10, 6:10] = np.clip(
        clean[6:10, 6:10, 6:10] + 0.35 + 0.02 * rng.standard_normal((4, 4, 4)), 0.0, 1.0
    ).astype(np.float32)

    vols: dict[AxisName, np.ndarray] = {"yz": clean, "xz": clean, "xy": corrupted_xy}
    unc = compute_orthogonal_uncertainty(vols)
    # Epistemic std must localize strictly inside the corrupted sub-cube
    assert float(unc["std"][6:10, 6:10, 6:10].mean()) > 0.15
    assert float(unc["std"][:4, :4, :4].mean()) == 0.0

    fused_mean = fuse_orthogonal_volumes(vols, mode="mean")
    fused_conf = fuse_orthogonal_volumes(vols, mode="confidence")
    rmse_mean = float(np.sqrt(np.mean((fused_mean - clean) ** 2)))
    rmse_conf = float(np.sqrt(np.mean((fused_conf - clean) ** 2)))
    assert rmse_conf < 0.25 * rmse_mean


def test_25d_context_slice_extraction_and_auto_inference() -> None:
    # Construct distinct values per axis so cross-axis bleeding would be immediately caught
    lr_sub = np.zeros((6, 6, 6), dtype=np.float32)
    lr_sub[:, :, :] = np.linspace(0.1, 0.9, 6, dtype=np.float32)[:, None, None]
    hr_sub = np.repeat(np.repeat(np.repeat(lr_sub, 2, axis=0), 2, axis=1), 2, axis=2)

    low_ctx, high_slices = extract_paired_slices_from_subvolume(
        lr_sub, hr_sub, scale_factor=2, axes=("yz", "xz"), context_slices=3
    )
    assert low_ctx.shape == (24, 3, 6, 6)
    assert high_slices.shape == (24, 12, 12)

    # First xz slice (index 12) is constant in z along each row and must NOT contain last yz slice!
    first_xz = low_ctx[12]
    assert np.allclose(first_xz[0], first_xz[2], atol=1e-5)

    ds = PairedSliceDataset(
        low_ctx, high_slices, patch_size_lr=4, scale_factor=2, augment=True, context_slices=3
    )
    lr_item, hr_item = ds[0]
    assert lr_item.shape == (3, 4, 4)
    assert hr_item.shape == (8, 8)

    with pytest.raises(ValueError, match="odd integer"):
        PairedSliceDataset(lr_sub, high_res=None, context_slices=2)

    # SwinIRGenerator with context_slices=3 should be auto-detected by super_resolve_volume_orthogonal
    swin_25d = SwinIRGenerator(
        in_channels=3,
        out_channels=16,
        num_blocks=1,
        depth_per_block=2,
        num_heads=4,
        window_size=4,
        super_res_factor=2,
    )
    sr_out = super_resolve_volume_orthogonal(
        swin_25d,
        lr_sub,
        scale_factor=2,
        fusion="confidence_fourier",
        return_uncertainty=True,
    )
    assert sr_out["fused"].shape == (12, 12, 12)
    assert sr_out["uncertainty_std"].shape == (12, 12, 12)


def test_physics_topology_gradient_and_frequency_losses_analytical() -> None:
    # 1) AttenuationConservationLoss: exact zero when sub-voxels average to LR
    cons_loss = AttenuationConservationLoss(psf_sigma=0.0)
    lr_2d = torch.full((2, 3, 8, 8), 0.5, dtype=torch.float32)  # 2.5D 3-channel input
    # Sub-voxel +/- 0.2 checkerboard averages to 0.5 in every 2x2 block!
    hr_2d = torch.full((2, 1, 16, 16), 0.5, dtype=torch.float32)
    hr_2d[:, :, 0::2, 0::2] = 0.7
    hr_2d[:, :, 0::2, 1::2] = 0.3
    hr_2d[:, :, 1::2, 0::2] = 0.3
    hr_2d[:, :, 1::2, 1::2] = 0.7
    assert float(cons_loss(hr_2d, lr_2d)) < 1e-6
    assert float(cons_loss(hr_2d + 0.15, lr_2d)) > 0.14

    # 3D conservation check with PSF > 0
    cons_3d = AttenuationConservationLoss(psf_sigma=0.5)
    lr_3d = torch.full((1, 1, 6, 6, 6), 0.4, dtype=torch.float32)
    hr_3d = torch.full((1, 1, 12, 12, 12), 0.4, dtype=torch.float32)
    assert float(cons_3d(hr_3d, lr_3d)) < 1e-6

    # 2) SoftCLDiceLoss: severing a narrow percolating pore throat increases loss much more than bulk tweak
    cl_loss = SoftCLDiceLoss(num_iter=4, temperature=20.0, pore_is_low=True, threshold=0.5)
    target = torch.full((1, 1, 24, 24), 0.85, dtype=torch.float32)  # solid = 0.85
    target[:, :, 4:20, 4:8] = 0.15  # left chamber (pore = 0.15)
    target[:, :, 4:20, 16:20] = 0.15  # right chamber
    target[:, :, 11:13, 8:16] = 0.15  # narrow 2-px connecting throat (16 pixels total)

    self_loss = float(cl_loss(target, target))
    assert self_loss < 2e-3

    broken_throat = target.clone()
    broken_throat[:, :, 11:13, 10:14] = 0.85  # block 8 pixels across the throat!

    bulk_pert = target.clone()
    bulk_pert[:, :, 6:8, 5:9] = 0.85  # flip exact same 8 pixels inside wide left chamber!

    loss_broken = float(cl_loss(broken_throat, target))
    loss_bulk = float(cl_loss(bulk_pert, target))
    assert loss_broken > loss_bulk * 1.5
    assert loss_broken > self_loss * 10.0

    # 3) SobelGradientLoss & FocalFrequencyLoss differentiability
    grad_loss = SobelGradientLoss()
    ffl_loss = FocalFrequencyLoss()
    pred_req = (target + 0.05 * torch.randn_like(target)).clamp(0.0, 1.0).requires_grad_(True)
    total = grad_loss(pred_req, target) + ffl_loss(pred_req, target)
    total.backward()
    assert pred_req.grad is not None
    assert torch.isfinite(pred_req.grad).all()


@pytest.mark.parametrize("scale", [2, 2.5, 4])
def test_swinir_generator_arbitrary_sizes_and_scales(scale: float) -> None:
    gen = SwinIRGenerator(
        in_channels=3,
        out_channels=16,
        num_blocks=2,
        depth_per_block=2,
        num_heads=4,
        window_size=8,
        super_res_factor=scale,
    )
    # Odd spatial dimensions (13 x 15) exercise automatic replicate-pad & crop around window_size=8
    x = torch.rand(2, 3, 13, 15, requires_grad=True)
    out = gen(x)
    exp_h = int(round(13 * scale))
    exp_w = int(round(15 * scale))
    assert out.shape == (2, 1, exp_h, exp_w)
    out.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_fsc_3d_and_frc_2d_resolution_metrology() -> None:
    rng = np.random.default_rng(19)
    hr = rng.random((24, 24, 24), dtype=np.float32)

    # Perfect reconstruction -> full Nyquist recovery (effective resolution == voxel_size_um)
    fsc_ident = compute_fsc_3d(hr, hr, voxel_size_um=0.8125)
    assert fsc_ident["spectral_auc"] > 0.99
    assert np.isclose(fsc_ident["nyquist_fraction_half_bit"], 1.0, atol=1e-5)
    assert np.isclose(fsc_ident["effective_resolution_half_bit_um"], 0.8125, atol=1e-4)

    # Mild vs heavy Gaussian blur -> monotonic loss of FSC cutoff wavenumber
    mild = gaussian_filter(hr, sigma=0.9).astype(np.float32)
    heavy = gaussian_filter(hr, sigma=2.2).astype(np.float32)
    s_mild = evaluate_spectral_metrics(mild, hr, voxel_size_um=0.8125)
    s_heavy = evaluate_spectral_metrics(heavy, hr, voxel_size_um=0.8125)
    assert s_mild["fsc_auc"] > s_heavy["fsc_auc"]
    assert s_mild["effective_resolution_half_bit_um"] <= s_heavy["effective_resolution_half_bit_um"]

    frc_2d = compute_frc_2d(mild[12], hr[12], voxel_size_um=0.8125)
    assert 0.0 < frc_2d["spectral_auc"] <= 1.0


def test_triphase_zn13x_and_tortuosity_uncertainty_and_lightning_fit(tmp_path: Path) -> None:
    lr_vol, hr_vol = synthesize_zn13x_phantom_pair(pair="2x", lr_shape=(16, 16, 16), seed=42)
    tif_lr = tmp_path / "lr.tif"
    tif_hr = tmp_path / "hr.tif"
    tifffile.imwrite(tif_lr, lr_vol)
    tifffile.imwrite(tif_hr, hr_vol)

    # Tri-phase self-check
    tri_self = evaluate_triphase_zn13x_metrics(hr_vol, hr_vol)
    assert np.isclose(tri_self["triphase_macro_dice"], 1.0, atol=1e-6)
    assert tri_self["zn13x_fraction_true"] > 0.05
    assert tri_self["na13x_fraction_true"] > 0.05

    # End-to-end 2.5D SwinIR + Physics losses Lightning fit & uncertainty-propagated transport
    dm = PairedVolumeDataModule(
        low_path=str(tif_lr),
        high_path=str(tif_hr),
        scale_factor=2,
        context_slices=3,
        norm_mode="none",
        train_frac=0.6,
        val_frac=0.2,
        guard_slices_lr=1,
        patch_size_lr=12,
        batch_size=2,
        eval_batch_size=2,
        seed=42,
    )
    model = SRGAN(
        learning_rate=2e-4,
        adaptive_size=4,
        super_res_factor=2,
        num_blocks=1,
        out_channels=16,
        context_slices=3,
        backbone="swinir",
        swin_window_size=4,
        swin_num_heads=4,
        output_activation="clamp",
        pixel_loss_type="l1",
        disc_features=[8, 16],
        use_adversarial_loss=False,
        use_vgg_loss=False,
        conservation_loss_weight=0.2,
        topology_loss_weight=0.05,
        gradient_loss_weight=0.1,
        ffl_loss_weight=0.05,
        output_dir=str(tmp_path / "swin_run"),
        save_val_images=True,
        max_val_images_saved=1,
    )
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    lr_te, hr_te = dm.test_volumes
    sr_res = super_resolve_volume_orthogonal(
        model.gen,
        lr_te,
        scale_factor=2,
        fusion="confidence_fourier",
        return_uncertainty=True,
    )
    bounds = compute_tortuosity_uncertainty_bounds(
        sr_res["fused"], sr_res["uncertainty_std"], true_vol=hr_te
    )
    assert bounds["d_eff_lower"] is not None and bounds["d_eff_upper"] is not None
    assert float(bounds["d_eff_lower"]) <= float(bounds["d_eff_upper"]) + 1e-8

    # Also verify 3D physics losses inside SR3DResNet
    lr_tr, hr_tr = dm.splits_3d["train"]
    lr_va, hr_va = dm.splits_3d["val"]
    tr_3d, va_3d = build_3d_subcube_dataloaders(
        lr_tr, hr_tr, lr_va, hr_va, scale_factor=2, patch_size_lr=4, train_samples=2, val_samples=1
    )
    sr3d = SR3DResNet(
        super_res_factor=2,
        out_channels=8,
        num_blocks=1,
        conservation_loss_weight=0.2,
        topology_loss_weight=0.05,
        gradient_loss_weight=0.05,
        ffl_loss_weight=0.05,
    )
    trainer_3d = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer_3d.fit(sr3d, train_dataloaders=tr_3d, val_dataloaders=va_3d)

    # CLI evaluate smoke with --include-spectral and --include-triphase
    pred_tif = tmp_path / "sr_pred.tif"
    target_tif = tmp_path / "hr_target.tif"
    out_json = tmp_path / "eval_stage3.json"
    tifffile.imwrite(pred_tif, sr_res["fused"])
    tifffile.imwrite(target_tif, hr_te)
    with pytest.raises(SystemExit) as exc_info:
        cli_main(
            [
                "evaluate",
                "--pred",
                str(pred_tif),
                "--target",
                str(target_tif),
                "--voxel-size",
                "0.8125",
                "--include-spectral",
                "--include-triphase",
                "--output",
                str(out_json),
            ]
        )
    assert exc_info.value.code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "spectral_metrics" in payload
    assert "triphase_zn13x_metrics" in payload
