"""Tests for Stage 2: Zn-13X synchrotron aligner, RRDB/RaGAN, SR3DResNet, transport physics, and OpenImpala."""

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

from openlsrx.cli import main as cli_main
from openlsrx.data import (
    ZN13X_I13_RESOLUTIONS,
    PairedVolumeDataModule,
    extract_aligned_slab_from_h5,
    prepare_zn13x_benchmark,
    register_multires_pair_3d,
    synthesize_zn13x_phantom_pair,
)
from openlsrx.inference import super_resolve_volume_orthogonal
from openlsrx.io import export_for_openimpala
from openlsrx.metrics import (
    compute_directional_tortuosity,
    compute_drainage_curve,
    compute_drainage_rmse,
    evaluate_transport_metrics,
    solve_steady_state_diffusion_axis,
)
from openlsrx.models import (
    SRGAN,
    RRDBGenerator,
    SR3DResNet,
    build_3d_subcube_dataloaders,
    super_resolve_volume_3d,
)


def test_analytical_straight_and_constricted_tortuosity() -> None:
    # 1) Straight duct along axis 0 (z): cross-section 4x4 out of 8x8 -> phi = 0.25, tau_z = 1.0
    duct = np.zeros((12, 8, 8), dtype=bool)
    duct[:, 2:6, 2:6] = True
    res = compute_directional_tortuosity(duct, tol=1e-8)

    assert res["porosity"] == pytest.approx(0.25)
    assert res["d_eff_z"] == pytest.approx(0.25, rel=1e-4)
    assert res["tortuosity_z"] == pytest.approx(1.0, rel=1e-4)
    assert res["d_eff_y"] == 0.0
    assert res["d_eff_x"] == 0.0
    assert not np.isfinite(res["tortuosity_y"])
    assert res["percolating_axes_count"] == 1.0

    # 2) Solid block -> zero diffusivity
    solid = np.zeros((8, 8, 8), dtype=bool)
    solid_res = solve_steady_state_diffusion_axis(solid, axis=0)
    assert solid_res["d_eff"] == 0.0
    assert solid_res["percolates"] == 0.0

    # 3) Constricted / baffled duct must have tortuosity strictly > 1.0
    baffled = duct.copy()
    baffled[4, 2:5, 2:6] = False  # leave narrow 1x4 slit
    baffled[8, 3:6, 2:6] = False
    baf_z = solve_steady_state_diffusion_axis(baffled, axis=0, tol=1e-8)
    assert baf_z["percolates"] == 1.0
    assert 0.0 < baf_z["d_eff"] < baf_z["porosity"]
    assert baf_z["tortuosity"] > 1.05


def test_capillary_drainage_and_evaluate_transport_metrics() -> None:
    _, hr = synthesize_zn13x_phantom_pair(pair="2x", lr_shape=(12, 12, 12), seed=7)
    mask = hr < 0.45

    pc_grid, snwp = compute_drainage_curve(mask, n_steps=8, voxel_size=0.8125)
    assert pc_grid.shape == (8,)
    assert snwp.shape == (8,)
    assert np.all(np.diff(snwp) >= -1e-12)
    assert np.all((snwp >= 0.0) & (snwp <= 1.0))

    assert compute_drainage_rmse(mask, mask, n_steps=8) == pytest.approx(0.0, abs=1e-12)

    identical = evaluate_transport_metrics(hr, hr, pore_is_low=True, drainage_steps=8)
    assert identical["d_eff_mae"] == pytest.approx(0.0, abs=1e-7)
    assert identical["drainage_snwp_rmse"] == pytest.approx(0.0, abs=1e-9)


def test_openimpala_export_and_cli_integration(tmp_path: Path) -> None:
    _, hr = synthesize_zn13x_phantom_pair(pair="2x", lr_shape=(10, 10, 10), seed=11)
    pred = np.clip(hr + 0.02, 0.0, 1.0).astype(np.float32)

    tif_p, inp_p = export_for_openimpala(
        pred,
        output_dir=tmp_path / "impala_direct",
        name="zn13x_sample",
        direction="z",
    )
    loaded = tifffile.imread(tif_p)
    assert loaded.dtype == np.uint8
    assert loaded.shape == pred.shape
    assert set(np.unique(loaded)).issubset({0, 1})

    inp_txt = inp_p.read_text(encoding="utf-8")
    assert 'eb2.geom_type = "tiff"' in inp_txt
    assert 'torteff.direction = "z"' in inp_txt
    assert f"amr.n_cell = {pred.shape[2]} {pred.shape[1]} {pred.shape[0]}" in inp_txt

    # CLI evaluate --include-transport --export-openimpala-dir + standalone export-openimpala
    pred_file = tmp_path / "pred.tif"
    hr_file = tmp_path / "hr.tif"
    out_json = tmp_path / "eval_transport.json"
    tifffile.imwrite(pred_file, pred)
    tifffile.imwrite(hr_file, hr)

    with pytest.raises(SystemExit) as exc_eval:
        cli_main(
            [
                "evaluate",
                "--pred",
                str(pred_file),
                "--target",
                str(hr_file),
                "--include-transport",
                "--export-openimpala-dir",
                str(tmp_path / "impala_from_eval"),
                "--output",
                str(out_json),
            ]
        )
    assert exc_eval.value.code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "transport_metrics" in payload
    assert "openimpala_export" in payload
    assert Path(payload["openimpala_export"]["inputs_path"]).is_file()

    with pytest.raises(SystemExit) as exc_exp:
        cli_main(
            [
                "export-openimpala",
                "--volume",
                str(pred_file),
                "--output-dir",
                str(tmp_path / "impala_cli"),
                "--name",
                "cli_rock",
            ]
        )
    assert exc_exp.value.code == 0
    assert (tmp_path / "impala_cli" / "cli_rock_openimpala.inputs").is_file()


def test_zn13x_registration_h5_extraction_and_cli_download(tmp_path: Path) -> None:
    lr_clean, hr_clean = synthesize_zn13x_phantom_pair(pair="2x", lr_shape=(20, 20, 20), seed=21)

    # Apply known 3D translation to LR by padding/cropping so HR aligns at offset (2, 1, 2)
    lr_shifted = np.pad(lr_clean, ((2, 0), (1, 0), (2, 0)), mode="reflect")[:20, :20, :20]
    aligned_lr, aligned_hr, info = register_multires_pair_3d(lr_shifted, hr_clean, scale_factor=2)
    assert aligned_hr.shape == tuple(s * 2 for s in aligned_lr.shape)
    assert float(info["coarse_mid_slice_ssim"]) > 0.65  # type: ignore[arg-type]

    # Mock Savu HDF5 files with minimal ROI coverage around modified test specs
    lo_spec = ZN13X_I13_RESOLUTIONS["1p625um"]
    hi_spec = ZN13X_I13_RESOLUTIONS["0p8125um"]
    lo_h5 = tmp_path / lo_spec.recon_filename
    hi_h5 = tmp_path / hi_spec.recon_filename

    assert lo_spec.roi_1040um_yx is not None and hi_spec.roi_1040um_yx is not None
    lo_shape = (
        lo_spec.roi_1040um_yx[1][0] + 16,
        lo_spec.anchor_z + 10,
        lo_spec.roi_1040um_yx[0][0] + 16,
    )
    hi_shape = (
        hi_spec.roi_1040um_yx[1][0] + 32,
        hi_spec.anchor_z + 20,
        hi_spec.roi_1040um_yx[0][0] + 32,
    )
    with h5py.File(lo_h5, "w") as f_lo:
        dset = f_lo.create_dataset("4-TomopyRecon-tomo/data", shape=lo_shape, dtype="f4")
        dset[
            lo_spec.roi_1040um_yx[1][0] : lo_spec.roi_1040um_yx[1][0] + 16,
            lo_spec.anchor_z - 8 : lo_spec.anchor_z + 8,
            lo_spec.roi_1040um_yx[0][0] : lo_spec.roi_1040um_yx[0][0] + 16,
        ] = 0.0002
    with h5py.File(hi_h5, "w") as f_hi:
        dset = f_hi.create_dataset("4-TomopyRecon-tomo/data", shape=hi_shape, dtype="f4")
        dset[
            hi_spec.roi_1040um_yx[1][0] : hi_spec.roi_1040um_yx[1][0] + 32,
            hi_spec.anchor_z - 16 : lo_spec.anchor_z + 16,
            hi_spec.roi_1040um_yx[0][0] : hi_spec.roi_1040um_yx[0][0] + 32,
        ] = 0.0002

    sub_lo, sub_hi, _ = extract_aligned_slab_from_h5(
        lo_h5, hi_h5, pair="2x", lr_slab_depth=16, lr_crop_yx=16
    )
    assert sub_hi.shape == tuple(s * 2 for s in sub_lo.shape)

    # CLI download-benchmarks for zn13x_2x and zn13x_8x (offline fallback / cache)
    bench_dir = tmp_path / "benchmarks"
    lr_8x, hr_8x, man_8x = prepare_zn13x_benchmark(
        pair="8x", dest_dir=bench_dir, tier="full", lr_size=10, prefer_offline=True, seed=5
    )
    assert lr_8x.is_file() and hr_8x.is_file()
    assert man_8x["scale_factor"] == 8

    with pytest.raises(SystemExit) as exc_dl:
        cli_main(
            [
                "download-benchmarks",
                "--datasets",
                "zn13x_2x",
                "--dest",
                str(bench_dir),
                "--size",
                "16",
                "--offline",
            ]
        )
    assert exc_dl.value.code == 0
    assert (bench_dir / "zn13x_2x_hr.tif").is_file()


def test_rrdb_arbitrary_scale_ragan_and_sr3d_end_to_end(tmp_path: Path) -> None:
    # 1) Arbitrary fractional scale 2.5x (0.8125 -> 0.325 um) & 5.0x (1.625 -> 0.325 um)
    gen_2p5 = RRDBGenerator(out_channels=12, growth_channels=8, num_blocks=2, super_res_factor=2.5)
    x = torch.rand(2, 1, 12, 12)
    y = gen_2p5(x)
    assert y.shape == (2, 1, 30, 30)

    gen_5x = RRDBGenerator(out_channels=12, growth_channels=8, num_blocks=1, super_res_factor=5.0)
    assert gen_5x(torch.rand(1, 1, 8, 8)).shape == (1, 1, 40, 40)

    vol_lr = np.random.default_rng(0).random((12, 12, 12), dtype=np.float32)
    fused_2p5 = super_resolve_volume_orthogonal(gen_2p5, vol_lr, scale_factor=2.5)["fused"]
    assert fused_2p5.shape == (30, 30, 30)

    # 2) End-to-end RRDB + RaGAN Lightning fit (warmup epoch 0 + active RaGAN epoch 1)
    lr_p, hr_p, _ = prepare_zn13x_benchmark(
        pair="2x", dest_dir=tmp_path / "zn", tier="full", lr_size=16, prefer_offline=True, seed=9
    )
    dm = PairedVolumeDataModule(
        low_path=str(lr_p),
        high_path=str(hr_p),
        scale_factor=2,
        norm_mode="none",
        batch_size=2,
        eval_batch_size=2,
        patch_size_lr=12,
        seed=9,
    )
    dm.setup("fit")
    model = SRGAN(
        learning_rate=2e-4,
        adaptive_size=4,
        super_res_factor=2,
        num_blocks=2,
        out_channels=12,
        growth_channels=8,
        backbone="rrdb",
        gan_loss_type="ragan",
        output_activation="clamp",
        pixel_loss_type="l1",
        disc_features=[12, 24],
        pretrain_epochs=1,
        use_adversarial_loss=True,
        use_vgg_loss=False,
        output_dir=str(tmp_path / "rrdb_run"),
        save_val_images=False,
    )
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path / "rrdb_run"),
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    # 3) End-to-end SR3DResNet + seamless raised-cosine 3D sliding-window inference
    lr_tr, hr_tr = dm.splits_3d["train"]
    lr_va, hr_va = dm.splits_3d["val"]
    tr_3d, va_3d = build_3d_subcube_dataloaders(
        lr_tr,
        hr_tr,
        lr_va,
        hr_va,
        scale_factor=2,
        patch_size_lr=min(8, int(min(lr_tr.shape))),
        train_samples=4,
        val_samples=2,
        batch_size=2,
    )
    sr3d = SR3DResNet(super_res_factor=2, out_channels=8, num_blocks=1)
    trainer_3d = pl.Trainer(
        default_root_dir=str(tmp_path / "sr3d_run"),
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer_3d.fit(sr3d, train_dataloaders=tr_3d, val_dataloaders=va_3d)
    sr_vol_3d = super_resolve_volume_3d(
        sr3d.gen,
        np.zeros((14, 14, 14), dtype=np.float32),
        scale_factor=2,
        patch_size_lr=8,
        overlap_lr=2,
    )
    assert sr_vol_3d.shape == (28, 28, 28)
    assert np.all((sr_vol_3d >= 0.0) & (sr_vol_3d <= 1.0))


def test_campaign_aggregator_generates_latex_markdown_and_plot(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "aggregate_campaign",
        Path(__file__).resolve().parents[1] / "scripts" / "benchmarks" / "aggregate_campaign.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    csv_file = tmp_path / "zn13x_synchrotron_results.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "seed",
                "method",
                "psnr",
                "ssim_3d_triplanar",
                "pore_dice",
                "porosity_abs_error",
                "through_plane_anisotropy_error",
                "d_eff_mae",
                "drainage_snwp_rmse",
            ],
        )
        writer.writeheader()
        for seed, psnr in ((42, 24.1), (43, 24.5)):
            writer.writerow(
                {
                    "dataset": "zn13x_2x",
                    "seed": seed,
                    "method": "OpenLSR-X-RRDB-RaGAN-3Axis-Fourier",
                    "psnr": psnr,
                    "ssim_3d_triplanar": 0.88,
                    "pore_dice": 0.91,
                    "porosity_abs_error": 0.008,
                    "through_plane_anisotropy_error": 0.005,
                    "d_eff_mae": 0.012,
                    "drainage_snwp_rmse": 0.031,
                }
            )

    out_dir = tmp_path / "agg"
    info = mod.aggregate_csv_files([csv_file], out_dir)
    assert Path(info["markdown_path"]).is_file()
    assert "±" in Path(info["markdown_path"]).read_text(encoding="utf-8")
    assert "\\toprule" in Path(info["latex_path"]).read_text(encoding="utf-8")
    assert Path(info["figure_path"]).is_file()
