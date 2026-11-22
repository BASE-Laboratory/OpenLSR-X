"""Regression tests for the Layer 2 basecoat migration in OpenLSR-X."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from basecoat import check_layer_imports, read_ome_zarr, read_volume

from openlsrx import export_for_openimpala, export_volume_ome_zarr, load_volume_any
from openlsrx.cli import main as cli_main
from openlsrx.data.zn13x import (
    SimilarityTransform3DSpec,
    compose_similarity_transforms_3d,
    resolve_zn13x_pair_7dof,
)
from openlsrx.models.sr3d import SR3DResNet

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_layer2_import_boundary_clean() -> None:
    """Verify [tool.basecoat.layers] passes with zero reverse/peer/raw-zarr imports."""
    violations = check_layer_imports(REPO_ROOT, raise_on_violation=True)
    assert violations == []


def test_export_volume_ome_zarr_and_load_volume_any_roundtrip(tmp_path: Path) -> None:
    """Verify OME-Zarr v0.4 export with mode-downsampled phase labels and out-of-core loading."""
    rng = np.random.default_rng(7)
    vol = rng.uniform(0.0, 1.0, size=(16, 20, 24)).astype(np.float32)
    sig_ortho = rng.uniform(0.0, 0.08, size=(16, 20, 24)).astype(np.float32)

    store = export_volume_ome_zarr(
        vol,
        tmp_path / "recon.ome.zarr",
        voxel_size_um=0.8125,
        include_pore_labels=True,
        threshold_method="fixed",
        fixed_threshold=0.45,
        sigma_ortho=sig_ortho,
        multiscale_levels=2,
    )
    assert store.is_dir()

    vset = read_ome_zarr(store, level=0, as_set=True)
    assert np.allclose(vset.image.data, vol)
    assert vset.image.spacing_in("um") == pytest.approx((0.8125, 0.8125, 0.8125))
    assert "phase" in vset.labels
    assert "sigma_ortho" in vset.uncertainties
    assert np.allclose(vset.uncertainties["sigma_ortho"].data, sig_ortho)
    expected_phase = np.where(vol < 0.45, np.uint8(1), np.uint8(0))
    assert np.array_equal(vset.labels["phase"].data, expected_phase)

    # Coarse pyramid level preserves discrete label set {0, 1} via mode pooling & mean-pools sigma_ortho
    vset_l1 = read_ome_zarr(store, level=1, as_set=True)
    assert vset_l1.image.shape == (8, 10, 12)
    assert set(np.unique(vset_l1.labels["phase"].data)).issubset({0, 1})
    assert vset_l1.uncertainties["sigma_ortho"].shape == (8, 10, 12)

    # Out-of-core ROI + max_slices loading through OpenLSR-X's load_volume_any
    full_loaded = load_volume_any(store)
    assert np.allclose(full_loaded, vol)
    roi_loaded = load_volume_any(
        store,
        roi_slices=(slice(2, 10), slice(4, 16), slice(6, 18)),
        max_slices=5,
    )
    assert roi_loaded.shape == (5, 12, 12)
    assert np.allclose(roi_loaded, vol[2:7, 4:16, 6:18])


def test_cli_predict_volume_and_evaluate_ome_zarr(tmp_path: Path) -> None:
    """Verify CLI predict-volume and evaluate operate natively on .ome.zarr stores."""
    rng = np.random.default_rng(11)
    lr = rng.uniform(0.1, 0.9, size=(8, 8, 8)).astype(np.float32)
    hr_ref = rng.uniform(0.1, 0.9, size=(16, 16, 16)).astype(np.float32)

    lr_store = export_volume_ome_zarr(
        lr,
        tmp_path / "lr.ome.zarr",
        voxel_size_um=1.625,
        include_pore_labels=False,
    )
    hr_store = export_volume_ome_zarr(
        hr_ref,
        tmp_path / "hr_ref.ome.zarr",
        voxel_size_um=0.8125,
        include_pore_labels=False,
    )

    net = SR3DResNet(super_res_factor=2, out_channels=8, num_blocks=1)
    ckpt_path = tmp_path / "tiny_3d.ckpt"
    torch.save(
        {
            "state_dict": net.state_dict(),
            "hyper_parameters": dict(net.hparams),
            "pytorch-lightning_version": "2.4.0",
        },
        ckpt_path,
    )

    pred_store = tmp_path / "pred_sr.ome.zarr"
    unc_store = tmp_path / "pred_unc.ome.zarr"
    with pytest.raises(SystemExit) as exc_pred:
        cli_main(
            [
                "predict-volume",
                "--checkpoint",
                str(ckpt_path),
                "--model-type",
                "sr3d",
                "--input",
                str(lr_store),
                "--output",
                str(pred_store),
                "--uncertainty-output",
                str(unc_store),
                "--device",
                "cpu",
            ]
        )
    assert exc_pred.value.code == 0
    assert pred_store.is_dir()
    assert unc_store.is_dir()
    assert read_ome_zarr(pred_store).shape == (16, 16, 16)
    assert read_ome_zarr(unc_store).shape == (16, 16, 16)

    eval_json = tmp_path / "eval_zarr.json"
    with pytest.raises(SystemExit) as exc_eval:
        cli_main(
            [
                "evaluate",
                "--pred",
                str(pred_store),
                "--target",
                str(hr_store),
                "--voxel-size",
                "0.8125",
                "--output",
                str(eval_json),
            ]
        )
    assert exc_eval.value.code == 0
    metrics = json.loads(eval_json.read_text(encoding="utf-8"))
    assert np.isfinite(float(metrics["image_metrics"]["psnr"]))


def test_deprecated_export_for_openimpala_warns_and_writes(tmp_path: Path) -> None:
    """Verify export_for_openimpala emits DeprecationWarning while keeping legacy outputs valid."""
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tif_p, inp_p = export_for_openimpala(
            mask,
            tmp_path / "legacy_impala",
            name="sample",
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert tif_p.is_file() and inp_p.is_file()
    assert np.array_equal(read_volume(tif_p).data, mask.astype(np.uint8))


def test_similarity_spec_basecoat_roundtrip_and_chaining() -> None:
    """Verify SimilarityTransform3DSpec bridges losslessly to basecoat.SimilarityTransform3D."""
    tx_8x, _, meta = resolve_zn13x_pair_7dof(pair="8x", tier_section="full_savu_32bit")
    bc = tx_8x.to_basecoat()
    roundtripped = SimilarityTransform3DSpec.from_basecoat(bc)

    pts_xyz = np.array([[100.0, 200.0, 300.0], [812.5, 650.0, 420.0]], dtype=np.float64)
    assert np.allclose(
        tx_8x.transform_points_xyz_um(pts_xyz),
        roundtripped.transform_points_xyz_um(pts_xyz),
        atol=1e-12,
    )
    comp = compose_similarity_transforms_3d([tx_8x])
    bc_mapped_xyz = bc.to_affine().transform_points(pts_xyz[:, ::-1])[:, ::-1]
    assert np.allclose(comp.transform_points_xyz_um(pts_xyz), bc_mapped_xyz, atol=1e-12)
    assert meta["registration_schema"] == "multimodal_benchmark_7dof_v2"


@pytest.mark.parametrize(
    ("scale", "lr_shape", "expected_hr_shape"),
    [
        (1.6, (10, 10, 10), (16, 16, 16)),
        (2.5, (8, 8, 8), (20, 20, 20)),
    ],
)
def test_sr3dresnet_and_transforms_rational_scales(
    scale: float,
    lr_shape: tuple[int, int, int],
    expected_hr_shape: tuple[int, int, int],
) -> None:
    """Verify SR3DResNet, sliding-window 3D inference, and spatial split snapping on rational scales."""
    from openlsrx.data.transforms import (
        PairedSliceDataset,
        rational_scale_pq,
        split_volume_spatially,
    )
    from openlsrx.models import SR3DResNet, build_3d_subcube_dataloaders, super_resolve_volume_3d

    p, q = rational_scale_pq(scale)
    assert np.isclose(p / q, scale)

    rng = np.random.default_rng(7)
    lr = rng.uniform(0.0, 1.0, size=lr_shape).astype(np.float32)
    hr = rng.uniform(0.0, 1.0, size=expected_hr_shape).astype(np.float32)

    train_dl, val_dl = build_3d_subcube_dataloaders(
        lr,
        hr,
        lr,
        hr,
        scale_factor=scale,
        patch_size_lr=min(lr_shape),
        train_samples=2,
        val_samples=1,
        batch_size=1,
    )
    lr_b, hr_b = next(iter(train_dl))
    model = SR3DResNet(super_res_factor=scale, out_channels=8, num_blocks=1)
    pred_b = model(lr_b)
    assert pred_b.shape == hr_b.shape
    assert len(val_dl) == 1

    sr_vol = super_resolve_volume_3d(
        model.gen,
        lr,
        scale_factor=scale,
        patch_size_lr=max(q, min(lr_shape) - q),
        overlap_lr=q,
        device="cpu",
    )
    assert sr_vol.shape == expected_hr_shape

    lr_big = rng.uniform(0.0, 1.0, size=(q * 6, q * 4, q * 4)).astype(np.float32)
    hr_big = rng.uniform(0.0, 1.0, size=(p * 6, p * 4, p * 4)).astype(np.float32)
    splits = split_volume_spatially(
        hr_big,
        scale_factor=scale,
        lr_volume=lr_big,
        guard_slices_lr=1,
    )
    train_lr, _ = splits["train"]
    assert train_lr is not None
    assert train_lr.shape[0] % q == 0
    ds_slice = PairedSliceDataset(
        rng.uniform(0.0, 1.0, size=(2, q * 4, q * 4)).astype(np.float32),
        rng.uniform(0.0, 1.0, size=(2, p * 4, p * 4)).astype(np.float32),
        scale_factor=scale,
        patch_size_lr=q * 2 + 1,
    )
    sl_lr, sl_hr = ds_slice[0]
    assert sl_lr.shape[-1] % q == 0
    assert int(round(sl_lr.shape[-1] * scale)) == sl_hr.shape[-1]


def test_fringe_guarded_pore_segmentation_and_metrics() -> None:
    """Verify Paganin-like tie-homogeneity damping suppresses high-k Fresnel fringes cleanly."""
    from openlsrx.metrics.petrophysics import evaluate_petrophysical_metrics, segment_pore_space
    from openlsrx.metrics.transport import (
        evaluate_transport_metrics,
        precompute_reference_transport,
    )

    zz, yy, xx = np.meshgrid(
        np.arange(16, dtype=np.float32),
        np.arange(16, dtype=np.float32),
        np.arange(16, dtype=np.float32),
        indexing="ij",
    )
    clean = np.where((xx >= 5.0) & (xx <= 10.0), 0.32, 0.68).astype(np.float32)
    checker = (((zz + yy + xx).astype(int) % 2) * 2 - 1).astype(np.float32) * 0.25
    fringed = clean + checker

    mask_unfiltered, _ = segment_pore_space(fringed, method="otsu", fringe_guard_alpha=0.0)
    mask_guarded, thr = segment_pore_space(fringed, method="otsu", fringe_guard_alpha=0.6)
    mask_true, _ = segment_pore_space(clean, method="otsu", fringe_guard_alpha=0.0)
    assert not np.array_equal(mask_unfiltered, mask_true)
    assert np.array_equal(mask_guarded, mask_true)
    assert 0.35 < thr < 0.65

    petro_unguarded = evaluate_petrophysical_metrics(fringed, clean, fringe_guard_alpha=0.0)
    petro = evaluate_petrophysical_metrics(fringed, clean, fringe_guard_alpha=0.6)
    assert petro["pore_dice"] > petro_unguarded["pore_dice"] + 0.20
    assert petro["pore_dice_auto_otsu"] > 0.99

    ref_t0 = precompute_reference_transport(clean, drainage_steps=4, fringe_guard_alpha=0.0)
    trans_unguarded = evaluate_transport_metrics(
        fringed,
        clean,
        drainage_steps=4,
        precomputed_true=ref_t0,
        fringe_guard_alpha=0.0,
    )
    ref_t = precompute_reference_transport(clean, drainage_steps=4, fringe_guard_alpha=0.6)
    trans = evaluate_transport_metrics(
        fringed,
        clean,
        drainage_steps=4,
        precomputed_true=ref_t,
        fringe_guard_alpha=0.6,
    )
    assert trans["d_eff_rel_error"] is not None and trans_unguarded["d_eff_rel_error"] is not None
    assert float(trans["d_eff_rel_error"]) < float(trans_unguarded["d_eff_rel_error"])
