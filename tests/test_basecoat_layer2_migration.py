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

    store = export_volume_ome_zarr(
        vol,
        tmp_path / "recon.ome.zarr",
        voxel_size_um=0.8125,
        include_pore_labels=True,
        threshold_method="fixed",
        fixed_threshold=0.45,
        multiscale_levels=2,
    )
    assert store.is_dir()

    vset = read_ome_zarr(store, level=0, as_set=True)
    assert np.allclose(vset.image.data, vol)
    assert vset.image.spacing_in("um") == pytest.approx((0.8125, 0.8125, 0.8125))
    assert "phase" in vset.labels
    expected_phase = np.where(vol < 0.45, np.uint8(1), np.uint8(0))
    assert np.array_equal(vset.labels["phase"].data, expected_phase)

    # Coarse pyramid level preserves discrete label set {0, 1} via mode pooling
    vset_l1 = read_ome_zarr(store, level=1, as_set=True)
    assert vset_l1.image.shape == (8, 10, 12)
    assert set(np.unique(vset_l1.labels["phase"].data)).issubset({0, 1})

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
