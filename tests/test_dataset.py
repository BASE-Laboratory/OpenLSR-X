"""Unit tests for synthetic PoreSpy generation, file/HDF5 loading, and leakage-free 3D splitting."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import tifffile

from openlsrx.data import (
    PairedVolumeDataModule,
    SyntheticPoreSpyDataModule,
    fetch_or_generate_rock_sample,
    generate_porespy_volume,
    normalize_volume,
    split_volume_spatially,
)


def test_spatial_3d_split_has_zero_voxel_overlap() -> None:
    # Tag every HR slice along axis 0 with a unique integer ID so any leakage is immediately caught
    hr = np.broadcast_to(np.arange(40, dtype=np.float32)[:, None, None], (40, 20, 20)).copy()
    lr = np.broadcast_to(np.arange(20, dtype=np.float32)[:, None, None], (20, 10, 10)).copy()

    splits = split_volume_spatially(
        hr_volume=hr,
        lr_volume=lr,
        scale_factor=2,
        train_frac=0.6,
        val_frac=0.2,
        split_axis=0,
        guard_slices_lr=1,
    )
    tr_ids = set(np.unique(splits["train"][1]).tolist())
    va_ids = set(np.unique(splits["val"][1]).tolist())
    te_ids = set(np.unique(splits["test"][1]).tolist())

    assert tr_ids.isdisjoint(va_ids)
    assert tr_ids.isdisjoint(te_ids)
    assert va_ids.isdisjoint(te_ids)
    # Guard margin of 1 LR slice (= 2 HR slices) separates adjacent splits
    assert min(va_ids) - max(tr_ids) == 3
    assert min(te_ids) - max(va_ids) == 3


def test_generate_porespy_and_datamodule_caching(tmp_path: Path) -> None:
    lr, hr, mask = generate_porespy_volume(
        shape=32,
        feature_regime="high_feature",
        noise_regime="high_noise",
        representation="attenuation",
        scale_factor=2,
        seed=7,
    )
    assert hr.shape == (32, 32, 32)
    assert lr.shape == (16, 16, 16)
    assert mask.dtype == bool
    assert 0.0 <= float(hr.min()) <= float(hr.max()) <= 1.0

    cache_dir = tmp_path / "ps_cache"
    dm = SyntheticPoreSpyDataModule(
        volume_size=32,
        scale_factor=2,
        patch_size_lr=8,
        batch_size=2,
        eval_batch_size=2,
        test_axis="perpendicular",
        cache_dir=str(cache_dir),
    )
    dm.setup("fit")
    dm.setup("test")
    batch_lr, batch_hr = next(iter(dm.train_dataloader()))
    assert batch_lr.shape == (2, 8, 8)
    assert batch_hr.shape == (2, 16, 16)
    assert len(list(cache_dir.glob("*.tif"))) == 2


def test_paired_volume_datamodule_hdf5_and_predict_only(tmp_path: Path) -> None:
    h5_hr = tmp_path / "i13_high.nxs"
    h5_lr = tmp_path / "i13_low.nxs"
    rng = np.random.default_rng(0)
    # Raw (Y, X, Z) layout before transpose_axes=(2, 0, 1) -> (Z, Y, X)
    with h5py.File(h5_hr, "w") as f:
        f.create_dataset(
            "4-TomopyRecon-tomo/data",
            data=rng.normal(0.0, 1e-4, size=(32, 32, 40)).astype(np.float32),
        )
    with h5py.File(h5_lr, "w") as f:
        f.create_dataset(
            "4-TomopyRecon-tomo/data",
            data=rng.normal(0.0, 1e-4, size=(16, 16, 20)).astype(np.float32),
        )

    dm = PairedVolumeDataModule(
        low_path=str(h5_lr),
        high_path=str(h5_hr),
        h5_dataset_key="4-TomopyRecon-tomo/data",
        transpose_axes=(2, 0, 1),
        scale_factor=2,
        norm_mode="percentile",
        batch_size=2,
        eval_batch_size=2,
    )
    dm.setup("fit")
    lr_b, hr_b = next(iter(dm.train_dataloader()))
    assert lr_b.shape == (2, 16, 16)
    assert hr_b.shape == (2, 32, 32)

    # Standalone predict stage should NOT require high_path
    pred_tif = tmp_path / "standalone_lr.tif"
    tifffile.imwrite(pred_tif, normalize_volume(rng.random((16, 16, 16), dtype=np.float32)))
    pred_dm = PairedVolumeDataModule(predict_path=str(pred_tif), scale_factor=2, predict_axis="xz")
    pred_dm.setup("predict")
    p_lr, _ = next(iter(pred_dm.predict_dataloader()))
    assert p_lr.shape[-2:] == (16, 16)


def test_public_rock_offline_fallback(tmp_path: Path) -> None:
    tif_path, manifest = fetch_or_generate_rock_sample(
        "bentheimer_sandstone",
        dest_dir=tmp_path / "rocks",
        shape=(32, 32, 32),
        allow_offline_fallback=True,
        seed=11,
    )
    assert tif_path.is_file()
    vol = tifffile.imread(tif_path)
    assert vol.shape == (32, 32, 32)
    assert manifest["lithology"] == "Sandstone"


def test_slice_directory_alignment_and_local_thickness(tmp_path: Path) -> None:
    from openlsrx.data import (
        VolumeAlignment,
        apply_alignment,
        downsample_orthogonal_slabs,
        load_volume_any,
    )

    lr, hr, _ = generate_porespy_volume(
        shape=24,
        feature_regime="low_feature",
        noise_regime="low_noise",
        representation="local_thickness",
        scale_factor=2,
        seed=5,
    )
    assert hr.shape == (24, 24, 24) and lr.shape == (12, 12, 12)

    slabs = downsample_orthogonal_slabs(
        hr, scale_factor=2, variable_slice_noise=True, contrast_shift=0.05, seed=9
    )
    assert slabs["yz"].shape == (24, 12, 12)
    assert slabs["xz"].shape == (24, 12, 12)
    assert slabs["xy"].shape == (24, 12, 12)

    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()
    for idx in range(16):
        tifffile.imwrite(slice_dir / f"slice_{idx:03d}.tif", hr[idx])

    loaded = load_volume_any(slice_dir, max_slices=12)
    assert loaded.shape == (12, 24, 24)

    aligned = apply_alignment(
        loaded,
        VolumeAlignment(z_range=(2, 10), y_range=(4, 20), x_range=(4, 20), rotation_deg=15.0),
    )
    assert aligned.shape == (8, 16, 16)
