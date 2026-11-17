"""Unified file-based 3D XCT volume loader and LightningDataModule (TIFF, slice dirs, HDF5)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import h5py
import imageio.v3 as iio
import numpy as np
import pytorch_lightning as pl
import tifffile
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

from .transforms import (
    NormMode,
    PairedSliceDataset,
    VolumeAlignment,
    apply_alignment,
    degrade_xct_volume,
    extract_paired_slices_from_subvolume,
    normalize_volume,
    split_volume_spatially,
)

_NATURAL_RE = re.compile(r"(\d+)")


def _natural_key(path: Path) -> list[int | str]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _NATURAL_RE.split(path.name)]


def load_volume_any(
    path: str | Path,
    h5_dataset_key: str = "4-TomopyRecon-tomo/data",
    transpose_axes: tuple[int, int, int] | None = None,
    max_slices: int | None = None,
) -> np.ndarray:
    """Load a 3D XCT volume from a multi-page TIFF, a directory of 2D slices, or an HDF5/NeXus file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"XCT data path does not exist: {p}")

    if p.is_dir():
        exts = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts], key=_natural_key)
        if not files:
            raise ValueError(f"No 2D image slices found in directory: {p}")
        if max_slices is not None:
            files = files[: int(max_slices)]
        slices = []
        for f in files:
            img = tifffile.imread(f) if f.suffix.lower() in {".tif", ".tiff"} else iio.imread(f)
            if img.ndim == 3:
                img = img[..., 0]
            slices.append(np.asarray(img, dtype=np.float32))
        vol = np.stack(slices, axis=0)
    elif p.suffix.lower() in {".h5", ".hdf5", ".nxs"}:
        with h5py.File(p, "r") as hf:
            if h5_dataset_key not in hf:
                raise KeyError(
                    f"HDF5 dataset key {h5_dataset_key!r} not found in {p}. "
                    f"Top-level keys: {list(hf.keys())}"
                )
            dset = hf[h5_dataset_key]
            vol = np.asarray(
                dset[:max_slices] if max_slices is not None else dset[()], dtype=np.float32
            )
    elif p.suffix.lower() in {".tif", ".tiff"}:
        vol = np.asarray(tifffile.imread(p), dtype=np.float32)
        if max_slices is not None and vol.ndim == 3:
            vol = vol[: int(max_slices)]
    elif p.suffix.lower() == ".npy":
        vol = np.asarray(np.load(p), dtype=np.float32)
        if max_slices is not None and vol.ndim == 3:
            vol = vol[: int(max_slices)]
    else:
        raise ValueError(f"Unsupported XCT volume file format: {p.suffix} ({p})")

    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    if vol.ndim != 3:
        raise ValueError(f"Loaded array must be 3D (D, H, W), got shape {vol.shape} from {p}.")
    if transpose_axes is not None:
        vol = np.transpose(vol, transpose_axes)
    return np.ascontiguousarray(vol, dtype=np.float32)


class PairedVolumeDataModule(pl.LightningDataModule):
    """Config-driven LightningDataModule for LabXCT, Synchrotron HDF5 (Diamond I13), and PoreSpy stacks.

    Supports three operating modes:
    1. **Paired real LR + HR volumes** (``high_path`` + ``low_path``).
    2. **Pre-downsampled orthogonal LR slabs** (``high_path`` + ``low_yz_path`` / ``low_xz_path`` / ``low_xy_path``).
    3. **Single HR volume with controlled XCT degradation** (``high_path`` only, ``degrade_from_hr=True``).
    And during ``predict`` stage, supports standalone ``predict_path`` without requiring training paths!
    """

    def __init__(
        self,
        high_path: str | None = None,
        low_path: str | None = None,
        low_yz_path: str | None = None,
        low_xz_path: str | None = None,
        low_xy_path: str | None = None,
        predict_path: str | None = None,
        h5_dataset_key: str = "4-TomopyRecon-tomo/data",
        transpose_axes: tuple[int, int, int] | None = None,
        scale_factor: int = 2,
        degrade_from_hr: bool = False,
        psf_sigma: float = 0.8,
        degrade_noise_std: float = 0.02,
        norm_mode: NormMode = "percentile",
        low_vmin: float | None = None,
        low_vmax: float | None = None,
        high_vmin: float | None = None,
        high_vmax: float | None = None,
        lower_pct: float = 0.5,
        upper_pct: float = 99.5,
        low_z_range: tuple[int, int] | None = None,
        low_y_range: tuple[int, int] | None = None,
        low_x_range: tuple[int, int] | None = None,
        high_z_range: tuple[int, int] | None = None,
        high_y_range: tuple[int, int] | None = None,
        high_x_range: tuple[int, int] | None = None,
        rotation_deg: float = 0.0,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        split_axis: int = 0,
        guard_slices_lr: int = 1,
        train_axes: tuple[str, ...] = ("yz",),
        test_axis: Literal["parallel", "perpendicular", "all"] = "perpendicular",
        predict_axis: Literal["yz", "xz", "xy"] = "xy",
        patch_size_lr: int | None = None,
        augment: bool = True,
        batch_size: int = 2,
        eval_batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.high_path = high_path
        self.low_path = low_path
        self.low_yz_path = low_yz_path
        self.low_xz_path = low_xz_path
        self.low_xy_path = low_xy_path
        self.predict_path = predict_path
        self.h5_dataset_key = h5_dataset_key
        self.transpose_axes = transpose_axes
        self.scale_factor = int(scale_factor)
        self.degrade_from_hr = bool(degrade_from_hr)
        self.psf_sigma = float(psf_sigma)
        self.degrade_noise_std = float(degrade_noise_std)

        self.norm_mode: NormMode = norm_mode
        self.low_vmin = low_vmin
        self.low_vmax = low_vmax
        self.high_vmin = high_vmin
        self.high_vmax = high_vmax
        self.lower_pct = float(lower_pct)
        self.upper_pct = float(upper_pct)

        self.low_alignment = VolumeAlignment(
            z_range=low_z_range,
            y_range=low_y_range,
            x_range=low_x_range,
            rotation_deg=rotation_deg,
        )
        self.high_alignment = VolumeAlignment(
            z_range=high_z_range,
            y_range=high_y_range,
            x_range=high_x_range,
            rotation_deg=rotation_deg,
        )

        self.train_frac = float(train_frac)
        self.val_frac = float(val_frac)
        self.split_axis = int(split_axis)
        self.guard_slices_lr = int(guard_slices_lr)
        self.train_axes = tuple(train_axes)
        self.test_axis = test_axis
        self.predict_axis = predict_axis
        self.patch_size_lr = patch_size_lr
        self.augment = bool(augment)
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.seed = int(seed)

        self.train_dataset: PairedSliceDataset | None = None
        self.val_dataset: PairedSliceDataset | None = None
        self.test_dataset: PairedSliceDataset | None = None
        self.predict_dataset: PairedSliceDataset | None = None
        self.splits_3d: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _norm(self, vol: np.ndarray, is_high: bool) -> np.ndarray:
        vmin = self.high_vmin if is_high else self.low_vmin
        vmax = self.high_vmax if is_high else self.low_vmax
        mode: NormMode = "fixed" if (vmin is not None and vmax is not None) else self.norm_mode
        return normalize_volume(
            vol,
            mode=mode,
            vmin=vmin,
            vmax=vmax,
            lower_pct=self.lower_pct,
            upper_pct=self.upper_pct,
        )

    def _prepare_paired_3d(self) -> None:
        if self.splits_3d:
            return
        if not self.high_path:
            raise ValueError("high_path is required for fit/validate/test stages.")

        raw_hr = load_volume_any(
            self.high_path,
            h5_dataset_key=self.h5_dataset_key,
            transpose_axes=self.transpose_axes,
        )
        hr_vol = self._norm(apply_alignment(raw_hr, self.high_alignment), is_high=True)
        # Ensure HR dimensions are exact multiples of scale_factor
        s = self.scale_factor
        d_hr = (hr_vol.shape[0] // s) * s
        h_hr = (hr_vol.shape[1] // s) * s
        w_hr = (hr_vol.shape[2] // s) * s
        hr_vol = hr_vol[:d_hr, :h_hr, :w_hr]

        if self.low_path:
            raw_lr = load_volume_any(
                self.low_path,
                h5_dataset_key=self.h5_dataset_key,
                transpose_axes=self.transpose_axes,
            )
            lr_vol = self._norm(apply_alignment(raw_lr, self.low_alignment), is_high=False)
            target_lr_shape = (d_hr // s, h_hr // s, w_hr // s)
            if lr_vol.shape != target_lr_shape:
                factors = tuple(t / c for t, c in zip(target_lr_shape, lr_vol.shape, strict=True))
                lr_vol = np.clip(
                    zoom(lr_vol, zoom=factors, order=1, mode="reflect").astype(np.float32),
                    0.0,
                    1.0,
                )
        elif self.low_yz_path:
            # Anisotropic slab input (D_hr, H_lr, W_lr): downsample along axis 0 to form isotropic LR
            yz_slab = load_volume_any(self.low_yz_path)
            yz_slab = self._norm(yz_slab[:d_hr, : h_hr // s, : w_hr // s], is_high=False)
            lr_vol = yz_slab.reshape(d_hr // s, s, h_hr // s, w_hr // s).mean(axis=1)
        elif self.degrade_from_hr:
            lr_vol = degrade_xct_volume(
                hr_volume=hr_vol,
                scale_factor=s,
                psf_sigma=self.psf_sigma,
                noise_std=self.degrade_noise_std,
                seed=self.seed,
            )
        else:
            raise ValueError(
                "Provide either low_path, low_yz_path, or set degrade_from_hr=True alongside high_path."
            )

        raw_splits = split_volume_spatially(
            hr_volume=hr_vol,
            lr_volume=lr_vol,
            scale_factor=s,
            train_frac=self.train_frac,
            val_frac=self.val_frac,
            split_axis=self.split_axis,
            guard_slices_lr=self.guard_slices_lr,
        )
        self.splits_3d = {k: (v[0], v[1]) for k, v in raw_splits.items() if v[0] is not None}

    def setup(self, stage: str | None = None) -> None:
        if stage == "predict" and self.predict_path:
            raw_pred = load_volume_any(
                self.predict_path,
                h5_dataset_key=self.h5_dataset_key,
                transpose_axes=self.transpose_axes,
            )
            pred_lr = self._norm(apply_alignment(raw_pred, self.low_alignment), is_high=False)
            axis_map = {"yz": 0, "xz": 1, "xy": 2}
            ordered = np.moveaxis(pred_lr, axis_map[self.predict_axis], 0)
            if self.scale_factor > 1:
                ordered = np.clip(
                    zoom(ordered, zoom=(self.scale_factor, 1, 1), order=3, mode="reflect").astype(
                        np.float32
                    ),
                    0.0,
                    1.0,
                )
            self.predict_dataset = PairedSliceDataset(
                ordered, high_res=None, scale_factor=self.scale_factor, augment=False
            )
            return

        self._prepare_paired_3d()

        if stage in (None, "fit"):
            lr_tr, hr_tr = self.splits_3d["train"]
            tr_low, tr_high = extract_paired_slices_from_subvolume(
                lr_tr, hr_tr, scale_factor=self.scale_factor, axes=self.train_axes
            )
            self.train_dataset = PairedSliceDataset(
                tr_low,
                tr_high,
                patch_size_lr=self.patch_size_lr,
                scale_factor=self.scale_factor,
                augment=self.augment,
            )

        if stage in (None, "fit", "validate"):
            lr_va, hr_va = self.splits_3d["val"]
            va_low, va_high = extract_paired_slices_from_subvolume(
                lr_va, hr_va, scale_factor=self.scale_factor, axes=self.train_axes
            )
            self.val_dataset = PairedSliceDataset(
                va_low, va_high, scale_factor=self.scale_factor, augment=False
            )

        if stage in (None, "test"):
            lr_te, hr_te = self.splits_3d["test"]
            if self.test_axis == "parallel":
                axes: tuple[str, ...] = ("yz",)
            elif self.test_axis == "perpendicular":
                axes = ("xz", "xy")
            else:
                axes = ("yz", "xz", "xy")
            te_low, te_high = extract_paired_slices_from_subvolume(
                lr_te, hr_te, scale_factor=self.scale_factor, axes=axes
            )
            self.test_dataset = PairedSliceDataset(
                te_low, te_high, scale_factor=self.scale_factor, augment=False
            )

        if stage in (None, "predict"):
            lr_te, hr_te = self.splits_3d["test"]
            pr_low, pr_high = extract_paired_slices_from_subvolume(
                lr_te, hr_te, scale_factor=self.scale_factor, axes=(self.predict_axis,)
            )
            self.predict_dataset = PairedSliceDataset(
                pr_low, pr_high, scale_factor=self.scale_factor, augment=False
            )

    @property
    def test_volumes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lr_test_subvolume, hr_test_subvolume)`` for 3D volumetric evaluation."""
        if "test" not in self.splits_3d:
            self._prepare_paired_3d()
        return self.splits_3d["test"]

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        assert self.predict_dataset is not None
        return DataLoader(
            self.predict_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
