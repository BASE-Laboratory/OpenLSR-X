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
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .transforms import (
    NormMode,
    PairedSliceDataset,
    VolumeAlignment,
    apply_alignment,
    build_circular_fov_mask,
    build_multi_axis_slice_dataset,
    compute_normalization_bounds,
    degrade_xct_volume,
    extract_paired_slices_from_subvolume,
    minimum_psf_guard_slices_lr,
    normalize_volume,
    sanitize_xct_volume,
    split_volume_spatially,
    zoom_cell_centered,
)

_NATURAL_RE = re.compile(r"(\d+)")


def _natural_key(path: Path) -> list[int | str]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _NATURAL_RE.split(path.name)]


def load_volume_any(
    path: str | Path,
    h5_dataset_key: str = "4-TomopyRecon-tomo/data",
    transpose_axes: tuple[int, int, int] | None = None,
    max_slices: int | None = None,
    roi_slices: tuple[slice, slice, slice] | None = None,
    preserve_2d: bool = False,
) -> np.ndarray:
    """Load a 2D/3D XCT volume from multi-page TIFF, a slice directory, `.npy`, or HDF5/NeXus.

    When loading large synchrotron HDF5 reconstructions (e.g., 53 GB Savu/TomoPy files),
    ``roi_slices`` (specified in post-transpose ``(Z, Y, X)`` coordinates) are mapped back to
    the on-disk HDF5 axis order and sliced *before* materializing the array in host RAM.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"XCT data path does not exist: {p}")

    roi_applied_on_disk = False
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
            slices.append(sanitize_xct_volume(img))
        vol = np.stack(slices, axis=0)
    elif p.suffix.lower() in {".h5", ".hdf5", ".nxs"}:
        with h5py.File(p, "r") as hf:
            if h5_dataset_key not in hf:
                # Fall back to single top-level dataset when unambiguous (e.g., Zenodo cropped .h5)
                top_dsets = [k for k in hf.keys() if isinstance(hf[k], h5py.Dataset)]
                if len(top_dsets) == 1:
                    dset = hf[top_dsets[0]]
                else:
                    raise KeyError(
                        f"HDF5 dataset key {h5_dataset_key!r} not found in {p}. "
                        f"Top-level keys: {list(hf.keys())}"
                    )
            else:
                dset = hf[h5_dataset_key]

            if roi_slices is not None and dset.ndim == 3:
                raw_slices = [slice(None), slice(None), slice(None)]
                if transpose_axes is not None:
                    for out_ax, sl in enumerate(roi_slices):
                        raw_slices[transpose_axes[out_ax]] = sl
                else:
                    raw_slices = list(roi_slices)
                vol = sanitize_xct_volume(np.asarray(dset[tuple(raw_slices)]))
                roi_applied_on_disk = True
            elif max_slices is not None and dset.ndim == 3:
                if transpose_axes is not None:
                    raw_slices = [slice(None), slice(None), slice(None)]
                    raw_slices[transpose_axes[0]] = slice(0, int(max_slices))
                    vol = sanitize_xct_volume(np.asarray(dset[tuple(raw_slices)]))
                else:
                    vol = sanitize_xct_volume(np.asarray(dset[: int(max_slices)]))
            else:
                vol = sanitize_xct_volume(np.asarray(dset[()]))
    elif p.suffix.lower() in {".tif", ".tiff"}:
        vol = sanitize_xct_volume(tifffile.imread(p))
        if max_slices is not None and vol.ndim == 3 and transpose_axes is None:
            vol = vol[: int(max_slices)]
    elif p.suffix.lower() == ".npy":
        vol = sanitize_xct_volume(np.load(p))
        if max_slices is not None and vol.ndim == 3 and transpose_axes is None:
            vol = vol[: int(max_slices)]
    else:
        raise ValueError(f"Unsupported XCT volume file format: {p.suffix} ({p})")

    if vol.ndim == 2:
        if preserve_2d:
            return np.ascontiguousarray(vol, dtype=np.float32)
        vol = vol[np.newaxis, ...]
    if vol.ndim != 3:
        raise ValueError(f"Loaded array must be 3D (D, H, W), got shape {vol.shape} from {p}.")
    if transpose_axes is not None:
        vol = np.transpose(vol, transpose_axes)
        if max_slices is not None and p.suffix.lower() not in {".h5", ".hdf5", ".nxs"}:
            vol = vol[: int(max_slices)]
    if roi_slices is not None and not roi_applied_on_disk:
        vol = vol[roi_slices]
    return np.ascontiguousarray(vol, dtype=np.float32)


def _alignment_to_roi_and_Residual(
    alignment: VolumeAlignment,
) -> tuple[tuple[slice, slice, slice] | None, VolumeAlignment | None]:
    has_crop = any(r is not None for r in (alignment.z_range, alignment.y_range, alignment.x_range))
    roi = (
        (
            slice(*alignment.z_range) if alignment.z_range is not None else slice(None),
            slice(*alignment.y_range) if alignment.y_range is not None else slice(None),
            slice(*alignment.x_range) if alignment.x_range is not None else slice(None),
        )
        if has_crop
        else None
    )
    residual = (
        VolumeAlignment(rotation_deg=alignment.rotation_deg)
        if abs(alignment.rotation_deg) >= 1e-6
        else None
    )
    return roi, residual


def _downsample_slab_along_axis(slab: np.ndarray, axis: int, target_len: int) -> np.ndarray:
    """Cell-centered 1D box-area downsampling of an anisotropic slab along ``axis``."""
    moved = np.moveaxis(np.asarray(slab, dtype=np.float32), axis, -1)
    orig_lead = moved.shape[:-1]
    c_len = moved.shape[-1]
    if c_len == target_len:
        return np.asarray(slab, dtype=np.float32)
    if c_len % target_len == 0:
        f = c_len // target_len
        binned = moved.reshape(*orig_lead, target_len, f).mean(axis=-1, dtype=np.float32)
        return np.moveaxis(binned, -1, axis)
    import torch

    flat = torch.from_numpy(np.ascontiguousarray(moved.reshape(-1, 1, c_len)))
    out = F.interpolate(flat, size=target_len, mode="area").numpy().reshape(*orig_lead, target_len)
    return np.ascontiguousarray(np.moveaxis(out, -1, axis), dtype=np.float32)


class PairedVolumeDataModule(pl.LightningDataModule):
    """Config-driven LightningDataModule for LabXCT, Synchrotron HDF5 (Diamond I13), and PoreSpy stacks.

    Enforces strict spatial split isolation:
    1. Splits raw 3D volumes into disjoint ``train`` / ``val`` / ``test`` slabs *before* computing
       intensity normalization bounds (fitted exclusively on ``train``) or applying synthetic
       3D PSF degradation (applied independently per split so ``val``/``test`` never leak).
    2. Uses cell-centered ``grid_mode=True`` spline resampling across integer and fractional
       ``scale_factor`` values (e.g., ``2``, ``2.5``, ``4``, ``8``).
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
        scale_factor: float = 2,
        degrade_from_hr: bool = False,
        psf_sigma: float = 0.8,
        degrade_noise_std: float = 0.02,
        norm_mode: NormMode = "percentile",
        circular_fov_mask: bool = False,
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
        predict_mode: Literal["orthogonal_3axis", "single_axis"] = "orthogonal_3axis",
        predict_fusion: str = "confidence_fourier",
        patch_size_lr: int | None = None,
        context_slices: int = 1,
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
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError(
                f"context_slices must be a positive odd integer (1, 3, 5, ...), got {context_slices}."
            )
        if float(scale_factor) <= 0.0:
            raise ValueError(f"scale_factor must be > 0, got {scale_factor}.")

        self.high_path = high_path
        self.low_path = low_path
        self.low_yz_path = low_yz_path
        self.low_xz_path = low_xz_path
        self.low_xy_path = low_xy_path
        self.predict_path = predict_path
        self.h5_dataset_key = h5_dataset_key
        self.transpose_axes = transpose_axes
        s_round = int(round(float(scale_factor)))
        self.scale_factor: float = (
            int(s_round)
            if abs(float(scale_factor) - float(s_round)) < 1e-6
            else float(scale_factor)
        )
        self.degrade_from_hr = bool(degrade_from_hr)
        self.psf_sigma = float(psf_sigma)
        self.degrade_noise_std = float(degrade_noise_std)

        self.norm_mode: NormMode = norm_mode
        self.circular_fov_mask = bool(circular_fov_mask)
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
        self.predict_mode = predict_mode
        self.predict_fusion = str(predict_fusion)
        self.patch_size_lr = patch_size_lr
        self.context_slices = int(context_slices)
        self.augment = bool(augment)
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.seed = int(seed)

        self.train_dataset: Dataset[tuple[Tensor, Tensor]] | None = None
        self.val_dataset: Dataset[tuple[Tensor, Tensor]] | None = None
        self.test_dataset: Dataset[tuple[Tensor, Tensor]] | None = None
        self.predict_dataset: PairedSliceDataset | None = None
        self.predict_lr_volume: np.ndarray | None = None
        self.predict_Spatial_shape_hr: tuple[int, int, int] | None = None
        self.splits_3d: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.norm_bounds_hr: tuple[float, float] | None = None
        self.norm_bounds_lr: tuple[float, float] | None = None

    def _fit_bounds(self, vol: np.ndarray, is_high: bool) -> tuple[float, float]:
        vmin = self.high_vmin if is_high else self.low_vmin
        vmax = self.high_vmax if is_high else self.low_vmax
        mode: NormMode = "fixed" if (vmin is not None and vmax is not None) else self.norm_mode
        fov = build_circular_fov_mask(vol.shape) if self.circular_fov_mask else None
        return compute_normalization_bounds(
            vol,
            mode=mode,
            vmin=vmin,
            vmax=vmax,
            lower_pct=self.lower_pct,
            upper_pct=self.upper_pct,
            mask=fov,
        )

    def _apply_bounds(
        self, vol: np.ndarray, bounds: tuple[float, float], is_high: bool
    ) -> np.ndarray:
        vmin = self.high_vmin if is_high else self.low_vmin
        vmax = self.high_vmax if is_high else self.low_vmax
        has_fixed = vmin is not None and vmax is not None
        if not has_fixed and self.norm_mode == "none":
            fov = build_circular_fov_mask(vol.shape) if self.circular_fov_mask else None
            return normalize_volume(vol, mode="none", mask=fov)
        fov = build_circular_fov_mask(vol.shape) if self.circular_fov_mask else None
        return normalize_volume(vol, mode="fixed", vmin=bounds[0], vmax=bounds[1], mask=fov)

    def _aligned_crop_shapes(
        self, raw_hr_shape: tuple[int, int, int]
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        s_float = float(self.scale_factor)
        s_int = int(round(s_float))
        if abs(s_float - float(s_int)) < 1e-6:
            lr_shape = tuple(max(1, dim // s_int) for dim in raw_hr_shape)
            hr_shape = tuple(d * s_int for d in lr_shape)
            return (lr_shape[0], lr_shape[1], lr_shape[2]), (hr_shape[0], hr_shape[1], hr_shape[2])
        lr_shape = tuple(max(1, int(round(dim / s_float))) for dim in raw_hr_shape)
        hr_shape = tuple(
            min(raw_d, max(1, int(round(lr_d * s_float))))
            for raw_d, lr_d in zip(raw_hr_shape, lr_shape, strict=True)
        )
        return (lr_shape[0], lr_shape[1], lr_shape[2]), (hr_shape[0], hr_shape[1], hr_shape[2])

    def _prepare_paired_3d(self) -> None:
        if self.splits_3d:
            return
        if not self.high_path:
            raise ValueError("high_path is required for fit/validate/test stages.")

        hi_roi, hi_res = _alignment_to_roi_and_Residual(self.high_alignment)
        raw_hr = load_volume_any(
            self.high_path,
            h5_dataset_key=self.h5_dataset_key,
            transpose_axes=self.transpose_axes,
            roi_slices=hi_roi,
        )
        hr_aligned = apply_alignment(raw_hr, hi_res)
        target_lr_shape, (d_hr, h_hr, w_hr) = self._aligned_crop_shapes(
            (int(hr_aligned.shape[0]), int(hr_aligned.shape[1]), int(hr_aligned.shape[2]))
        )
        hr_aligned = hr_aligned[:d_hr, :h_hr, :w_hr]

        has_orthogonal_slabs = any(
            p is not None for p in (self.low_yz_path, self.low_xz_path, self.low_xy_path)
        )

        lr_aligned: np.ndarray | None = None
        if self.low_path:
            lo_roi, lo_res = _alignment_to_roi_and_Residual(self.low_alignment)
            raw_lr = load_volume_any(
                self.low_path,
                h5_dataset_key=self.h5_dataset_key,
                transpose_axes=self.transpose_axes,
                roi_slices=lo_roi,
            )
            lr_aligned = apply_alignment(raw_lr, lo_res)
            if lr_aligned.shape != target_lr_shape:
                factors = tuple(
                    t / float(c) for t, c in zip(target_lr_shape, lr_aligned.shape, strict=True)
                )
                lr_aligned = zoom_cell_centered(lr_aligned, zoom_factors=factors, order=1)
        elif has_orthogonal_slabs:
            slab_estimates: list[np.ndarray] = []
            d_lr, h_lr, w_lr = target_lr_shape
            if self.low_yz_path is not None:
                yz = load_volume_any(self.low_yz_path)[:d_hr, :h_lr, :w_lr]
                slab_estimates.append(_downsample_slab_along_axis(yz, axis=0, target_len=d_lr))
            if self.low_xz_path is not None:
                # xz slab stored as (H_hr, D_lr, W_lr) -> moveaxis(0 -> 1) -> (D_lr, H_hr, W_lr)
                xz_raw = load_volume_any(self.low_xz_path)[:h_hr, :d_lr, :w_lr]
                xz = np.moveaxis(xz_raw, 0, 1)
                slab_estimates.append(_downsample_slab_along_axis(xz, axis=1, target_len=h_lr))
            if self.low_xy_path is not None:
                # xy slab stored as (W_hr, D_lr, H_lr) -> moveaxis(0 -> 2) -> (D_lr, H_lr, W_hr)
                xy_raw = load_volume_any(self.low_xy_path)[:w_hr, :d_lr, :h_lr]
                xy = np.moveaxis(xy_raw, 0, 2)
                slab_estimates.append(_downsample_slab_along_axis(xy, axis=2, target_len=w_lr))
            lr_aligned = np.asarray(
                np.mean(np.stack(slab_estimates, axis=0), axis=0, dtype=np.float32),
                dtype=np.float32,
            )
        elif not self.degrade_from_hr:
            raise ValueError(
                "Provide low_path, low_yz_path/low_xz_path/low_xy_path, or set degrade_from_hr=True alongside high_path."
            )

        eff_guard = (
            max(
                self.guard_slices_lr, minimum_psf_guard_slices_lr(self.psf_sigma, self.scale_factor)
            )
            if (lr_aligned is not None and self.psf_sigma > 0.0)
            else self.guard_slices_lr
        )
        if target_lr_shape[self.split_axis] - 2 * eff_guard < 3:
            eff_guard = self.guard_slices_lr

        raw_splits = split_volume_spatially(
            hr_volume=hr_aligned,
            lr_volume=lr_aligned,
            scale_factor=self.scale_factor,
            train_frac=self.train_frac,
            val_frac=self.val_frac,
            split_axis=self.split_axis,
            guard_slices_lr=eff_guard,
        )

        # Fit normalization bounds STRICTLY on train split only (zero test/val intensity leakage)
        _, hr_tr_raw = raw_splits["train"]
        self.norm_bounds_hr = self._fit_bounds(hr_tr_raw, is_high=True)

        if lr_aligned is not None:
            lr_tr_raw = raw_splits["train"][0]
            assert lr_tr_raw is not None
            self.norm_bounds_lr = self._fit_bounds(lr_tr_raw, is_high=False)
            for split_name, (lr_raw_sub, hr_raw_sub) in raw_splits.items():
                assert lr_raw_sub is not None
                hr_sub = self._apply_bounds(hr_raw_sub, self.norm_bounds_hr, is_high=True)
                lr_sub = self._apply_bounds(lr_raw_sub, self.norm_bounds_lr, is_high=False)
                self.splits_3d[split_name] = (lr_sub, hr_sub)
        else:
            # Degrade each spatial split independently AFTER splitting (zero cross-split PSF bleed)
            for idx_split, (split_name, (_, hr_raw_sub)) in enumerate(raw_splits.items()):
                hr_sub = self._apply_bounds(hr_raw_sub, self.norm_bounds_hr, is_high=True)
                lr_sub = degrade_xct_volume(
                    hr_volume=hr_sub,
                    scale_factor=self.scale_factor,
                    psf_sigma=self.psf_sigma,
                    noise_std=self.degrade_noise_std,
                    seed=self.seed + 1009 * (idx_split + 1),
                )
                self.splits_3d[split_name] = (lr_sub, hr_sub)

    def setup(self, stage: str | None = None) -> None:
        if stage == "predict" and self.predict_path:
            lo_roi, lo_res = _alignment_to_roi_and_Residual(self.low_alignment)
            raw_pred = load_volume_any(
                self.predict_path,
                h5_dataset_key=self.h5_dataset_key,
                transpose_axes=self.transpose_axes,
                roi_slices=lo_roi,
            )
            aligned_pred = apply_alignment(raw_pred, lo_res)
            bounds = self.norm_bounds_lr or self._fit_bounds(aligned_pred, is_high=False)
            pred_lr = self._apply_bounds(aligned_pred, bounds, is_high=False)
            self.predict_lr_volume = pred_lr
            self.predict_Spatial_shape_hr = (
                int(round(pred_lr.shape[0] * float(self.scale_factor))),
                int(round(pred_lr.shape[1] * float(self.scale_factor))),
                int(round(pred_lr.shape[2] * float(self.scale_factor))),
            )

            axis_map = {"yz": 0, "xz": 1, "xy": 2}
            ordered = np.moveaxis(pred_lr, axis_map[self.predict_axis], 0)
            ctx_stride = 1
            if float(self.scale_factor) != 1.0:
                ordered = np.clip(
                    zoom_cell_centered(
                        ordered, zoom_factors=(float(self.scale_factor), 1.0, 1.0), order=3
                    ),
                    0.0,
                    1.0,
                )
                ctx_stride = max(1, int(round(float(self.scale_factor))))
            self.predict_dataset = PairedSliceDataset(
                ordered,
                high_res=None,
                scale_factor=self.scale_factor,
                augment=False,
                context_slices=self.context_slices,
                context_stride=ctx_stride,
            )
            return

        self._prepare_paired_3d()

        if stage in (None, "fit"):
            lr_tr, hr_tr = self.splits_3d["train"]
            self.train_dataset = build_multi_axis_slice_dataset(
                lr_tr,
                hr_tr,
                scale_factor=self.scale_factor,
                axes=self.train_axes,
                patch_size_lr=self.patch_size_lr,
                augment=self.augment,
                context_slices=self.context_slices,
            )

        if stage in (None, "fit", "validate"):
            lr_va, hr_va = self.splits_3d["val"]
            self.val_dataset = build_multi_axis_slice_dataset(
                lr_va,
                hr_va,
                scale_factor=self.scale_factor,
                axes=self.train_axes,
                patch_size_lr=None,
                augment=False,
                context_slices=self.context_slices,
            )

        if stage in (None, "test"):
            lr_te, hr_te = self.splits_3d["test"]
            if self.test_axis == "parallel":
                axes: tuple[str, ...] = ("yz",)
            elif self.test_axis == "perpendicular":
                axes = ("xz", "xy")
            else:
                axes = ("yz", "xz", "xy")
            self.test_dataset = build_multi_axis_slice_dataset(
                lr_te,
                hr_te,
                scale_factor=self.scale_factor,
                axes=axes,
                patch_size_lr=None,
                augment=False,
                context_slices=self.context_slices,
            )

        if stage in (None, "predict"):
            lr_te, hr_te = self.splits_3d["test"]
            self.predict_lr_volume = lr_te
            self.predict_Spatial_shape_hr = (
                int(hr_te.shape[0]),
                int(hr_te.shape[1]),
                int(hr_te.shape[2]),
            )
            pr_low, pr_high = extract_paired_slices_from_subvolume(
                lr_te,
                hr_te,
                scale_factor=self.scale_factor,
                axes=(self.predict_axis,),
                context_slices=self.context_slices,
            )
            self.predict_dataset = PairedSliceDataset(
                pr_low,
                pr_high,
                scale_factor=self.scale_factor,
                augment=False,
                context_slices=self.context_slices,
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
