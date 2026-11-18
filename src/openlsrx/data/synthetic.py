"""Deterministic synthetic 3D porous-media XCT generation and LightningDataModule."""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import porespy as ps
import pytorch_lightning as pl
import tifffile
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .transforms import (
    PairedSliceDataset,
    build_multi_axis_slice_dataset,
    compute_normalization_bounds,
    degrade_xct_volume,
    extract_paired_slices_from_subvolume,
    normalize_volume,
    split_volume_spatially,
)

FeatureRegime = Literal["low_feature", "high_feature"]
NoiseRegime = Literal["low_noise", "high_noise"]
SyntheticRepresentation = Literal["attenuation", "local_thickness"]
TestAxisMode = Literal["parallel", "perpendicular", "all"]

_BLOBS_PARAMS = inspect.signature(ps.generators.blobs).parameters
_THICKNESS_PARAMS = inspect.signature(ps.filters.local_thickness).parameters


def _porespy_blobs(
    shape: tuple[int, int, int],
    blobiness: float,
    porosity: float,
    seed: int,
) -> np.ndarray:
    """Version-compatible wrapper around ``porespy.generators.blobs`` (PoreSpy 2.x and 3.x)."""
    kwargs: dict[str, object] = {
        "shape": list(shape),
        "blobiness": blobiness,
        "porosity": porosity,
    }
    if "seed" in _BLOBS_PARAMS:
        kwargs["seed"] = int(seed)
    else:
        np.random.seed(int(seed))
    if "periodic" in _BLOBS_PARAMS:
        kwargs["periodic"] = False
    return np.asarray(ps.generators.blobs(**kwargs), dtype=bool)


def _porespy_local_thickness(pore_mask: np.ndarray, sizes: int = 25) -> np.ndarray:
    """Version-compatible wrapper around ``porespy.filters.local_thickness``."""
    if "method" in _THICKNESS_PARAMS:
        return np.asarray(
            ps.filters.local_thickness(pore_mask, method="dt", sizes=sizes),
            dtype=np.float32,
        )
    return np.asarray(
        ps.filters.local_thickness(pore_mask, mode="hybrid", sizes=sizes),
        dtype=np.float32,
    )


def _generate_hr_porespy_field(
    vol_shape: tuple[int, int, int],
    feature_regime: FeatureRegime,
    noise_regime: NoiseRegime,
    representation: SyntheticRepresentation,
    hr_noise_std: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate unsplit HR field + boolean pore mask + contrast_shift before LR degradation."""
    rng = np.random.default_rng(seed)

    if feature_regime == "low_feature":
        pore_mask = _porespy_blobs(vol_shape, blobiness=1.0, porosity=0.40, seed=seed)
    elif feature_regime == "high_feature":
        macro = _porespy_blobs(vol_shape, blobiness=0.7, porosity=0.22, seed=seed)
        micro = _porespy_blobs(vol_shape, blobiness=2.2, porosity=0.45, seed=seed + 101)
        pore_mask = np.logical_or(macro, np.logical_and(~macro, micro))
    else:
        raise ValueError(
            f"Unsupported feature_regime={feature_regime!r}. Expected 'low_feature' or 'high_feature'."
        )

    default_hr_sigma = 0.008 if noise_regime == "low_noise" else 0.018
    sigma_hr = default_hr_sigma if hr_noise_std is None else float(hr_noise_std)

    if representation == "attenuation":
        clean = np.where(pore_mask, 0.15, 0.85).astype(np.float32)
        clean = gaussian_filter(clean, sigma=0.45, mode="reflect")
        if sigma_hr > 0.0:
            clean = clean + rng.normal(0.0, sigma_hr, size=clean.shape).astype(np.float32)
        hr_raw = np.clip(clean, 0.0, 1.0).astype(np.float32, copy=False)
        contrast_shift = 0.0
    elif representation == "local_thickness":
        thk = _porespy_local_thickness(pore_mask, sizes=25)
        hr_clean = normalize_volume(thk, mode="minmax")
        if sigma_hr > 0.0:
            hr_raw = (
                hr_clean + rng.normal(0.0, sigma_hr, size=hr_clean.shape).astype(np.float32)
            ).astype(np.float32, copy=False)
        else:
            hr_raw = hr_clean
        contrast_shift = 0.05 if noise_regime == "high_noise" else 0.02
    else:
        raise ValueError(
            f"Unsupported representation={representation!r}. Expected 'attenuation' or 'local_thickness'."
        )
    return hr_raw, pore_mask, contrast_shift


def generate_porespy_volume(
    shape: int | tuple[int, int, int] = 128,
    feature_regime: FeatureRegime = "high_feature",
    noise_regime: NoiseRegime = "low_noise",
    representation: SyntheticRepresentation = "attenuation",
    scale_factor: float = 2,
    hr_noise_std: float | None = None,
    lr_noise_std: float | None = None,
    psf_sigma: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a deterministic 3D porous-media benchmark tuple ``(lr_vol, hr_vol, pore_mask)``."""
    vol_shape = (shape, shape, shape) if isinstance(shape, int) else tuple(int(x) for x in shape)
    if len(vol_shape) != 3:
        raise ValueError(f"shape must be an int or 3-tuple, got {shape!r}.")

    default_lr_sigma = 0.015 if noise_regime == "low_noise" else 0.055
    sigma_lr = default_lr_sigma if lr_noise_std is None else float(lr_noise_std)

    hr_raw, pore_mask, contrast_shift = _generate_hr_porespy_field(
        vol_shape=vol_shape,
        feature_regime=feature_regime,
        noise_regime=noise_regime,
        representation=representation,
        hr_noise_std=hr_noise_std,
        seed=seed,
    )
    hr_vol = (
        normalize_volume(hr_raw, mode="minmax") if representation == "local_thickness" else hr_raw
    )

    lr_vol = degrade_xct_volume(
        hr_volume=hr_vol,
        scale_factor=scale_factor,
        psf_sigma=psf_sigma,
        noise_std=sigma_lr,
        contrast_shift=contrast_shift,
        seed=seed + 999,
    )
    return lr_vol, hr_vol, pore_mask


class SyntheticPoreSpyDataModule(pl.LightningDataModule):
    """Leakage-free 3D synthetic PoreSpy LightningDataModule for XCT super-resolution.

    Enforces strict spatial isolation across ``train`` / ``val`` / ``test``:
    - Splits the 3D HR volume spatially *before* fitting ``minmax`` bounds (on ``train`` only)
      and *before* applying 3D PSF Gaussian blur + downsampling independently per split, so no
      ``val``/``test`` voxel or intensity bound ever leaks into ``train``.
    """

    def __init__(
        self,
        volume_size: int | tuple[int, int, int] = 128,
        feature_regime: FeatureRegime = "high_feature",
        noise_regime: NoiseRegime = "low_noise",
        representation: SyntheticRepresentation = "attenuation",
        scale_factor: float = 2,
        psf_sigma: float = 0.8,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        split_axis: int = 0,
        guard_slices_lr: int = 1,
        train_axes: tuple[str, ...] = ("yz",),
        test_axis: TestAxisMode = "perpendicular",
        predict_axis: Literal["yz", "xz", "xy"] = "xy",
        predict_mode: Literal["orthogonal_3axis", "single_axis"] = "orthogonal_3axis",
        predict_fusion: str = "confidence_fourier",
        patch_size_lr: int | None = None,
        context_slices: int = 1,
        augment: bool = True,
        batch_size: int = 4,
        eval_batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        seed: int = 42,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError(
                f"context_slices must be a positive odd integer (1, 3, 5, ...), got {context_slices}."
            )

        self.volume_size = volume_size
        self.feature_regime: FeatureRegime = feature_regime
        self.noise_regime: NoiseRegime = noise_regime
        self.representation: SyntheticRepresentation = representation
        s_round = int(round(float(scale_factor)))
        self.scale_factor: float = (
            int(s_round)
            if abs(float(scale_factor) - float(s_round)) < 1e-6
            else float(scale_factor)
        )
        self.psf_sigma = float(psf_sigma)
        self.train_frac = float(train_frac)
        self.val_frac = float(val_frac)
        self.split_axis = int(split_axis)
        self.guard_slices_lr = int(guard_slices_lr)
        self.train_axes = tuple(train_axes)
        self.test_axis: TestAxisMode = test_axis
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
        self.cache_dir = cache_dir

        self.train_dataset: Dataset[tuple[Tensor, Tensor]] | None = None
        self.val_dataset: Dataset[tuple[Tensor, Tensor]] | None = None
        self.test_dataset: Dataset[tuple[Tensor, Tensor]] | None = None
        self.predict_dataset: PairedSliceDataset | None = None
        self.predict_lr_volume: np.ndarray | None = None
        self.predict_Spatial_shape_hr: tuple[int, int, int] | None = None

        self.splits_3d: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _prepare_leakage_free_splits(self) -> None:
        vol_shape: tuple[int, int, int] = (
            (self.volume_size, self.volume_size, self.volume_size)
            if isinstance(self.volume_size, int)
            else (int(self.volume_size[0]), int(self.volume_size[1]), int(self.volume_size[2]))
        )
        sigma_lr = 0.015 if self.noise_regime == "low_noise" else 0.055
        contrast_shift = (
            (0.05 if self.noise_regime == "high_noise" else 0.02)
            if self.representation == "local_thickness"
            else 0.0
        )

        hr_raw: np.ndarray | None = None
        if self.cache_dir is not None:
            cache_root = Path(self.cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            spec = json.dumps(
                {
                    "volume_size": self.volume_size,
                    "feature_regime": self.feature_regime,
                    "noise_regime": self.noise_regime,
                    "representation": self.representation,
                    "scale_factor": self.scale_factor,
                    "psf_sigma": self.psf_sigma,
                    "seed": self.seed,
                },
                sort_keys=True,
            )
            digest = hashlib.sha256(spec.encode("utf-8")).hexdigest()[:12]
            hr_file = cache_root / f"porespy_hr_{digest}.tif"
            lr_file = cache_root / f"porespy_lr_{digest}.tif"
            lock_file = cache_root / f".porespy_{digest}.lock"
            with lock_file.open("w") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                if hr_file.is_file() and lr_file.is_file():
                    hr_raw = tifffile.imread(hr_file).astype(np.float32)
                else:
                    hr_gen, _, _ = _generate_hr_porespy_field(
                        vol_shape=vol_shape,
                        feature_regime=self.feature_regime,
                        noise_regime=self.noise_regime,
                        representation=self.representation,
                        hr_noise_std=None,
                        seed=self.seed,
                    )
                    lr_preview = degrade_xct_volume(
                        hr_volume=normalize_volume(hr_gen, mode="minmax")
                        if self.representation == "local_thickness"
                        else hr_gen,
                        scale_factor=self.scale_factor,
                        psf_sigma=self.psf_sigma,
                        noise_std=sigma_lr,
                        contrast_shift=contrast_shift,
                        seed=self.seed + 999,
                    )
                    pid = os.getpid()
                    hr_tmp = cache_root / f".porespy_hr_{digest}.{pid}.tmp"
                    lr_tmp = cache_root / f".porespy_lr_{digest}.{pid}.tmp"
                    tifffile.imwrite(hr_tmp, hr_gen, photometric="minisblack")
                    tifffile.imwrite(lr_tmp, lr_preview, photometric="minisblack")
                    hr_tmp.replace(hr_file)
                    lr_tmp.replace(lr_file)
                    hr_raw = hr_gen

        if hr_raw is None:
            hr_raw, _, _ = _generate_hr_porespy_field(
                vol_shape=vol_shape,
                feature_regime=self.feature_regime,
                noise_regime=self.noise_regime,
                representation=self.representation,
                hr_noise_std=None,
                seed=self.seed,
            )

        raw_splits = split_volume_spatially(
            hr_volume=hr_raw,
            lr_volume=None,
            scale_factor=self.scale_factor,
            train_frac=self.train_frac,
            val_frac=self.val_frac,
            split_axis=self.split_axis,
            guard_slices_lr=self.guard_slices_lr,
        )

        # Fit normalization strictly on train split when local_thickness is used
        if self.representation == "local_thickness":
            tr_lo, tr_hi = compute_normalization_bounds(raw_splits["train"][1], mode="minmax")
        else:
            tr_lo, tr_hi = 0.0, 1.0

        for idx_split, (split_name, (_, hr_sub_raw)) in enumerate(raw_splits.items()):
            hr_sub = (
                normalize_volume(hr_sub_raw, mode="fixed", vmin=tr_lo, vmax=tr_hi)
                if self.representation == "local_thickness"
                else np.clip(hr_sub_raw, 0.0, 1.0).astype(np.float32, copy=False)
            )
            lr_sub = degrade_xct_volume(
                hr_volume=hr_sub,
                scale_factor=self.scale_factor,
                psf_sigma=self.psf_sigma,
                noise_std=sigma_lr,
                contrast_shift=contrast_shift,
                seed=self.seed + 999 + 131 * idx_split,
            )
            self.splits_3d[split_name] = (lr_sub, hr_sub)

    def setup(self, stage: str | None = None) -> None:
        if not self.splits_3d:
            self._prepare_leakage_free_splits()

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
        """Return ``(lr_test_subvolume, hr_test_subvolume)`` for 3D volumetric benchmarks."""
        if "test" not in self.splits_3d:
            self.setup("test")
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
