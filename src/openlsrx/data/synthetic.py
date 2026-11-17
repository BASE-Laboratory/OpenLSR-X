"""Deterministic synthetic 3D porous-media XCT generation and LightningDataModule."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Literal

import numpy as np
import porespy as ps
import pytorch_lightning as pl
import tifffile
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from .transforms import (
    PairedSliceDataset,
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


def generate_porespy_volume(
    shape: int | tuple[int, int, int] = 128,
    feature_regime: FeatureRegime = "high_feature",
    noise_regime: NoiseRegime = "low_noise",
    representation: SyntheticRepresentation = "attenuation",
    scale_factor: int = 2,
    hr_noise_std: float | None = None,
    lr_noise_std: float | None = None,
    psf_sigma: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a deterministic 3D porous-media benchmark tuple ``(lr_vol, hr_vol, pore_mask)``.

    Args:
        shape: Isotropic integer side length or ``(D, H, W)`` tuple for the HR volume.
        feature_regime:
            - ``'low_feature'``: Single-scale coarse pore network (``blobiness=1.0, porosity=0.40``).
            - ``'high_feature'``: Multi-scale dual-porosity network combining macro-pores
              and micro-throats (inspired by carbonate/sandstone XCT).
        noise_regime: ``'low_noise'`` vs. ``'high_noise'`` acquisition presets.
        representation:
            - ``'attenuation'``: Realistic two-phase greyscale X-ray attenuation volume where
              pore phase has low intensity (~0.15) and solid matrix has high intensity (~0.85)
              with partial-volume boundary smoothing.
            - ``'local_thickness'``: Normalized 3D local pore-throat thickness field (matching
              Calum Green's original PoreSpy synthetic formulation).
        scale_factor: Spatial downsampling factor between ``hr_vol`` and ``lr_vol``.
        hr_noise_std: Optional override for HR Gaussian acquisition noise.
        lr_noise_std: Optional override for LR Gaussian acquisition noise.
        psf_sigma: Detector point-spread Gaussian sigma applied before voxel binning.
        seed: Deterministic RNG seed.

    Returns:
        ``(lr_vol, hr_vol, pore_mask)`` where ``hr_vol`` and ``pore_mask`` have shape ``(D, H, W)``
        and ``lr_vol`` has isotropic shape ``(D // s, H // s, W // s)``.
    """
    vol_shape = (shape, shape, shape) if isinstance(shape, int) else tuple(int(x) for x in shape)
    if len(vol_shape) != 3:
        raise ValueError(f"shape must be an int or 3-tuple, got {shape!r}.")

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
    default_lr_sigma = 0.015 if noise_regime == "low_noise" else 0.055
    sigma_hr = default_hr_sigma if hr_noise_std is None else float(hr_noise_std)
    sigma_lr = default_lr_sigma if lr_noise_std is None else float(lr_noise_std)

    if representation == "attenuation":
        # Solid grain matrix = 0.85, pore space = 0.15, with mild partial-volume interface blur
        clean = np.where(pore_mask, 0.15, 0.85).astype(np.float32)
        clean = gaussian_filter(clean, sigma=0.45, mode="reflect")
        if sigma_hr > 0.0:
            clean = clean + rng.normal(0.0, sigma_hr, size=clean.shape).astype(np.float32)
        hr_vol = np.clip(clean, 0.0, 1.0).astype(np.float32, copy=False)
        contrast_shift = 0.0
    elif representation == "local_thickness":
        thk = _porespy_local_thickness(pore_mask, sizes=25)
        hr_clean = normalize_volume(thk, mode="minmax")
        if sigma_hr > 0.0:
            hr_noisy = hr_clean + rng.normal(0.0, sigma_hr, size=hr_clean.shape).astype(np.float32)
            hr_vol = normalize_volume(hr_noisy, mode="minmax")
        else:
            hr_vol = hr_clean
        contrast_shift = 0.05 if noise_regime == "high_noise" else 0.02
    else:
        raise ValueError(
            f"Unsupported representation={representation!r}. Expected 'attenuation' or 'local_thickness'."
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

    Generates (or loads from ``cache_dir``) a 3D HR/LR porous-media volume pair, splits it
    into spatially disjoint 3D slabs along ``split_axis``, trains on 2D slices along
    ``train_axes`` (default ``('yz',)``), and evaluates on parallel (``'yz'``) or
    perpendicular (``'xz'``, ``'xy'``) orthogonal planes.
    """

    def __init__(
        self,
        volume_size: int | tuple[int, int, int] = 128,
        feature_regime: FeatureRegime = "high_feature",
        noise_regime: NoiseRegime = "low_noise",
        representation: SyntheticRepresentation = "attenuation",
        scale_factor: int = 2,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        split_axis: int = 0,
        guard_slices_lr: int = 1,
        train_axes: tuple[str, ...] = ("yz",),
        test_axis: TestAxisMode = "perpendicular",
        predict_axis: Literal["yz", "xz", "xy"] = "xy",
        patch_size_lr: int | None = None,
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

        self.volume_size = volume_size
        self.feature_regime: FeatureRegime = feature_regime
        self.noise_regime: NoiseRegime = noise_regime
        self.representation: SyntheticRepresentation = representation
        self.scale_factor = int(scale_factor)
        self.train_frac = float(train_frac)
        self.val_frac = float(val_frac)
        self.split_axis = int(split_axis)
        self.guard_slices_lr = int(guard_slices_lr)
        self.train_axes = tuple(train_axes)
        self.test_axis: TestAxisMode = test_axis
        self.predict_axis = predict_axis
        self.patch_size_lr = patch_size_lr
        self.augment = bool(augment)
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers and self.num_workers > 0)
        self.seed = int(seed)
        self.cache_dir = cache_dir

        self.train_dataset: PairedSliceDataset | None = None
        self.val_dataset: PairedSliceDataset | None = None
        self.test_dataset: PairedSliceDataset | None = None
        self.predict_dataset: PairedSliceDataset | None = None

        self.splits_3d: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _load_or_generate(self) -> tuple[np.ndarray, np.ndarray]:
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
                    "seed": self.seed,
                },
                sort_keys=True,
            )
            digest = hashlib.sha256(spec.encode("utf-8")).hexdigest()[:12]
            hr_file = cache_root / f"porespy_hr_{digest}.tif"
            lr_file = cache_root / f"porespy_lr_{digest}.tif"
            if hr_file.is_file() and lr_file.is_file():
                return (
                    tifffile.imread(lr_file).astype(np.float32),
                    tifffile.imread(hr_file).astype(np.float32),
                )

            lr_vol, hr_vol, _ = generate_porespy_volume(
                shape=self.volume_size,
                feature_regime=self.feature_regime,
                noise_regime=self.noise_regime,
                representation=self.representation,
                scale_factor=self.scale_factor,
                seed=self.seed,
            )
            tifffile.imwrite(hr_file, hr_vol, photometric="minisblack")
            tifffile.imwrite(lr_file, lr_vol, photometric="minisblack")
            return lr_vol, hr_vol

        lr_vol, hr_vol, _ = generate_porespy_volume(
            shape=self.volume_size,
            feature_regime=self.feature_regime,
            noise_regime=self.noise_regime,
            representation=self.representation,
            scale_factor=self.scale_factor,
            seed=self.seed,
        )
        return lr_vol, hr_vol

    def setup(self, stage: str | None = None) -> None:
        if not self.splits_3d:
            lr_vol, hr_vol = self._load_or_generate()
            raw_splits = split_volume_spatially(
                hr_volume=hr_vol,
                lr_volume=lr_vol,
                scale_factor=self.scale_factor,
                train_frac=self.train_frac,
                val_frac=self.val_frac,
                split_axis=self.split_axis,
                guard_slices_lr=self.guard_slices_lr,
            )
            self.splits_3d = {k: (v[0], v[1]) for k, v in raw_splits.items() if v[0] is not None}

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
