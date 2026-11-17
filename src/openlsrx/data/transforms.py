"""Vectorized XCT normalization, alignment, realistic degradation, and leakage-free 3D splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, zoom
from torch import Tensor
from torch.utils.data import Dataset

NormMode = Literal["minmax", "percentile", "fixed"]


@dataclass(frozen=True)
class VolumeAlignment:
    """Configurable spatial ROI crop and in-plane rotation for raw 3D XCT volumes."""

    z_range: tuple[int, int] | None = None
    y_range: tuple[int, int] | None = None
    x_range: tuple[int, int] | None = None
    rotation_deg: float = 0.0


def normalize_volume(
    volume: np.ndarray,
    mode: NormMode = "minmax",
    vmin: float | None = None,
    vmax: float | None = None,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Vectorized normalization of a 2D/3D XCT array into ``[0, 1]`` float32."""
    arr = np.asarray(volume, dtype=np.float32)
    if mode == "minmax":
        lo = float(arr.min())
        hi = float(arr.max())
    elif mode == "percentile":
        lo, hi = (float(v) for v in np.percentile(arr, [lower_pct, upper_pct]))
    elif mode == "fixed":
        if vmin is None or vmax is None:
            raise ValueError("vmin and vmax must be provided when mode='fixed'.")
        lo, hi = float(vmin), float(vmax)
    else:
        raise ValueError(f"Unsupported normalization mode={mode!r}.")

    denom = hi - lo
    if abs(denom) < eps:
        return np.zeros_like(arr, dtype=np.float32)
    normed = (arr - lo) / denom
    return np.clip(normed, 0.0, 1.0).astype(np.float32, copy=False)


def rotate_slice(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2D slice around its center using reflection border padding."""
    if abs(angle_deg) < 1e-6:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_alignment(volume: np.ndarray, alignment: VolumeAlignment | None) -> np.ndarray:
    """Apply optional 3D bounding-box crop and slice-wise rotation to a 3D volume."""
    arr = np.asarray(volume, dtype=np.float32)
    if alignment is None:
        return arr

    if abs(alignment.rotation_deg) >= 1e-6:
        rotated = np.empty_like(arr)
        for i in range(arr.shape[0]):
            rotated[i] = rotate_slice(arr[i], alignment.rotation_deg)
        arr = rotated

    z_sl = slice(*alignment.z_range) if alignment.z_range is not None else slice(None)
    y_sl = slice(*alignment.y_range) if alignment.y_range is not None else slice(None)
    x_sl = slice(*alignment.x_range) if alignment.x_range is not None else slice(None)
    return np.ascontiguousarray(arr[z_sl, y_sl, x_sl], dtype=np.float32)


def apply_contrast_shift(volume: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """Vectorized interior contrast shift preserving zero background."""
    if abs(shift) < 1e-8:
        return np.asarray(volume, dtype=np.float32)
    out = np.array(volume, dtype=np.float32, copy=True)
    mask = (out > 0.0) & (out < (1.0 - shift - 0.05))
    out[mask] += float(shift)
    return np.clip(out, 0.0, 1.0)


def degrade_xct_volume(
    hr_volume: np.ndarray,
    scale_factor: int = 2,
    psf_sigma: float = 0.8,
    noise_std: float = 0.02,
    poisson_peak: float | None = None,
    contrast_shift: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate realistic isotropic 3D low-resolution XCT acquisition from an HR volume.

    Models detector optical/scintillator blur (Gaussian PSF), optional photon shot noise
    (Poisson), electronic readout noise (Gaussian), and ``scale_factor`` voxel binning.
    """
    hr = np.asarray(hr_volume, dtype=np.float32)
    if hr.ndim != 3:
        raise ValueError(f"Expected 3D HR volume (D, H, W), got shape {hr.shape}.")
    for dim in hr.shape:
        if dim % scale_factor != 0:
            raise ValueError(
                f"HR volume shape {hr.shape} must be divisible by scale_factor={scale_factor}."
            )

    rng = np.random.default_rng(seed)
    work = apply_contrast_shift(hr, shift=contrast_shift)
    if psf_sigma > 0.0:
        work = gaussian_filter(work, sigma=float(psf_sigma), mode="reflect")

    # Voxel-binning / Lanczos-equivalent antialiasing downsample to (D/s, H/s, W/s)
    s = scale_factor
    d, h, w = work.shape
    binned = work.reshape(d // s, s, h // s, s, w // s, s).mean(axis=(1, 3, 5))

    if poisson_peak is not None and poisson_peak > 0.0:
        counts = rng.poisson(np.clip(binned, 0.0, 1.0) * float(poisson_peak))
        binned = counts.astype(np.float32) / float(poisson_peak)

    if noise_std > 0.0:
        binned = binned + rng.normal(0.0, float(noise_std), size=binned.shape).astype(np.float32)

    return np.clip(binned, 0.0, 1.0).astype(np.float32, copy=False)


def downsample_orthogonal_slabs(
    hr_volume: np.ndarray,
    scale_factor: int = 2,
    psf_sigma: float = 0.6,
    noise_std: float = 0.02,
    variable_slice_noise: bool = False,
    contrast_shift: float = 0.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create paired orthogonal LR slice stacks ``yz``, ``xz``, and ``xy`` from a 3D HR cube.

    For each axis:
    - ``'yz'`` has shape ``(D, H // s, W // s)`` paired with ``hr_volume`` ``(D, H, W)``
    - ``'xz'`` has shape ``(H, D // s, W // s)`` paired with ``np.moveaxis(hr_volume, 1, 0)``
    - ``'xy'`` has shape ``(W, D // s, H // s)`` paired with ``np.moveaxis(hr_volume, 2, 0)``
    """
    hr = np.asarray(hr_volume, dtype=np.float32)
    if hr.ndim != 3:
        raise ValueError(f"Expected 3D HR volume, got shape {hr.shape}.")

    rng = np.random.default_rng(seed)
    work = apply_contrast_shift(hr, shift=contrast_shift)
    if psf_sigma > 0.0:
        work = gaussian_filter(work, sigma=float(psf_sigma), mode="reflect")

    slabs: dict[str, np.ndarray] = {}
    axis_map = {"yz": 0, "xz": 1, "xy": 2}
    for name, dim in axis_map.items():
        ordered = np.moveaxis(work, dim, 0)  # (N, H_in, W_in)
        n_slices, h_in, w_in = ordered.shape
        h_out, w_out = h_in // scale_factor, w_in // scale_factor
        down = np.empty((n_slices, h_out, w_out), dtype=np.float32)
        for i in range(n_slices):
            sl = cv2.resize(
                ordered[i],
                dsize=(w_out, h_out),
                interpolation=cv2.INTER_LANCZOS4,
            )
            sigma = (
                float(noise_std) * float(np.sqrt(rng.uniform(0.2, 1.0)))
                if variable_slice_noise
                else float(noise_std)
            )
            if sigma > 0.0:
                sl = sl + rng.normal(0.0, sigma, size=sl.shape).astype(np.float32)
            down[i] = sl
        slabs[name] = np.clip(down, 0.0, 1.0)
    return slabs


def split_volume_spatially(
    hr_volume: np.ndarray,
    lr_volume: np.ndarray | None = None,
    scale_factor: int = 2,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    split_axis: int = 0,
    guard_slices_lr: int = 0,
) -> dict[str, tuple[np.ndarray | None, np.ndarray]]:
    """Partition a 3D HR (and optional isotropic 3D LR) volume into disjoint spatial sub-volumes.

    Splitting along ``split_axis`` into contiguous non-overlapping 3D slabs guarantees
    that orthogonal ``xy``, ``xz``, and ``yz`` slices extracted from the ``test`` sub-volume
    never share voxels with ``train`` or ``val`` slices.
    """
    hr = np.asarray(hr_volume, dtype=np.float32)
    if hr.ndim != 3:
        raise ValueError(f"Expected 3D HR volume, got shape {hr.shape}.")
    total_hr = hr.shape[split_axis]
    total_lr = total_hr // scale_factor
    if total_lr < 3:
        raise ValueError(
            f"Volume dimension {total_hr} along axis {split_axis} is too small for 3-way split "
            f"at scale_factor={scale_factor}."
        )

    usable_lr = total_lr - 2 * guard_slices_lr
    if usable_lr < 3:
        raise ValueError("guard_slices_lr leaves fewer than 3 LR slices across splits.")

    n_train_lr = max(1, int(round(usable_lr * train_frac)))
    n_val_lr = max(1, int(round(usable_lr * val_frac)))
    if n_train_lr + n_val_lr >= usable_lr:
        n_train_lr = max(1, usable_lr - 2)
        n_val_lr = 1
    n_test_lr = usable_lr - n_train_lr - n_val_lr

    t0_lr, t1_lr = 0, n_train_lr
    v0_lr = t1_lr + guard_slices_lr
    v1_lr = v0_lr + n_val_lr
    s0_lr = v1_lr + guard_slices_lr
    s1_lr = s0_lr + n_test_lr

    def _slice_axis(arr: np.ndarray, start: int, stop: int) -> np.ndarray:
        idx = [slice(None), slice(None), slice(None)]
        idx[split_axis] = slice(start, stop)
        return np.ascontiguousarray(arr[tuple(idx)], dtype=np.float32)

    splits: dict[str, tuple[np.ndarray | None, np.ndarray]] = {}
    for name, (a_lr, b_lr) in (
        ("train", (t0_lr, t1_lr)),
        ("val", (v0_lr, v1_lr)),
        ("test", (s0_lr, s1_lr)),
    ):
        hr_sub = _slice_axis(hr, a_lr * scale_factor, b_lr * scale_factor)
        lr_sub = _slice_axis(lr_volume, a_lr, b_lr) if lr_volume is not None else None
        splits[name] = (lr_sub, hr_sub)
    return splits


def extract_paired_slices_from_subvolume(
    lr_sub: np.ndarray,
    hr_sub: np.ndarray,
    scale_factor: int = 2,
    axes: tuple[str, ...] = ("yz",),
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned ``(low_slices, high_slices)`` from paired isotropic 3D ``(lr_sub, hr_sub)``.

    For each requested axis in ``('yz', 'xz', 'xy')``, ``lr_sub`` is cubic-interpolated along
    that slicing axis by ``scale_factor`` so its slice count matches ``hr_sub`` along that axis.
    """
    axis_map = {"yz": 0, "xz": 1, "xy": 2}
    low_list: list[np.ndarray] = []
    high_list: list[np.ndarray] = []

    for ax in axes:
        dim = axis_map[ax]
        lr_ord = np.moveaxis(lr_sub, dim, 0)
        hr_ord = np.moveaxis(hr_sub, dim, 0)
        if lr_ord.shape[0] != hr_ord.shape[0]:
            lr_ord = zoom(lr_ord, zoom=(scale_factor, 1, 1), order=3, mode="reflect").astype(
                np.float32, copy=False
            )
            lr_ord = np.clip(lr_ord, 0.0, 1.0)
        low_list.append(lr_ord)
        high_list.append(hr_ord)

    return np.concatenate(low_list, axis=0), np.concatenate(high_list, axis=0)


class PairedSliceDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch Dataset of paired ``(low_res, high_res)`` 2D XCT slices.

    Supports optional aligned random spatial patch cropping and dihedral ``D_4``
    (90-degree rotations + flips) equivariant augmentations during training.
    """

    def __init__(
        self,
        low_res: Tensor | np.ndarray,
        high_res: Tensor | np.ndarray | None = None,
        patch_size_lr: int | None = None,
        scale_factor: int = 2,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.low_res = torch.as_tensor(low_res, dtype=torch.float32)
        self.high_res = (
            torch.as_tensor(high_res, dtype=torch.float32) if high_res is not None else None
        )
        if self.low_res.ndim == 4 and self.low_res.shape[1] == 1:
            self.low_res = self.low_res.squeeze(1)
        if self.high_res is not None and self.high_res.ndim == 4 and self.high_res.shape[1] == 1:
            self.high_res = self.high_res.squeeze(1)

        if self.high_res is not None and self.low_res.shape[0] != self.high_res.shape[0]:
            raise ValueError(
                f"Mismatched slice counts: low_res={self.low_res.shape[0]} vs high_res={self.high_res.shape[0]}."
            )

        self.patch_size_lr = int(patch_size_lr) if patch_size_lr is not None else None
        self.scale_factor = int(scale_factor)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return int(self.low_res.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        lr = self.low_res[idx]
        hr = self.high_res[idx] if self.high_res is not None else lr

        if self.patch_size_lr is not None and self.high_res is not None:
            p_lr = self.patch_size_lr
            h_lr, w_lr = int(lr.shape[-2]), int(lr.shape[-1])
            if h_lr >= p_lr and w_lr >= p_lr:
                top = int(torch.randint(0, h_lr - p_lr + 1, (1,)).item())
                left = int(torch.randint(0, w_lr - p_lr + 1, (1,)).item())
                lr = lr[top : top + p_lr, left : left + p_lr]
                s = self.scale_factor
                hr = hr[top * s : (top + p_lr) * s, left * s : (left + p_lr) * s]

        if self.augment and self.high_res is not None:
            k_rot = int(torch.randint(0, 4, (1,)).item())
            if k_rot > 0:
                lr = torch.rot90(lr, k=k_rot, dims=(-2, -1))
                hr = torch.rot90(hr, k=k_rot, dims=(-2, -1))
            if bool(torch.randint(0, 2, (1,)).item()):
                lr = torch.flip(lr, dims=(-1,))
                hr = torch.flip(hr, dims=(-1,))

        return lr.contiguous(), hr.contiguous()


# Alias preserving clean familiarity with original OpenLSR-X naming
DatasetObject = PairedSliceDataset
