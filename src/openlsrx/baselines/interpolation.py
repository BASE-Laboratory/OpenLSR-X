"""Classical 2D slice and 3D volumetric interpolation baselines for XCT super-resolution."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch import Tensor

Interp2DMode = Literal["nearest", "bilinear", "bicubic", "lanczos4"]
Interp3DMode = Literal["nearest", "trilinear", "tricubic"]


def upsample_2d(
    low_res: Tensor | np.ndarray,
    scale_factor: int = 2,
    mode: Interp2DMode = "bicubic",
    clamp_01: bool = True,
) -> np.ndarray:
    """Upsample a 2D image ``(H, W)`` or stack of 2D slices ``(N, H, W)`` by ``scale_factor``."""
    arr = np.asarray(low_res, dtype=np.float32)
    single = arr.ndim == 2
    if single:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D (H, W) or 3D (N, H, W) slice stack, got shape {arr.shape}.")

    n, h, w = arr.shape
    out_h, out_w = h * scale_factor, w * scale_factor

    out: np.ndarray
    if mode == "lanczos4":
        out = np.empty((n, out_h, out_w), dtype=np.float32)
        for i in range(n):
            out[i] = cv2.resize(arr[i], (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    elif mode in ("nearest", "bilinear", "bicubic"):
        t = torch.from_numpy(arr).unsqueeze(1)
        align = None if mode == "nearest" else False
        up = F.interpolate(t, size=(out_h, out_w), mode=mode, align_corners=align)
        out = up.squeeze(1).numpy()
    else:
        raise ValueError(
            f"Unsupported 2D interpolation mode={mode!r}. "
            "Expected 'nearest', 'bilinear', 'bicubic', or 'lanczos4'."
        )

    if clamp_01:
        out = np.clip(out, 0.0, 1.0)
    return out[0] if single else out


def upsample_3d(
    low_volume: Tensor | np.ndarray,
    scale_factors: int | tuple[int, int, int] = 2,
    mode: Interp3DMode = "trilinear",
    clamp_01: bool = True,
) -> np.ndarray:
    """Upsample a 3D volume ``(D, H, W)`` isotropically or anisotropically."""
    arr = np.asarray(low_volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume (D, H, W), got shape {arr.shape}.")

    factors = (
        (scale_factors, scale_factors, scale_factors)
        if isinstance(scale_factors, int)
        else tuple(int(s) for s in scale_factors)
    )
    out_shape = (arr.shape[0] * factors[0], arr.shape[1] * factors[1], arr.shape[2] * factors[2])

    if mode == "tricubic":
        out = zoom(arr, zoom=factors, order=3, mode="reflect").astype(np.float32, copy=False)
    elif mode in ("nearest", "trilinear"):
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        align = None if mode == "nearest" else False
        up = F.interpolate(t, size=out_shape, mode=mode, align_corners=align)
        out = up.squeeze(0).squeeze(0).numpy()
    else:
        raise ValueError(
            f"Unsupported 3D interpolation mode={mode!r}. Expected 'nearest', 'trilinear', or 'tricubic'."
        )

    if clamp_01:
        out = np.clip(out, 0.0, 1.0)
    return out
