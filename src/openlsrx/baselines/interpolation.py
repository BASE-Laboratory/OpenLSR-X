"""Classical 2D slice and 3D volumetric interpolation baselines for XCT super-resolution."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from openlsrx.data.transforms import sanitize_xct_volume, zoom_cell_centered

Interp2DMode = Literal["nearest", "bilinear", "bicubic", "lanczos4"]
Interp3DMode = Literal["nearest", "trilinear", "tricubic"]


def upsample_2d(
    low_res: Tensor | np.ndarray,
    scale_factor: float = 2,
    mode: Interp2DMode = "bicubic",
    clamp_01: bool = True,
) -> np.ndarray:
    """Upsample a 2D image ``(H, W)`` or stack of 2D slices ``(N, H, W)`` by ``scale_factor``."""
    arr = sanitize_xct_volume(np.asarray(low_res, dtype=np.float32))
    single = arr.ndim == 2
    if single:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D (H, W) or 3D (N, H, W) slice stack, got shape {arr.shape}.")

    n, h, w = arr.shape
    out_h = max(1, int(round(h * float(scale_factor))))
    out_w = max(1, int(round(w * float(scale_factor))))

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
    scale_factors: float | tuple[float, float, float] = 2,
    mode: Interp3DMode = "trilinear",
    clamp_01: bool = True,
) -> np.ndarray:
    """Upsample a 3D volume ``(D, H, W)`` isotropically or anisotropically with cell-centered alignment."""
    arr = sanitize_xct_volume(np.asarray(low_volume, dtype=np.float32))
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume (D, H, W), got shape {arr.shape}.")

    factors = (
        (float(scale_factors), float(scale_factors), float(scale_factors))
        if isinstance(scale_factors, (int, float))
        else tuple(float(s) for s in scale_factors)
    )
    out_shape = (
        max(1, int(round(arr.shape[0] * factors[0]))),
        max(1, int(round(arr.shape[1] * factors[1]))),
        max(1, int(round(arr.shape[2] * factors[2]))),
    )

    if mode == "tricubic":
        eff_zoom = (
            out_shape[0] / float(arr.shape[0]),
            out_shape[1] / float(arr.shape[1]),
            out_shape[2] / float(arr.shape[2]),
        )
        out = zoom_cell_centered(arr, zoom_factors=eff_zoom, order=3)
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
