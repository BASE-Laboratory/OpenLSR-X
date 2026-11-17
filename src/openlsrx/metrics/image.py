"""Standard 2D and 3D image-fidelity metrics for XCT super-resolution."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def _to_4d_tensor(arr: Tensor | np.ndarray) -> Tensor:
    """Normalize a 2D ``(H, W)``, 3D ``(N, H, W)``, or 4D ``(B, C, H, W)`` array to float32 ``(N, 1, H, W)``."""
    t = torch.as_tensor(arr, dtype=torch.float32)
    if t.ndim == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.ndim == 3:
        return t.unsqueeze(1)
    if t.ndim == 4:
        return t
    raise ValueError(f"Expected 2D, 3D, or 4D array/tensor, got shape {tuple(t.shape)}.")


def compute_mae(pred: Tensor | np.ndarray, target: Tensor | np.ndarray) -> float:
    """Mean absolute error between prediction and ground truth."""
    p = torch.as_tensor(pred, dtype=torch.float32)
    t = torch.as_tensor(target, dtype=torch.float32, device=p.device)
    return float(torch.mean(torch.abs(p - t)).item())


def compute_rmse(pred: Tensor | np.ndarray, target: Tensor | np.ndarray) -> float:
    """Root mean squared error between prediction and ground truth."""
    p = torch.as_tensor(pred, dtype=torch.float32)
    t = torch.as_tensor(target, dtype=torch.float32, device=p.device)
    return float(torch.sqrt(torch.mean((p - t).square())).item())


def compute_psnr(
    pred: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Peak Signal-to-Noise Ratio (dB) over 2D slices or full 3D volumes."""
    p = torch.as_tensor(pred, dtype=torch.float32)
    t = torch.as_tensor(target, dtype=torch.float32, device=p.device)
    mse = float(torch.mean((p - t).square()).item())
    if mse == 0.0:
        return float("inf")
    return float(peak_signal_noise_ratio(p, t, data_range=data_range).item())


def compute_ssim(
    pred: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    data_range: float = 1.0,
    chunk_size: int = 32,
) -> float:
    """Structural Similarity Index (SSIM) averaged over slices or batches."""
    p = _to_4d_tensor(pred)
    t = _to_4d_tensor(target).to(device=p.device)
    if torch.equal(p, t):
        return 1.0
    min_side = min(p.shape[-2], p.shape[-1])
    kernel_size = min(11, min_side if min_side % 2 == 1 else max(1, min_side - 1))
    if kernel_size < 3:
        return 1.0 if torch.allclose(p, t) else 0.0

    scores: list[float] = []
    weights: list[int] = []
    for start in range(0, p.shape[0], chunk_size):
        p_chunk = p[start : start + chunk_size]
        t_chunk = t[start : start + chunk_size]
        val = structural_similarity_index_measure(
            p_chunk,
            t_chunk,
            data_range=data_range,
            kernel_size=kernel_size,
        )
        val_t = val[0] if isinstance(val, tuple) else val
        scores.append(float(val_t.item()))
        weights.append(int(p_chunk.shape[0]))

    total = sum(weights)
    return sum(s * w for s, w in zip(scores, weights, strict=True)) / max(total, 1)


def compute_ssim_3d_isotropic(
    pred_vol: Tensor | np.ndarray,
    target_vol: Tensor | np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Tri-planar SSIM averaged across ``yz``, ``xz``, and ``xy`` slices of a 3D volume."""
    p = torch.as_tensor(pred_vol, dtype=torch.float32)
    t = torch.as_tensor(target_vol, dtype=torch.float32, device=p.device)
    if p.ndim != 3 or t.ndim != 3:
        raise ValueError(
            f"Expected 3D volumes (D, H, W), got {tuple(p.shape)} and {tuple(t.shape)}."
        )
    ssim_yz = compute_ssim(p, t, data_range=data_range)
    ssim_xz = compute_ssim(p.permute(1, 0, 2), t.permute(1, 0, 2), data_range=data_range)
    ssim_xy = compute_ssim(p.permute(2, 0, 1), t.permute(2, 0, 1), data_range=data_range)
    return (ssim_yz + ssim_xz + ssim_xy) / 3.0


def compute_ms_ssim(
    pred: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    data_range: float = 1.0,
) -> float | None:
    """Multi-Scale SSIM when spatial dimensions are large enough (>= 161 px), else ``None``."""
    p = _to_4d_tensor(pred)
    t = _to_4d_tensor(target).to(device=p.device)
    if min(p.shape[-2], p.shape[-1]) < 161:
        return None
    val = multiscale_structural_similarity_index_measure(p, t, data_range=data_range)
    score = float(val.item())
    return score if not math.isnan(score) else None


def evaluate_image_metrics(
    pred: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    data_range: float = 1.0,
) -> dict[str, float]:
    """Compute a complete dictionary of image-fidelity metrics."""
    p = torch.as_tensor(pred, dtype=torch.float32)
    t = torch.as_tensor(target, dtype=torch.float32, device=p.device)
    results: dict[str, float] = {
        "psnr": compute_psnr(p, t, data_range=data_range),
        "ssim": compute_ssim(p, t, data_range=data_range),
        "mae": compute_mae(p, t),
        "rmse": compute_rmse(p, t),
    }
    if p.ndim == 3:
        results["ssim_3d_triplanar"] = compute_ssim_3d_isotropic(p, t, data_range=data_range)
    ms_ssim = compute_ms_ssim(p, t, data_range=data_range)
    if ms_ssim is not None:
        results["ms_ssim"] = ms_ssim
    return results
