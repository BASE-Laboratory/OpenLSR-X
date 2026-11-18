"""Physics-, topology-, and frequency-informed super-resolution losses for quantitative XCT.

Replaces or complements RGB ImageNet VGG perceptual priors with domain invariants:
1. ``AttenuationConservationLoss``: Coarse-voxel X-ray linear attenuation conservation under
   scale-matched optical PSF + area/volume downsampling (supports 2D, 2.5D through-plane
   Laplacian z-PSF correction, and 3D).
2. ``SoftCLDiceLoss``: Differentiable soft-skeleton topological connectivity loss penalizing
   pinched pore throats and spurious isolated pores (adapted from Shit et al., CVPR 2021),
   with bimodal quantile thresholding and single-phase homogeneity gating.
3. ``SobelGradientLoss``: 2D/3D interfacial edge sharpness preservation across phase boundaries.
4. ``FocalFrequencyLoss``: Adaptive spectral loss weighting poorly recovered high wavenumbers
   in 2D/3D Fourier space (adapted from Jiang et al., ICCV 2021).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords**2) / (2.0 * (sigma**2)))
    return kernel / kernel.sum()


def _safe_pad_1d(x: Tensor, pad_tuple: tuple[int, ...], dim_size: int, r: int) -> Tensor:
    mode = "reflect" if (dim_size > r and dim_size > 1) else "replicate"
    return F.pad(x, pad_tuple, mode=mode)


def _apply_separable_gaussian(x: Tensor, sigma: float) -> Tensor:
    """Apply reflect/replicate-padded separable Gaussian blur to a 4D (2D) or 5D (3D) tensor."""
    if sigma <= 0.0:
        return x
    k1d = _gaussian_kernel_1d(sigma, device=x.device, dtype=x.dtype)
    r = k1d.numel() // 2
    c = x.shape[1]
    if x.ndim == 4:
        kh = k1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)
        kw = k1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
        out = _safe_pad_1d(x, (0, 0, r, r), int(x.shape[2]), r)
        out = F.conv2d(out, kh, groups=c)
        out = _safe_pad_1d(out, (r, r, 0, 0), int(out.shape[3]), r)
        return F.conv2d(out, kw, groups=c)
    if x.ndim == 5:
        kd = k1d.view(1, 1, -1, 1, 1).expand(c, 1, -1, 1, 1)
        kh = k1d.view(1, 1, 1, -1, 1).expand(c, 1, 1, -1, 1)
        kw = k1d.view(1, 1, 1, 1, -1).expand(c, 1, 1, 1, -1)
        out = _safe_pad_1d(x, (0, 0, 0, 0, r, r), int(x.shape[2]), r)
        out = F.conv3d(out, kd, groups=c)
        out = _safe_pad_1d(out, (0, 0, r, r, 0, 0), int(out.shape[3]), r)
        out = F.conv3d(out, kh, groups=c)
        out = _safe_pad_1d(out, (r, r, 0, 0, 0, 0), int(out.shape[4]), r)
        return F.conv3d(out, kw, groups=c)
    raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}.")


class AttenuationConservationLoss(nn.Module):
    """Enforces linear-attenuation (``mu``) downsampling consistency between HR predictions and LR inputs.

    In reconstructed absorption micro-CT, coarse voxel linear attenuation approximates the spatial
    average of sub-voxel linear attenuation convolved with the system point-spread function (assuming
    joint intensity calibration between LR and HR). When ``low_res`` provides ``2.5D`` multi-slice
    context (``C >= 3`` sampled at unit native-LR spacing ``+-Delta z_LR``) and ``psf_sigma > 0``,
    incorporates the leading-order through-plane ``z``-PSF curvature term
    ``0.5 * sigma_{z,LR}^2 * d^2 mu / dz_{LR}^2`` so horizontal phase boundaries are not biased
    toward out-of-plane partial-volume blur.
    """

    def __init__(
        self,
        psf_sigma: float = 0.0,
        reduction_norm: Literal["l1", "mse"] = "l1",
    ) -> None:
        super().__init__()
        self.psf_sigma = float(psf_sigma)
        self.reduction_norm: Literal["l1", "mse"] = reduction_norm

    def forward(self, pred_hr: Tensor, low_res: Tensor) -> Tensor:
        pred_f32 = pred_hr.to(torch.float32)
        lr_full = low_res.to(torch.float32)

        z_correction: Tensor | None = None
        if lr_full.shape[1] != pred_f32.shape[1]:
            mid = lr_full.shape[1] // 2
            lr_f32 = lr_full[:, mid : mid + pred_f32.shape[1]]
            if pred_f32.ndim == 4 and lr_full.shape[1] >= 3 and self.psf_sigma > 0.0:
                scale_est = max(1.0, float(pred_f32.shape[-1]) / max(float(lr_full.shape[-1]), 1.0))
                sigma_lr_z = self.psf_sigma / scale_est
                lap_z = (
                    lr_full[:, mid - 1 : mid]
                    - 2.0 * lr_full[:, mid : mid + 1]
                    + lr_full[:, mid + 1 : mid + 2]
                )
                z_correction = 0.5 * (sigma_lr_z**2) * lap_z
        else:
            lr_f32 = lr_full

        blurred = _apply_separable_gaussian(pred_f32, self.psf_sigma)
        target_size = lr_f32.shape[2:]
        downsampled = F.interpolate(blurred, size=target_size, mode="area")
        if z_correction is not None:
            downsampled = downsampled + z_correction

        if self.reduction_norm == "mse":
            return F.mse_loss(downsampled, lr_f32)
        return F.l1_loss(downsampled, lr_f32)


def _soft_erode(img: Tensor) -> Tensor:
    if img.ndim == 4:
        p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        return torch.minimum(p1, p2)
    if img.ndim == 5:
        p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
        p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        return torch.minimum(torch.minimum(p1, p2), p3)
    raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(img.shape)}.")


def _soft_dilate(img: Tensor) -> Tensor:
    if img.ndim == 4:
        return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)
    if img.ndim == 5:
        return F.max_pool3d(img, kernel_size=3, stride=1, padding=1)
    raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(img.shape)}.")


def _soft_open(img: Tensor) -> Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: Tensor, num_iter: int = 5) -> Tensor:
    """Differentiable morphological soft-skeletonization for 2D or 3D probability maps."""
    skel = F.relu(img - _soft_open(img))
    work = img
    for _ in range(max(0, int(num_iter) - 1)):
        work = _soft_erode(work)
        delta = F.relu(work - _soft_open(work))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftCLDiceLoss(nn.Module):
    """Differentiable centerline-Dice (soft-clDice) pore-throat connectivity loss for XCT.

    Uses a bimodal quantile midpoint ``0.5 * (Q_20 + Q_80)`` when ``threshold=None`` and gates
    out single-phase homogeneous patches (dynamic range ``< min_phase_contrast``) so pure solid
    grains or pure macro-pores never have acquisition noise amplified into fake skeletons.
    """

    def __init__(
        self,
        num_iter: int = 5,
        temperature: float = 15.0,
        pore_is_low: bool = True,
        threshold: float | None = None,
        min_phase_contrast: float = 0.08,
        smooth: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_iter = int(num_iter)
        self.temperature = float(temperature)
        self.pore_is_low = bool(pore_is_low)
        self.threshold = float(threshold) if threshold is not None else None
        self.min_phase_contrast = float(min_phase_contrast)
        self.smooth = float(smooth)

    def _to_soft_pore_prob(self, x: Tensor, thr: Tensor) -> Tensor:
        diff = (thr - x) if self.pore_is_low else (x - thr)
        return torch.sigmoid(self.temperature * diff)

    def forward(self, pred_hr: Tensor, target_hr: Tensor) -> Tensor:
        pred_f32 = pred_hr.to(torch.float32)
        true_f32 = target_hr.to(torch.float32)

        spatial_dims = tuple(range(2, true_f32.ndim))
        b, c = true_f32.shape[0], true_f32.shape[1]
        flat_true = true_f32.detach().reshape(b, c, -1)
        dyn_range = flat_true.amax(dim=-1) - flat_true.amin(dim=-1)  # (B, C)

        if self.threshold is not None:
            thr = torch.full_like(true_f32.mean(dim=spatial_dims, keepdim=True), self.threshold)
            active_mask = torch.ones((b, c), device=pred_f32.device, dtype=torch.float32)
        else:
            q20 = torch.quantile(flat_true, 0.20, dim=-1)
            q80 = torch.quantile(flat_true, 0.80, dim=-1)
            mid_q = 0.5 * (q20 + q80)
            view_shape = (b, c) + (1,) * len(spatial_dims)
            thr = mid_q.reshape(view_shape)
            active_mask = (dyn_range >= self.min_phase_contrast).to(torch.float32)

        if float(active_mask.sum().item()) == 0.0:
            return pred_f32.sum() * 0.0

        p_pred = self._to_soft_pore_prob(pred_f32, thr)
        p_true = self._to_soft_pore_prob(true_f32, thr)

        skel_pred = soft_skeletonize(p_pred, num_iter=self.num_iter)
        skel_true = soft_skeletonize(p_true, num_iter=self.num_iter)

        t_prec = ((skel_pred * p_true).sum(dim=spatial_dims) + self.smooth) / (
            skel_pred.sum(dim=spatial_dims) + self.smooth
        )
        t_sens = ((skel_true * p_pred).sum(dim=spatial_dims) + self.smooth) / (
            skel_true.sum(dim=spatial_dims) + self.smooth
        )
        cl_dice = (2.0 * t_prec * t_sens) / (t_prec + t_sens + self.smooth)
        per_sample = (1.0 - cl_dice) * active_mask
        return per_sample.sum() / active_mask.sum().clamp_min(1.0)


class SobelGradientLoss(nn.Module):
    """Differentiable 2D/3D spatial gradient L1 loss preserving sharp phase boundaries."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _spatial_gradients(x: Tensor) -> list[Tensor]:
        grads: list[Tensor] = []
        for dim in range(2, x.ndim):
            if x.shape[dim] < 2:
                continue
            sl_hi = [slice(None)] * x.ndim
            sl_lo = [slice(None)] * x.ndim
            sl_hi[dim] = slice(1, None)
            sl_lo[dim] = slice(0, -1)
            grads.append(x[tuple(sl_hi)] - x[tuple(sl_lo)])
        return grads

    def forward(self, pred_hr: Tensor, target_hr: Tensor) -> Tensor:
        pred_grads = self._spatial_gradients(pred_hr.to(torch.float32))
        true_grads = self._spatial_gradients(target_hr.to(torch.float32))
        if not pred_grads:
            return pred_hr.to(torch.float32).sum() * 0.0
        losses = [F.l1_loss(pg, tg) for pg, tg in zip(pred_grads, true_grads, strict=True)]
        return torch.stack(losses).mean()


class FocalFrequencyLoss(nn.Module):
    """Adaptive 2D/3D Fourier-domain loss focusing optimization on hard high-frequency modes."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, pred_hr: Tensor, target_hr: Tensor) -> Tensor:
        pred_f32 = pred_hr.to(torch.float32)
        true_f32 = target_hr.to(torch.float32)
        spatial_dims = tuple(range(2, pred_f32.ndim))

        fft_pred = torch.fft.rfftn(pred_f32, dim=spatial_dims, norm="ortho")
        fft_true = torch.fft.rfftn(true_f32, dim=spatial_dims, norm="ortho")

        diff = fft_pred - fft_true
        dist_sq = diff.real.square() + diff.imag.square()
        raw_w = dist_sq.detach().clamp_min(1e-12).pow(0.5 * self.alpha)
        max_w = raw_w.amax(dim=spatial_dims, keepdim=True).clamp_min(1e-12)
        weight = raw_w / max_w
        return (weight * dist_sq).mean()
