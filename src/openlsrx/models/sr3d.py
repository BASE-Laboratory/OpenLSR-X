"""Memory-matched native 3D convolutional residual super-resolution baseline (SR3DResNet).

Provides a direct volumetric 3D CNN baseline to benchmark GPU memory footprint, training
throughput, and petrophysical fidelity against 2D + 3-Axis Orthogonal Fourier Fusion.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset

from ..metrics import compute_mae, compute_psnr, compute_rmse, compute_ssim_3d_isotropic


def _safe_conv3d(conv: nn.Conv3d, x: Tensor) -> Tensor:
    """Apply ``conv`` falling back from ``reflect`` to ``replicate`` when ``min(D, H, W) <= pad``."""
    if conv.padding_mode == "reflect" and isinstance(conv.padding, tuple):
        pd, ph, pw = int(conv.padding[0]), int(conv.padding[1]), int(conv.padding[2])
        if (pd > 0 or ph > 0 or pw > 0) and (
            x.shape[2] <= pd or x.shape[3] <= ph or x.shape[4] <= pw
        ):
            x_pad = F.pad(x, (pw, pw, ph, ph, pd, pd), mode="replicate")
            return F.conv3d(
                x_pad,
                conv.weight,
                conv.bias,
                stride=conv.stride,
                padding=0,
                dilation=conv.dilation,
                groups=conv.groups,
            )
    return conv(x)


class _SafeReflectConv3d(nn.Module):
    """Thin-slab-safe 3D convolution wrapper preserving exact ``Conv3d`` parameter layout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_ch, out_ch, kernel_size=kernel_size, padding=padding, padding_mode="reflect"
        )

    def forward(self, x: Tensor) -> Tensor:
        return _safe_conv3d(self.conv, x)


class ResidualBlock3D(nn.Module):
    """3D reflect-padded residual block without batch normalization."""

    def __init__(self, channels: int = 32, res_scale: float = 0.2) -> None:
        super().__init__()
        self.res_scale = float(res_scale)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect")

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(_safe_conv3d(self.conv1, x))
        h = _safe_conv3d(self.conv2, h)
        return x + self.res_scale * h


class Generator3D(nn.Module):
    """3D residual super-resolution network using trilinear resize-Conv3d upsampling."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 24,
        num_blocks: int = 4,
        super_res_factor: int = 2,
    ) -> None:
        super().__init__()
        if super_res_factor < 1:
            raise ValueError(f"Generator3D super_res_factor must be >= 1, got {super_res_factor}.")
        rem = int(super_res_factor)
        up_steps: list[int] = []
        while rem % 2 == 0:
            up_steps.append(2)
            rem //= 2
        while rem % 3 == 0:
            up_steps.append(3)
            rem //= 3
        if rem != 1:
            raise ValueError(
                f"Generator3D super_res_factor must factor into 2s and 3s (1, 2, 3, 4, 6, 8, ...), "
                f"got {super_res_factor}."
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_blocks = int(num_blocks)
        self.super_res_factor = int(super_res_factor)

        self.conv_first = nn.Conv3d(
            self.in_channels, self.out_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock3D(self.out_channels) for _ in range(self.num_blocks)]
        )
        self.trunk_conv = nn.Conv3d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )

        up_layers: list[nn.Module] = []
        for step in up_steps:
            up_layers.extend(
                [
                    nn.Upsample(scale_factor=step, mode="trilinear", align_corners=False),
                    _SafeReflectConv3d(
                        self.out_channels, self.out_channels, kernel_size=3, padding=1
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        self.upsample = nn.Sequential(*up_layers)
        self.conv_last = nn.Conv3d(
            self.out_channels, self.in_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        nn.init.zeros_(self.conv_last.bias)  # type: ignore[arg-type]
        self.conv_last.weight.data.mul_(0.1)

    def forward(self, x: Tensor) -> Tensor:
        base = F.interpolate(
            x,
            scale_factor=self.super_res_factor,
            mode="trilinear",
            align_corners=False,
        )
        feat = _safe_conv3d(self.conv_first, x)
        feat = feat + _safe_conv3d(self.trunk_conv, self.trunk(feat))
        out = base + _safe_conv3d(self.conv_last, self.upsample(feat))
        clamped = torch.clamp(out, 0.0, 1.0)
        return out + (clamped - out).detach() if self.training else clamped


class PairedSubCubeDataset(Dataset[tuple[Tensor, Tensor]]):
    """Random/grid 3D sub-cube sampler with full 3D octahedral ($O_h$) symmetry augmentation."""

    def __init__(
        self,
        lr_vol: np.ndarray,
        hr_vol: np.ndarray,
        scale_factor: int = 2,
        patch_size_lr: int = 16,
        samples_per_epoch: int = 32,
        deterministic: bool = False,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.lr_vol = np.asarray(lr_vol, dtype=np.float32)
        self.hr_vol = np.asarray(hr_vol, dtype=np.float32)
        self.scale_factor = int(scale_factor)
        self.patch_size_lr = tuple(min(int(patch_size_lr), s) for s in self.lr_vol.shape)
        self.samples_per_epoch = int(samples_per_epoch)
        self.deterministic = bool(deterministic)
        self.augment = bool(augment) and not self.deterministic
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        pd, ph, pw = self.patch_size_lr
        d, h, w = self.lr_vol.shape
        if self.deterministic:
            rng = np.random.default_rng(idx + 1000)
            z0 = int(rng.integers(0, max(1, d - pd + 1)))
            y0 = int(rng.integers(0, max(1, h - ph + 1)))
            x0 = int(rng.integers(0, max(1, w - pw + 1)))
        else:
            z0 = int(self.rng.integers(0, max(1, d - pd + 1)))
            y0 = int(self.rng.integers(0, max(1, h - ph + 1)))
            x0 = int(self.rng.integers(0, max(1, w - pw + 1)))

        s = self.scale_factor
        lr_patch = self.lr_vol[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
        hr_patch = self.hr_vol[
            z0 * s : (z0 + pd) * s,
            y0 * s : (y0 + ph) * s,
            x0 * s : (x0 + pw) * s,
        ]

        if self.augment:
            # 1) Independent 3D reflections along (Z, Y, X)
            for ax in range(3):
                if bool(self.rng.integers(0, 2)):
                    lr_patch = np.flip(lr_patch, axis=ax)
                    hr_patch = np.flip(hr_patch, axis=ax)
            # 2) Shape-preserving 3D axis permutations (full 48-element O_h group on cubic patches)
            if pd == ph == pw:
                perm = tuple(int(a) for a in self.rng.permutation(3))
                if perm != (0, 1, 2):
                    lr_patch = np.transpose(lr_patch, axes=perm)
                    hr_patch = np.transpose(hr_patch, axes=perm)
            elif ph == pw and bool(self.rng.integers(0, 2)):
                lr_patch = np.swapaxes(lr_patch, 1, 2)
                hr_patch = np.swapaxes(hr_patch, 1, 2)

        return (
            torch.from_numpy(np.ascontiguousarray(lr_patch, dtype=np.float32)).unsqueeze(0),
            torch.from_numpy(np.ascontiguousarray(hr_patch, dtype=np.float32)).unsqueeze(0),
        )


def build_3d_subcube_dataloaders(
    lr_train: np.ndarray,
    hr_train: np.ndarray,
    lr_val: np.ndarray,
    hr_val: np.ndarray,
    scale_factor: int = 2,
    patch_size_lr: int = 16,
    train_samples: int = 32,
    val_samples: int = 8,
    batch_size: int = 2,
    augment: bool = True,
    seed: int = 42,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Build train and validation 3D sub-cube DataLoaders from disjoint 3D slabs."""
    tr_ds = PairedSubCubeDataset(
        lr_train,
        hr_train,
        scale_factor=scale_factor,
        patch_size_lr=patch_size_lr,
        samples_per_epoch=train_samples,
        deterministic=False,
        augment=augment,
        seed=seed,
    )
    va_ds = PairedSubCubeDataset(
        lr_val,
        hr_val,
        scale_factor=scale_factor,
        patch_size_lr=patch_size_lr,
        samples_per_epoch=val_samples,
        deterministic=True,
        augment=False,
        seed=seed,
    )
    return (
        DataLoader(tr_ds, batch_size=batch_size, shuffle=False),
        DataLoader(va_ds, batch_size=batch_size, shuffle=False),
    )


class SR3DResNet(pl.LightningModule):
    """PyTorch Lightning 3D CNN super-resolution baseline for XCT volumes."""

    def __init__(
        self,
        learning_rate: float = 2e-4,
        super_res_factor: int = 2,
        out_channels: int = 24,
        num_blocks: int = 4,
        loss_type: Literal["l1", "mse"] = "l1",
        pore_is_low: bool = True,
        conservation_loss_weight: float = 0.0,
        topology_loss_weight: float = 0.0,
        gradient_loss_weight: float = 0.0,
        ffl_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = float(learning_rate)
        self.super_res_factor = int(super_res_factor)
        self.pore_is_low = bool(pore_is_low)
        self.conservation_loss_weight = float(conservation_loss_weight)
        self.topology_loss_weight = float(topology_loss_weight)
        self.gradient_loss_weight = float(gradient_loss_weight)
        self.ffl_loss_weight = float(ffl_loss_weight)
        self.gen = Generator3D(
            in_channels=1,
            out_channels=out_channels,
            num_blocks=num_blocks,
            super_res_factor=super_res_factor,
        )
        self.criterion: nn.Module = nn.L1Loss() if loss_type == "l1" else nn.MSELoss()
        from ..losses import (
            AttenuationConservationLoss,
            FocalFrequencyLoss,
            SobelGradientLoss,
            SoftCLDiceLoss,
        )

        self.conservation_criterion = AttenuationConservationLoss()
        self.topology_criterion = SoftCLDiceLoss(pore_is_low=self.pore_is_low)
        self.gradient_criterion = SobelGradientLoss()
        self.ffl_criterion = FocalFrequencyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.gen(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        lr_cube, hr_cube = batch
        pred = self.gen(lr_cube)
        loss = self.criterion(pred, hr_cube)
        if self.conservation_loss_weight > 0.0:
            loss = loss + self.conservation_loss_weight * self.conservation_criterion(pred, lr_cube)
        if self.topology_loss_weight > 0.0:
            loss = loss + self.topology_loss_weight * self.topology_criterion(pred, hr_cube)
        if self.gradient_loss_weight > 0.0:
            loss = loss + self.gradient_loss_weight * self.gradient_criterion(pred, hr_cube)
        if self.ffl_loss_weight > 0.0:
            loss = loss + self.ffl_loss_weight * self.ffl_criterion(pred, hr_cube)
        self.log(
            "train/loss_3d",
            loss.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=int(lr_cube.shape[0]),
        )
        return loss

    @torch.no_grad()
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        lr_cube, hr_cube = batch
        pred = self.gen(lr_cube).to(torch.float32)
        target = hr_cube.to(torch.float32)
        bs = int(lr_cube.shape[0])
        ssim_mean = float(
            np.mean([compute_ssim_3d_isotropic(pred[i, 0], target[i, 0]) for i in range(bs)])
        )
        self.log(
            "val/psnr", compute_psnr(pred, target), prog_bar=True, on_epoch=True, batch_size=bs
        )
        self.log(
            "val/ssim_3d",
            ssim_mean,
            on_epoch=True,
            batch_size=bs,
        )
        self.log("val/mae", compute_mae(pred, target), on_epoch=True, batch_size=bs)
        self.log("val/rmse", compute_rmse(pred, target), on_epoch=True, batch_size=bs)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.learning_rate)


@torch.no_grad()
def super_resolve_volume_3d(
    generator_3d: nn.Module,
    low_volume: np.ndarray,
    scale_factor: int = 2,
    patch_size_lr: int = 24,
    overlap_lr: int = 4,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Sliding-window 3D inference with raised-cosine feathering to eliminate tile seams."""
    vol = np.asarray(low_volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D LR volume, got shape {vol.shape}.")

    if device is None:
        try:
            dev = next(generator_3d.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    was_training = generator_3d.training
    generator_3d.eval()

    d, h, w = vol.shape
    pd, ph, pw = min(patch_size_lr, d), min(patch_size_lr, h), min(patch_size_lr, w)
    step_d = max(1, pd - overlap_lr)
    step_h = max(1, ph - overlap_lr)
    step_w = max(1, pw - overlap_lr)

    s = int(scale_factor)
    out_sum = np.zeros((d * s, h * s, w * s), dtype=np.float32)
    weight_sum = np.zeros((d * s, h * s, w * s), dtype=np.float32)

    def _window_1d(n: int) -> np.ndarray:
        if n <= 2:
            return np.ones(n, dtype=np.float32)
        return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))).astype(np.float32) + 1e-3

    win_3d = (
        _window_1d(pd * s)[:, None, None]
        * _window_1d(ph * s)[None, :, None]
        * _window_1d(pw * s)[None, None, :]
    )

    def _starts(total: int, patch: int, step: int) -> list[int]:
        if total <= patch:
            return [0]
        pts = list(range(0, total - patch, step))
        if not pts or pts[-1] != total - patch:
            pts.append(total - patch)
        return pts

    try:
        for z0 in _starts(d, pd, step_d):
            for y0 in _starts(h, ph, step_h):
                for x0 in _starts(w, pw, step_w):
                    patch = vol[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]
                    inp = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(dev)
                    pred = F.pad(inp, (0, 0, 0, 0, 0, 0))  # keep contiguous tensor
                    sr_patch = generator_3d(pred)[0, 0].to(torch.float32).cpu().numpy()
                    out_sum[
                        z0 * s : (z0 + pd) * s,
                        y0 * s : (y0 + ph) * s,
                        x0 * s : (x0 + pw) * s,
                    ] += sr_patch * win_3d
                    weight_sum[
                        z0 * s : (z0 + pd) * s,
                        y0 * s : (y0 + ph) * s,
                        x0 * s : (x0 + pw) * s,
                    ] += win_3d
    finally:
        if was_training:
            generator_3d.train()

    return np.clip(out_sum / np.maximum(weight_sum, 1e-8), 0.0, 1.0)
