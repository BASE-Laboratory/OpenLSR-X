"""PyTorch Lightning SRGAN / SRResNet module for 2D-to-3D XCT super-resolution."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytorch_lightning as pl
import tifffile
import torch
from torch import Tensor, nn, optim

from ..inference.orthogonal import (
    AXIS_TO_DIM,
    FusionMode,
    super_resolve_volume_orthogonal,
)
from ..losses import (
    AttenuationConservationLoss,
    FocalFrequencyLoss,
    SobelGradientLoss,
    SoftCLDiceLoss,
    VGGLoss,
)
from ..metrics.image import compute_mae, compute_ssim
from .blocks import NormType
from .rrdb import RRDBGenerator
from .submodels import Discriminator, Generator, OutputActivation
from .swinir import SwinIRGenerator

BackboneType = Literal["srgan", "rrdb", "swinir"]
GANLossType = Literal["standard", "ragan"]


def _ensure_4d(x: Tensor) -> Tensor:
    """Convert ``(B, H, W)`` or ``(B, C, H, W)`` tensors to 4D ``(B, C, H, W)``."""
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(
        f"Expected 3D (B, H, W) or 4D (B, C, H, W) tensor, got shape {tuple(x.shape)}."
    )


class SRGAN(pl.LightningModule):
    """Learnable Super-Resolution Module (SRGAN / RRDB-RaGAN / SwinIR-2.5D) for XCT data.

    Supports:
    - Classical ``Generator`` (``backbone='srgan'``), normalization-free ``RRDBGenerator``
      (``backbone='rrdb'``), or shifted-window attention ``SwinIRGenerator`` (``backbone='swinir'``)
    - Single-slice ``2D`` (``context_slices=1``) or multi-slice ``2.5D`` (``context_slices=3, 5``)
    - Standard BCE adversarial loss or Relativistic Average GAN (``gan_loss_type='ragan'``)
    - Quantitative domain losses: ``AttenuationConservationLoss``, ``SoftCLDiceLoss``
      (pore-throat connectivity), ``SobelGradientLoss``, and ``FocalFrequencyLoss``
    - Built-in PSNR, SSIM, MAE, and RMSE logging on validation and test splits
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        adaptive_size: int = 8,
        accumulate_n_batches: int = 1,
        super_res_factor: int | float = 2,
        num_blocks: int = 10,
        out_channels: int = 64,
        growth_channels: int = 32,
        context_slices: int = 1,
        backbone: BackboneType = "srgan",
        swin_window_size: int = 8,
        swin_num_heads: int = 4,
        gan_loss_type: GANLossType = "standard",
        norm_type: NormType = "pixel",
        output_activation: OutputActivation = "custom_sigmoid",
        sigmoid_k: float = 1.0,
        pixel_loss_type: Literal["mse", "l1"] = "mse",
        use_adversarial_loss: bool = True,
        adversarial_weight: float = 1e-3,
        pretrain_epochs: int = 0,
        use_vgg_loss: bool = False,
        vgg_loss_weight: float = 0.006,
        vgg_layers: int = 18,
        vgg_pretrained: bool = True,
        pore_is_low: bool = True,
        conservation_loss_weight: float = 0.0,
        conservation_psf_sigma: float = 0.0,
        topology_loss_weight: float = 0.0,
        gradient_loss_weight: float = 0.0,
        ffl_loss_weight: float = 0.0,
        disc_features: Sequence[int] | None = None,
        sample_id: str = "",
        strict_loading: bool = True,
        lr_scheduler_type: Literal["constant", "CAWR"] = "constant",
        lr_scheduler_args: dict[str, Any] | None = None,
        output_dir: str | None = None,
        save_val_images: bool = True,
        max_val_images_saved: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if accumulate_n_batches < 1:
            raise ValueError(f"accumulate_n_batches must be >= 1, got {accumulate_n_batches}.")
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError(
                f"context_slices must be a positive odd integer (1, 3, 5, ...), got {context_slices}."
            )
        if lr_scheduler_type not in ("constant", "CAWR"):
            raise ValueError(
                f"Unsupported lr_scheduler_type={lr_scheduler_type!r}. Expected 'constant' or 'CAWR'."
            )
        if backbone not in ("srgan", "rrdb", "swinir"):
            raise ValueError(
                f"Unsupported backbone={backbone!r}. Expected 'srgan', 'rrdb', or 'swinir'."
            )
        if gan_loss_type not in ("standard", "ragan"):
            raise ValueError(
                f"Unsupported gan_loss_type={gan_loss_type!r}. Expected 'standard' or 'ragan'."
            )

        self.sample_id = str(sample_id)
        self.learning_rate = float(learning_rate)
        self.adaptive_size = int(adaptive_size)
        self.accumulate_n_batches = int(accumulate_n_batches)
        self.super_res_factor: int | float = (
            int(super_res_factor)
            if float(super_res_factor).is_integer()
            else float(super_res_factor)
        )
        self.num_blocks = int(num_blocks)
        self.out_channels = int(out_channels)
        self.growth_channels = int(growth_channels)
        self.context_slices = int(context_slices)
        self.backbone: BackboneType = backbone
        self.swin_window_size = int(swin_window_size)
        self.swin_num_heads = int(swin_num_heads)
        self.gan_loss_type: GANLossType = gan_loss_type
        self.norm_type: NormType = norm_type
        self.output_activation: OutputActivation = output_activation
        self.sigmoid_k = float(sigmoid_k)
        self.pixel_loss_type = pixel_loss_type
        self.use_adversarial_loss = bool(use_adversarial_loss)
        self.adversarial_weight = float(adversarial_weight)
        self.pretrain_epochs = int(pretrain_epochs)
        self.use_vgg_loss = bool(use_vgg_loss)
        self.vgg_loss_weight = float(vgg_loss_weight)
        self.vgg_layers = int(vgg_layers)
        self.vgg_pretrained = bool(vgg_pretrained)
        self.pore_is_low = bool(pore_is_low)
        self.conservation_loss_weight = float(conservation_loss_weight)
        self.conservation_psf_sigma = float(conservation_psf_sigma)
        self.topology_loss_weight = float(topology_loss_weight)
        self.gradient_loss_weight = float(gradient_loss_weight)
        self.ffl_loss_weight = float(ffl_loss_weight)
        self.disc_features = list(disc_features) if disc_features is not None else None
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_args = dict(
            lr_scheduler_args or {"T_0": 50, "T_mult": 1, "eta_min": 1e-6}
        )
        self.output_dir = output_dir
        self.save_val_images = bool(save_val_images)
        self.max_val_images_saved = int(max_val_images_saved)
        self.strict_loading = bool(strict_loading)

        if self.backbone == "rrdb":
            self.gen: nn.Module = RRDBGenerator(
                in_channels=self.context_slices,
                out_channels=self.out_channels,
                growth_channels=self.growth_channels,
                num_blocks=self.num_blocks,
                super_res_factor=self.super_res_factor,
                sigmoid_k=self.sigmoid_k,
                output_activation=self.output_activation,
                out_img_channels=1,
            )
        elif self.backbone == "swinir":
            self.gen = SwinIRGenerator(
                in_channels=self.context_slices,
                out_channels=self.out_channels,
                num_blocks=self.num_blocks,
                num_heads=self.swin_num_heads,
                window_size=self.swin_window_size,
                super_res_factor=self.super_res_factor,
                sigmoid_k=self.sigmoid_k,
                output_activation=self.output_activation,
                out_img_channels=1,
            )
        else:
            if (
                not isinstance(self.super_res_factor, int)
                or self.super_res_factor < 1
                or (self.super_res_factor & (self.super_res_factor - 1)) != 0
            ):
                raise ValueError(
                    f"backbone='srgan' requires integer power-of-2 super_res_factor (1, 2, 4, 8, ...), "
                    f"got {self.super_res_factor}. Use backbone='rrdb' or 'swinir' for 3x or fractional scales."
                )
            self.gen = Generator(
                in_channels=self.context_slices,
                out_channels=self.out_channels,
                num_blocks=self.num_blocks,
                super_res_factor=self.super_res_factor,
                sigmoid_k=self.sigmoid_k,
                norm_type=self.norm_type,
                output_activation=self.output_activation,
                out_img_channels=1,
            )

        # Allocate Discriminator ONLY when adversarial training is enabled (avoids multi-GB unused FC weights/Adam states)
        self.disc: Discriminator | None = (
            Discriminator(
                in_channels=1,
                features=self.disc_features,
                adaptive_size=self.adaptive_size,
                norm_type=self.norm_type,
            )
            if self.use_adversarial_loss
            else None
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.pixel_criterion: nn.Module = (
            nn.MSELoss() if self.pixel_loss_type == "mse" else nn.L1Loss()
        )
        self.vgg = (
            VGGLoss(vgg_layers=self.vgg_layers, pretrained=self.vgg_pretrained)
            if self.use_vgg_loss
            else None
        )
        self.conservation_criterion = AttenuationConservationLoss(
            psf_sigma=self.conservation_psf_sigma
        )
        self.topology_criterion = SoftCLDiceLoss(pore_is_low=self.pore_is_low)
        self.gradient_criterion = SobelGradientLoss()
        self.ffl_criterion = FocalFrequencyLoss()

        self.automatic_optimization = False
        self._predict_slice_counter = 0
        self._test_slice_counter = 0
        self._val_se_sum = 0.0
        self._val_elem_count = 0
        self._test_se_sum = 0.0
        self._test_elem_count = 0
        self._saved_radiometric_state: dict[str, object] | None = None

    def _capture_datamodule_radiometric_state(self) -> dict[str, object] | None:
        if self._trainer is not None:
            dm = getattr(self.trainer, "datamodule", None)
            export_fn = getattr(dm, "export_radiometric_state", None)
            if callable(export_fn):
                exported = export_fn()
                if isinstance(exported, dict):
                    self._saved_radiometric_state = dict(exported)
        return self._saved_radiometric_state

    def on_fit_start(self) -> None:
        self._capture_datamodule_radiometric_state()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = self._capture_datamodule_radiometric_state()
        if state is not None:
            checkpoint["openlsrx_radiometric_state"] = state

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        raw_state = checkpoint.get("openlsrx_radiometric_state")
        if isinstance(raw_state, dict):
            self._saved_radiometric_state = dict(raw_state)

    def forward(self, x: Tensor) -> Tensor:
        return self.gen(_ensure_4d(x))

    def _resolve_root_dir(self) -> Path:
        if self.output_dir:
            root = Path(self.output_dir)
        elif self._trainer is not None and self.trainer.log_dir:
            root = Path(self.trainer.log_dir)
        else:
            root = Path("outputs")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _is_adversarial_active(self) -> bool:
        return (
            self.use_adversarial_loss
            and self.disc is not None
            and (self.current_epoch >= self.pretrain_epochs)
        )

    def _compute_physics_loss(self, fake: Tensor, high_res: Tensor, low_res: Tensor) -> Tensor:
        phys = torch.zeros((), device=self.device, dtype=torch.float32)
        if self.conservation_loss_weight > 0.0:
            phys = phys + self.conservation_loss_weight * self.conservation_criterion(fake, low_res)
        if self.topology_loss_weight > 0.0:
            phys = phys + self.topology_loss_weight * self.topology_criterion(fake, high_res)
        if self.gradient_loss_weight > 0.0:
            phys = phys + self.gradient_loss_weight * self.gradient_criterion(fake, high_res)
        if self.ffl_loss_weight > 0.0:
            phys = phys + self.ffl_loss_weight * self.ffl_criterion(fake, high_res)
        return phys

    def _effective_accum_divisor(self, batch_idx: int) -> int:
        if self.accumulate_n_batches <= 1:
            return 1
        num_batches = getattr(self.trainer, "num_training_batches", 0)
        if (
            isinstance(num_batches, (int, float))
            and math.isfinite(num_batches)
            and int(num_batches) > 0
        ):
            total = int(num_batches)
            rem = total % self.accumulate_n_batches
            if rem != 0 and batch_idx >= (total - rem):
                return rem
        return self.accumulate_n_batches

    def _maybe_clip_gradients(self, optimizer: Any) -> None:
        clip_val = getattr(self.trainer, "gradient_clip_val", None)
        clip_algo = getattr(self.trainer, "gradient_clip_algorithm", None)
        if clip_val is not None and float(clip_val) > 0.0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=clip_algo,
            )

    def training_step(self, train_batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        low_res, high_res = train_batch
        low_res = _ensure_4d(low_res)
        high_res = _ensure_4d(high_res)

        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            opt_gen = optimizers[0]
            opt_disc = optimizers[1] if len(optimizers) > 1 else None
        else:
            opt_gen = optimizers
            opt_disc = None

        should_step = (
            (batch_idx + 1) % self.accumulate_n_batches == 0
        ) or self.trainer.is_last_batch
        eff_accum = float(self._effective_accum_divisor(batch_idx))
        adv_active = self._is_adversarial_active()

        # 1) Discriminator update (only when adversarial loss is active)
        disc_loss_val = torch.zeros((), device=self.device, dtype=torch.float32)
        if adv_active and opt_disc is not None and self.disc is not None:
            self.toggle_optimizer(opt_disc)
            with torch.no_grad():
                fake_detached = self.gen(low_res)
            disc_real = self.disc(high_res)
            disc_fake = self.disc(fake_detached)

            if self.gan_loss_type == "ragan":
                rel_real = disc_real - disc_fake.mean(dim=0, keepdim=True)
                rel_fake = disc_fake - disc_real.mean(dim=0, keepdim=True)
                disc_loss_real = self.bce(rel_real, torch.ones_like(rel_real))
                disc_loss_fake = self.bce(rel_fake, torch.zeros_like(rel_fake))
                disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
            else:
                # One-sided label smoothing on real targets in [0.9, 1.0]
                smooth_real = torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
                disc_loss_real = self.bce(disc_real, smooth_real)
                disc_loss_fake = self.bce(disc_fake, torch.zeros_like(disc_fake))
                disc_loss = disc_loss_real + disc_loss_fake
            disc_loss_val = disc_loss.detach().to(torch.float32)

            self.manual_backward(disc_loss.to(torch.float32) / eff_accum)
            if should_step:
                self._maybe_clip_gradients(opt_disc)
                opt_disc.step()
                opt_disc.zero_grad(set_to_none=True)
            self.untoggle_optimizer(opt_disc)

        # 2) Generator update
        self.toggle_optimizer(opt_gen)
        fake = self.gen(low_res)
        pixel_loss = self.pixel_criterion(fake, high_res).to(torch.float32)

        if adv_active and self.disc is not None:
            disc_fake_for_gen = self.disc(fake)
            if self.gan_loss_type == "ragan":
                with torch.no_grad():
                    disc_real_for_gen = self.disc(high_res)
                rel_fake_g = disc_fake_for_gen - disc_real_for_gen.mean(dim=0, keepdim=True)
                rel_real_g = disc_real_for_gen - disc_fake_for_gen.mean(dim=0, keepdim=True)
                raw_adv = 0.5 * (
                    self.bce(rel_fake_g, torch.ones_like(rel_fake_g))
                    + self.bce(rel_real_g, torch.zeros_like(rel_real_g))
                )
                adv_loss = (self.adversarial_weight * raw_adv).to(torch.float32)
            else:
                adv_loss = (
                    self.adversarial_weight
                    * self.bce(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
                ).to(torch.float32)
        else:
            adv_loss = torch.zeros((), device=self.device, dtype=torch.float32)

        if self.use_vgg_loss and self.vgg is not None:
            vgg_loss = (self.vgg_loss_weight * self.vgg(fake, high_res)).to(torch.float32)
        else:
            vgg_loss = torch.zeros((), device=self.device, dtype=torch.float32)

        physics_loss = self._compute_physics_loss(fake, high_res, low_res)

        gen_loss = pixel_loss + adv_loss + vgg_loss + physics_loss
        self.manual_backward(gen_loss / eff_accum)
        if should_step:
            self._maybe_clip_gradients(opt_gen)
            opt_gen.step()
            opt_gen.zero_grad(set_to_none=True)
        self.untoggle_optimizer(opt_gen)

        batch_size = int(low_res.shape[0])
        self.log(
            "train/gen_loss",
            gen_loss.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/pixel_loss",
            pixel_loss.detach(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/adv_loss",
            adv_loss.detach(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/vgg_loss",
            vgg_loss.detach(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/physics_loss",
            physics_loss.detach(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/disc_loss",
            disc_loss_val,
            prog_bar=self.use_adversarial_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def on_train_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        sched_list = schedulers if isinstance(schedulers, list) else [schedulers]
        for sched in sched_list:
            if (
                isinstance(sched, optim.lr_scheduler.LRScheduler)
                and int(getattr(sched.optimizer, "_step_count", 0)) > 1
            ):
                sched.step()

    def on_validation_epoch_start(self) -> None:
        self._val_se_sum = 0.0
        self._val_elem_count = 0

    @torch.no_grad()
    def validation_step(self, val_batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        low_res, high_res = val_batch
        low_4d = _ensure_4d(low_res)
        high_4d = _ensure_4d(high_res)
        upsampled = self.gen(low_4d)

        pred_f32 = upsampled.to(torch.float32)
        high_f32 = high_4d.to(torch.float32)

        batch_size = int(low_4d.shape[0])
        num_elems = int(pred_f32.numel())
        self._val_se_sum += float(torch.sum((pred_f32 - high_f32).square()).item())
        self._val_elem_count += num_elems

        self.log(
            "val/ssim",
            compute_ssim(pred_f32, high_f32),
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/mae",
            compute_mae(pred_f32, high_f32),
            on_epoch=True,
            batch_size=num_elems,
            sync_dist=True,
        )

        if self.save_val_images and self.global_rank == 0 and batch_idx < self.max_val_images_saved:
            stage_name = "sanity" if self.trainer.sanity_checking else "validation"
            epoch_dir = self._resolve_root_dir() / stage_name / f"epoch_{self.current_epoch:04d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)

            tifffile.imwrite(
                epoch_dir / f"upsampled_{batch_idx:04d}.tiff",
                pred_f32[0, 0].cpu().numpy(),
            )
            tifffile.imwrite(
                epoch_dir / f"high_{batch_idx:04d}.tiff",
                high_f32[0, 0].cpu().numpy(),
            )
            c_mid = low_4d.shape[1] // 2
            tifffile.imwrite(
                epoch_dir / f"low_{batch_idx:04d}.tiff",
                low_4d[0, c_mid].to(torch.float32).cpu().numpy(),
            )

    def _reduce_epoch_se(self, se_sum: float, elem_count: int) -> tuple[float, float]:
        stats = torch.tensor([se_sum, float(elem_count)], device=self.device, dtype=torch.float64)
        if self._trainer is not None and int(getattr(self.trainer, "world_size", 1)) > 1:
            gathered = torch.as_tensor(self.all_gather(stats))
            se_tot = float(gathered[..., 0].sum().item())
            cnt_tot = float(gathered[..., 1].sum().item())
        else:
            se_tot = float(stats[0].item())
            cnt_tot = float(stats[1].item())
        mse = se_tot / max(cnt_tot, 1.0)
        rmse = math.sqrt(max(mse, 0.0))
        psnr = 100.0 if mse <= 0.0 else -10.0 * math.log10(mse)
        return psnr, rmse

    def on_validation_epoch_end(self) -> None:
        if self._val_elem_count <= 0:
            return
        psnr, rmse = self._reduce_epoch_se(self._val_se_sum, self._val_elem_count)
        self.log("val/psnr", psnr, prog_bar=True, sync_dist=False)
        self.log("val/rmse", rmse, sync_dist=False)

    def on_test_epoch_start(self) -> None:
        self._test_slice_counter = 0
        self._test_se_sum = 0.0
        self._test_elem_count = 0

    @torch.no_grad()
    def test_step(self, test_batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        low_res, high_res = test_batch
        low_4d = _ensure_4d(low_res)
        high_4d = _ensure_4d(high_res)
        upsampled = self.gen(low_4d)

        pred_f32 = upsampled.to(torch.float32)
        high_f32 = high_4d.to(torch.float32)
        low_f32 = low_4d.to(torch.float32)
        batch_size = int(low_4d.shape[0])
        num_elems = int(pred_f32.numel())
        self._test_se_sum += float(torch.sum((pred_f32 - high_f32).square()).item())
        self._test_elem_count += num_elems

        self.log(
            "test/ssim",
            compute_ssim(pred_f32, high_f32),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "test/mae",
            compute_mae(pred_f32, high_f32),
            on_epoch=True,
            batch_size=num_elems,
            sync_dist=True,
        )

        if self.global_rank == 0:
            test_dir = self._resolve_root_dir() / "test_saved"
            test_dir.mkdir(parents=True, exist_ok=True)
            c_mid = low_f32.shape[1] // 2
            for i in range(batch_size):
                idx = self._test_slice_counter
                self._test_slice_counter += 1
                tifffile.imwrite(
                    test_dir / f"upsampled_{idx:05d}.tiff", pred_f32[i, 0].cpu().numpy()
                )
                tifffile.imwrite(test_dir / f"high_{idx:05d}.tiff", high_f32[i, 0].cpu().numpy())
                tifffile.imwrite(test_dir / f"low_{idx:05d}.tiff", low_f32[i, c_mid].cpu().numpy())

    def on_test_epoch_end(self) -> None:
        if self._test_elem_count <= 0:
            return
        psnr, rmse = self._reduce_epoch_se(self._test_se_sum, self._test_elem_count)
        self.log("test/psnr", psnr, sync_dist=False)
        self.log("test/rmse", rmse, sync_dist=False)

    def _predict_base_dir(self) -> Path:
        dm = getattr(self.trainer, "datamodule", None) if self._trainer is not None else None
        pred_axis = getattr(dm, "predict_axis", None)
        folder_name = f"predicted_{pred_axis}" if pred_axis else "predicted"
        return self._resolve_root_dir() / folder_name

    def _clean_stale_predict_slices(self, base_dir: Path) -> None:
        if not base_dir.is_dir():
            return
        prefix = f"{self.sample_id}_" if self.sample_id else ""
        escaped = re.escape(prefix)
        patterns = {
            "predicted": re.compile(rf"^{escaped}predicted_(\d+)\.tiff$"),
            "low_res": re.compile(rf"^{escaped}low_(\d+)\.tiff$"),
            "high_res": re.compile(rf"^{escaped}high_(\d+)\.tiff$"),
        }
        for subdir_name, pat in patterns.items():
            sdir = base_dir / subdir_name
            if not sdir.is_dir():
                continue
            for p in sdir.iterdir():
                if pat.match(p.name) is not None:
                    p.unlink(missing_ok=True)

    def on_predict_epoch_start(self) -> None:
        self._predict_slice_counter = 0
        if self._trainer is not None and int(getattr(self.trainer, "world_size", 1)) > 1:
            warnings.warn(
                "SRGAN.predict writes TIFF stacks on rank 0 only; run predict with devices=1 "
                "or `openlsrx predict-volume` so slices are not sharded across ranks.",
                UserWarning,
                stacklevel=2,
            )
        if self.global_rank == 0:
            self._clean_stale_predict_slices(self._predict_base_dir())

    @torch.no_grad()
    def predict_step(
        self,
        predict_batch: tuple[Tensor, Tensor] | Tensor,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        if isinstance(predict_batch, (tuple, list)):
            low_res, high_res = predict_batch[0], predict_batch[1]
        else:
            low_res, high_res = predict_batch, None

        low_4d = _ensure_4d(low_res)
        predicted_4d = self.gen(low_4d).to(torch.float32)
        low_f32 = low_4d.to(torch.float32)
        high_f32 = _ensure_4d(high_res).to(torch.float32) if high_res is not None else None

        if self.global_rank == 0:
            base_dir = self._predict_base_dir()
            pred_dir = base_dir / "predicted"
            low_dir = base_dir / "low_res"
            pred_dir.mkdir(parents=True, exist_ok=True)
            low_dir.mkdir(parents=True, exist_ok=True)
            if high_f32 is not None:
                high_dir = base_dir / "high_res"
                high_dir.mkdir(parents=True, exist_ok=True)

            prefix = f"{self.sample_id}_" if self.sample_id else ""
            batch_size = int(low_4d.shape[0])
            c_mid = low_f32.shape[1] // 2
            for i in range(batch_size):
                idx = self._predict_slice_counter
                self._predict_slice_counter += 1
                tifffile.imwrite(
                    pred_dir / f"{prefix}predicted_{idx:05d}.tiff",
                    predicted_4d[i, 0].cpu().numpy(),
                )
                tifffile.imwrite(
                    low_dir / f"{prefix}low_{idx:05d}.tiff",
                    low_f32[i, c_mid].cpu().numpy(),
                )
                if high_f32 is not None:
                    tifffile.imwrite(
                        base_dir / "high_res" / f"{prefix}high_{idx:05d}.tiff",
                        high_f32[i, 0].cpu().numpy(),
                    )

        return predicted_4d.squeeze(1)

    def on_predict_epoch_end(self) -> None:
        if self.global_rank != 0:
            return

        base_dir = self._predict_base_dir()
        dm = getattr(self.trainer, "datamodule", None) if self._trainer is not None else None
        pred_axis = getattr(dm, "predict_axis", None)

        prefix = f"{self.sample_id}_" if self.sample_id else ""
        escaped_prefix = re.escape(prefix)
        stack_specs = [
            (
                "predicted",
                f"{prefix}predicted_stack.tiff",
                re.compile(rf"^{escaped_prefix}predicted_(\d+)\.tiff$"),
            ),
            (
                "low_res",
                f"{prefix}low_res_stack.tiff",
                re.compile(rf"^{escaped_prefix}low_(\d+)\.tiff$"),
            ),
            (
                "high_res",
                f"{prefix}high_res_stack.tiff",
                re.compile(rf"^{escaped_prefix}high_(\d+)\.tiff$"),
            ),
        ]

        if base_dir.is_dir():
            for subdir_name, stack_filename, pattern in stack_specs:
                slice_dir = base_dir / subdir_name
                if not slice_dir.is_dir():
                    continue

                matched: list[tuple[int, Path]] = []
                for p in slice_dir.iterdir():
                    m = pattern.match(p.name)
                    if m is not None:
                        matched.append((int(m.group(1)), p))
                if not matched:
                    continue

                matched.sort(key=lambda item: item[0])
                output_stack = base_dir / stack_filename

                if pred_axis in AXIS_TO_DIM:
                    # Restore canonical (D, H, W) axis order after single-axis sweep along pred_axis
                    ordered_vol = np.stack([tifffile.imread(img_p) for _, img_p in matched], axis=0)
                    canonical_vol = np.ascontiguousarray(
                        np.moveaxis(ordered_vol, 0, AXIS_TO_DIM[pred_axis])
                    )
                    tifffile.imwrite(
                        output_stack,
                        canonical_vol,
                        bigtiff=True,
                        photometric="minisblack",
                    )
                else:
                    with tifffile.TiffWriter(output_stack, bigtiff=True) as writer:
                        for _, img_path in matched:
                            writer.write(
                                tifffile.imread(img_path),
                                photometric="minisblack",
                                contiguous=True,
                            )

        # True 3-axis orthogonal volumetric reconstruction + inter-view disagreement uncertainty (sigma_ortho)
        pred_lr_vol = getattr(dm, "predict_lr_volume", None) if dm is not None else None
        pred_hr_vol = getattr(dm, "predict_hr_volume", None) if dm is not None else None
        pred_mode = (
            getattr(dm, "predict_mode", "orthogonal_3axis") if dm is not None else "single_axis"
        )
        if pred_lr_vol is not None and pred_mode == "orthogonal_3axis":
            fusion_mode = cast(FusionMode, str(getattr(dm, "predict_fusion", "confidence_fourier")))
            eval_bs = int(getattr(dm, "eval_batch_size", 4))
            ortho = super_resolve_volume_orthogonal(
                generator=self.gen,
                low_volume=np.asarray(pred_lr_vol, dtype=np.float32),
                scale_factor=self.super_res_factor,
                axes=("yz", "xz", "xy"),
                fusion=fusion_mode,
                batch_size=eval_bs,
                device=self.device,
                context_slices=self.context_slices,
                return_uncertainty=True,
            )
            ortho_dir = self._resolve_root_dir() / "predicted_orthogonal"
            ortho_dir.mkdir(parents=True, exist_ok=True)
            fused_vol = np.ascontiguousarray(ortho["fused"], dtype=np.float32)
            unc_vol = np.ascontiguousarray(ortho["uncertainty_std"], dtype=np.float32)
            tifffile.imwrite(
                ortho_dir / f"{prefix}predicted_stack.tiff",
                fused_vol,
                bigtiff=True,
                photometric="minisblack",
            )
            tifffile.imwrite(
                ortho_dir / f"{prefix}uncertainty_std_stack.tiff",
                unc_vol,
                bigtiff=True,
                photometric="minisblack",
            )
            tifffile.imwrite(
                ortho_dir / f"{prefix}low_res_stack.tiff",
                np.ascontiguousarray(pred_lr_vol, dtype=np.float32),
                bigtiff=True,
                photometric="minisblack",
            )
            if pred_hr_vol is not None:
                tifffile.imwrite(
                    ortho_dir / f"{prefix}high_res_stack.tiff",
                    np.ascontiguousarray(pred_hr_vol, dtype=np.float32),
                    bigtiff=True,
                    photometric="minisblack",
                )
            if base_dir.is_dir():
                tifffile.imwrite(
                    base_dir / f"{prefix}predicted_orthogonal_fused.tiff",
                    fused_vol,
                    bigtiff=True,
                    photometric="minisblack",
                )
                tifffile.imwrite(
                    base_dir / f"{prefix}predicted_uncertainty_std.tiff",
                    unc_vol,
                    bigtiff=True,
                    photometric="minisblack",
                )
                # Backward-compatible filename alias
                tifffile.imwrite(
                    base_dir / f"{prefix}predicted_epistemic_std.tiff",
                    unc_vol,
                    bigtiff=True,
                    photometric="minisblack",
                )

    def configure_optimizers(self):
        opt_gen = optim.Adam(self.gen.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        if self.lr_scheduler_type == "constant":
            gen_scheduler: optim.lr_scheduler.LRScheduler = optim.lr_scheduler.LambdaLR(
                opt_gen, lr_lambda=lambda _step: 1.0
            )
        else:
            gen_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_gen, **self.lr_scheduler_args
            )

        if self.disc is None:
            return [{"optimizer": opt_gen, "lr_scheduler": gen_scheduler}]

        opt_disc = optim.Adam(self.disc.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        if self.lr_scheduler_type == "constant":
            disc_scheduler: optim.lr_scheduler.LRScheduler = optim.lr_scheduler.LambdaLR(
                opt_disc, lr_lambda=lambda _step: 1.0
            )
        else:
            disc_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_disc, **self.lr_scheduler_args
            )

        return [
            {"optimizer": opt_gen, "lr_scheduler": gen_scheduler},
            {"optimizer": opt_disc, "lr_scheduler": disc_scheduler},
        ]
