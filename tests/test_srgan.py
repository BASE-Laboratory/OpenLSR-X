"""Integration tests for the SRGAN LightningModule, checkpointing, and BigTIFF export."""

from __future__ import annotations

import os
from pathlib import Path

import pytorch_lightning as pl
import tifffile
import torch
from torch.utils.data import DataLoader

from openlsrx.data import PairedSliceDataset
from openlsrx.models import SRGAN


def test_srgan_end_to_end_fit_test_predict_and_checkpoint(tmp_path: Path) -> None:
    cwd_before = Path.cwd()

    low = torch.rand(6, 16, 16)
    high = torch.rand(6, 32, 32)
    ds = PairedSliceDataset(low, high, scale_factor=2, augment=True)
    eval_ds = PairedSliceDataset(low[:4], high[:4], scale_factor=2, augment=False)

    train_loader = DataLoader(ds, batch_size=2, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False)

    out_dir = tmp_path / "run_outputs"
    model = SRGAN(
        learning_rate=1e-3,
        adaptive_size=2,
        accumulate_n_batches=2,
        super_res_factor=2,
        num_blocks=1,
        out_channels=8,
        disc_features=[8, 8],
        pretrain_epochs=1,
        use_adversarial_loss=True,
        use_vgg_loss=True,
        vgg_layers=6,
        vgg_pretrained=False,
        sample_id="unit",
        output_dir=str(out_dir),
    )

    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)
    assert "val/psnr" in trainer.callback_metrics
    assert "val/ssim" in trainer.callback_metrics

    test_metrics = trainer.test(model, dataloaders=eval_loader, verbose=False)
    assert len(test_metrics) == 1
    assert "test/psnr" in test_metrics[0]

    preds = trainer.predict(model, dataloaders=eval_loader)
    assert preds is not None and len(preds) == 2

    # Verify working directory was NEVER mutated by validation, test, or predict steps
    assert Path(os.getcwd()) == cwd_before

    # Verify multi-page BigTIFF stacks contain all 4 slices across both batches
    pred_stack_path = out_dir / "predicted" / "unit_predicted_stack.tiff"
    assert pred_stack_path.is_file()
    stack = tifffile.imread(pred_stack_path)
    assert stack.shape == (4, 32, 32)

    # Verify strict checkpoint save + load (without VGG weight bloat)
    ckpt_path = tmp_path / "strict_roundtrip.ckpt"
    trainer.save_checkpoint(ckpt_path)
    reloaded = SRGAN.load_from_checkpoint(ckpt_path, strict=True)
    assert reloaded.sample_id == "unit"
    assert reloaded.super_res_factor == 2


def test_srresnet_ablation_mode(tmp_path: Path) -> None:
    low = torch.rand(4, 12, 12)
    high = torch.rand(4, 24, 24)
    loader = DataLoader(PairedSliceDataset(low, high, scale_factor=2), batch_size=2)

    model = SRGAN(
        learning_rate=1e-3,
        adaptive_size=2,
        super_res_factor=2,
        num_blocks=1,
        out_channels=8,
        disc_features=[8],
        pixel_loss_type="l1",
        use_adversarial_loss=False,
        use_vgg_loss=False,
        lr_scheduler_type="CAWR",
        lr_scheduler_args={"T_0": 2, "T_mult": 1, "eta_min": 1e-5},
        output_dir=str(tmp_path / "srresnet_out"),
        save_val_images=False,
    )
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    assert float(trainer.callback_metrics["train/disc_loss"]) == 0.0
