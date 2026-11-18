#!/usr/bin/env python3
"""Reproducible 2x2 Factorial PoreSpy 3D XCT Super-Resolution Benchmark.

Evaluates 2D-trained SRResNet and SRGAN (in-plane ``yz``, orthogonal ``xy``, and
3-axis Fourier fusion) against 3D Nearest, Trilinear, and Tricubic baselines across
Low/High Noise x Low/High Feature-Frequency porous-media regimes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytorch_lightning as pl
import tifffile
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from openlsrx.baselines import upsample_3d
from openlsrx.data import FeatureRegime, NoiseRegime, SyntheticPoreSpyDataModule
from openlsrx.inference import super_resolve_volume_orthogonal
from openlsrx.metrics import evaluate_image_metrics, evaluate_petrophysical_metrics
from openlsrx.models import SRGAN


def _score_reconstruction(
    dataset_tag: str,
    seed: int,
    method: str,
    pred_vol,
    hr_vol,
    stack_axis: int = 0,
) -> dict[str, object]:
    img_m = evaluate_image_metrics(pred_vol, hr_vol)
    petro_m = evaluate_petrophysical_metrics(
        pred_vol, hr_vol, pore_is_low=True, stack_axis=stack_axis
    )
    row: dict[str, object] = {
        "regime": dataset_tag,
        "seed": seed,
        "method": method,
    }
    row.update(img_m)
    row.update(petro_m)
    return row


def _train_model(
    dm: SyntheticPoreSpyDataModule,
    work_dir: Path,
    scale_factor: int,
    epochs: int,
    num_blocks: int,
    out_channels: int,
    disc_features: list[int] | None,
    use_adversarial: bool,
    use_vgg: bool,
    vgg_pretrained: bool,
    seed: int,
) -> SRGAN:
    pl.seed_everything(seed, workers=True)
    model = SRGAN(
        learning_rate=2e-4 if epochs <= 10 else 1e-4,
        adaptive_size=4 if out_channels <= 16 else 8,
        super_res_factor=scale_factor,
        num_blocks=num_blocks,
        out_channels=out_channels,
        disc_features=disc_features,
        pixel_loss_type="l1" if not use_adversarial else "mse",
        use_adversarial_loss=use_adversarial,
        pretrain_epochs=max(1, epochs // 4) if use_adversarial else 0,
        use_vgg_loss=use_vgg,
        vgg_pretrained=vgg_pretrained,
        output_dir=str(work_dir),
        save_val_images=False,
    )
    use_ckpt = epochs > 10
    ckpt_cb: ModelCheckpoint | None = None
    callbacks: list[pl.Callback] = []
    if use_ckpt:
        ckpt_cb = ModelCheckpoint(
            dirpath=str(work_dir / "checkpoints"),
            filename="best",
            monitor="val/psnr",
            mode="max",
            save_top_k=1,
        )
        callbacks.append(ckpt_cb)
    trainer = pl.Trainer(
        default_root_dir=str(work_dir),
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=use_ckpt,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)
    if ckpt_cb is not None and ckpt_cb.best_model_path:
        ckpt = torch.load(ckpt_cb.best_model_path, map_location=model.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
    return model


def run_porespy_study(
    output_dir: Path,
    volume_size: int = 128,
    scale_factor: int = 2,
    epochs: int = 100,
    seeds: list[int] | None = None,
    fast: bool = False,
    use_vgg: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = seeds or ([42] if fast else [42, 43, 44])

    if fast:
        volume_size = 48
        epochs = 3
        num_blocks = 2
        out_channels = 16
        disc_features: list[int] | None = [16, 32]
        regimes: list[tuple[NoiseRegime, FeatureRegime]] = [
            ("low_noise", "high_feature"),
            ("high_noise", "high_feature"),
        ]
    else:
        num_blocks = 12
        out_channels = 64
        disc_features = None
        regimes = [
            ("low_noise", "low_feature"),
            ("low_noise", "high_feature"),
            ("high_noise", "low_feature"),
            ("high_noise", "high_feature"),
        ]

    rows: list[dict[str, object]] = []
    saved_example = False

    for noise_regime, feature_regime in regimes:
        tag = f"{noise_regime}__{feature_regime}"
        for seed in seed_list:
            print(f"[porespy_study] Running regime={tag} seed={seed} ...")
            dm = SyntheticPoreSpyDataModule(
                volume_size=volume_size,
                feature_regime=feature_regime,
                noise_regime=noise_regime,
                representation="attenuation",
                scale_factor=scale_factor,
                train_frac=0.60,
                val_frac=0.15,
                guard_slices_lr=1,
                batch_size=4,
                eval_batch_size=4,
                seed=seed,
                cache_dir=str(output_dir / "cache"),
            )
            dm.setup("fit")
            lr_test, hr_test = dm.test_volumes

            # 1) Classical 3D interpolation baselines
            for interp_mode, label in (
                ("nearest", "Nearest-3D"),
                ("trilinear", "Trilinear-3D"),
                ("tricubic", "Tricubic-3D"),
            ):
                pred_interp = upsample_3d(lr_test, scale_factors=scale_factor, mode=interp_mode)  # type: ignore[arg-type]
                rows.append(_score_reconstruction(tag, seed, label, pred_interp, hr_test))

            # 2) SRResNet (generator-only ablation)
            srresnet = _train_model(
                dm=dm,
                work_dir=output_dir / "runs" / f"{tag}_s{seed}_srresnet",
                scale_factor=scale_factor,
                epochs=epochs,
                num_blocks=num_blocks,
                out_channels=out_channels,
                disc_features=disc_features,
                use_adversarial=False,
                use_vgg=False,
                vgg_pretrained=False,
                seed=seed,
            )
            srres_vols = super_resolve_volume_orthogonal(
                srresnet.gen, lr_test, scale_factor=scale_factor, fusion="fourier"
            )
            rows.append(
                _score_reconstruction(
                    tag, seed, "SRResNet-SingleAxis(yz)", srres_vols["yz"], hr_test, stack_axis=0
                )
            )
            rows.append(
                _score_reconstruction(
                    tag, seed, "SRResNet-3Axis-Fourier", srres_vols["fused"], hr_test, stack_axis=0
                )
            )

            # 3) OpenLSR-X SRGAN
            srgan = _train_model(
                dm=dm,
                work_dir=output_dir / "runs" / f"{tag}_s{seed}_srgan",
                scale_factor=scale_factor,
                epochs=epochs,
                num_blocks=num_blocks,
                out_channels=out_channels,
                disc_features=disc_features,
                use_adversarial=True,
                use_vgg=use_vgg,
                vgg_pretrained=(use_vgg and not fast),
                seed=seed,
            )
            srgan_vols = super_resolve_volume_orthogonal(
                srgan.gen, lr_test, scale_factor=scale_factor, fusion="fourier"
            )
            rows.append(
                _score_reconstruction(
                    tag, seed, "OpenLSR-X-InPlane(yz)", srgan_vols["yz"], hr_test, stack_axis=0
                )
            )
            rows.append(
                _score_reconstruction(
                    tag, seed, "OpenLSR-X-Orthogonal(xy)", srgan_vols["xy"], hr_test, stack_axis=2
                )
            )
            rows.append(
                _score_reconstruction(
                    tag, seed, "OpenLSR-X-3Axis-Fourier", srgan_vols["fused"], hr_test, stack_axis=0
                )
            )

            if not saved_example:
                vol_dir = output_dir / "example_volumes"
                vol_dir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(vol_dir / "lr_test.tif", lr_test)
                tifffile.imwrite(vol_dir / "hr_test.tif", hr_test)
                tifffile.imwrite(
                    vol_dir / "tricubic_test.tif",
                    upsample_3d(lr_test, scale_factors=scale_factor, mode="tricubic"),
                )
                tifffile.imwrite(vol_dir / "srgan_yz_test.tif", srgan_vols["yz"])
                tifffile.imwrite(vol_dir / "srgan_fused_test.tif", srgan_vols["fused"])
                saved_example = True

    csv_path = output_dir / "porespy_study_results.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[porespy_study] Saved {len(rows)} benchmark rows to {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Factorial PoreSpy 3D XCT SR Benchmark.")
    parser.add_argument("--output-dir", default="results/porespy_study", help="Output directory.")
    parser.add_argument("--volume-size", type=int, default=128, help="Cubic HR volume size.")
    parser.add_argument("--scale-factor", type=int, default=2, choices=[2, 4], help="SR factor.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds.")
    parser.add_argument("--use-vgg", action="store_true", help="Enable VGG-19 perceptual loss.")
    parser.add_argument("--fast", action="store_true", help="Run rapid CPU smoke benchmark.")
    ns = parser.parse_args()

    run_porespy_study(
        output_dir=Path(ns.output_dir),
        volume_size=ns.volume_size,
        scale_factor=ns.scale_factor,
        epochs=ns.epochs,
        seeds=ns.seeds,
        fast=ns.fast,
        use_vgg=ns.use_vgg,
    )


if __name__ == "__main__":
    main()
