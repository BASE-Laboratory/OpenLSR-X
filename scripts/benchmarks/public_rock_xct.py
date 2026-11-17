#!/usr/bin/env python3
"""Open Rock Micro-CT Super-Resolution Benchmark (Bentheimer Sandstone & Indiana Limestone)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytorch_lightning as pl

from openlsrx.baselines import upsample_3d
from openlsrx.data import PairedVolumeDataModule, fetch_or_generate_rock_sample
from openlsrx.inference import super_resolve_volume_orthogonal
from openlsrx.metrics import evaluate_image_metrics, evaluate_petrophysical_metrics
from openlsrx.models import SRGAN


def run_public_rock_benchmark(
    output_dir: Path,
    data_dir: Path = Path("data/benchmarks"),
    volume_size: int = 128,
    scale_factor: int = 2,
    epochs: int = 120,
    seeds: list[int] | None = None,
    fast: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = seeds or ([42] if fast else [42, 43, 44])

    if fast:
        volume_size = 48
        epochs = 3
        num_blocks = 2
        out_channels = 16
        disc_features: list[int] | None = [16, 32]
    else:
        num_blocks = 12
        out_channels = 64
        disc_features = None

    rows: list[dict[str, object]] = []
    for rock_name in ("bentheimer_sandstone", "indiana_limestone"):
        hr_tif, manifest = fetch_or_generate_rock_sample(
            dataset_name=rock_name,
            dest_dir=data_dir,
            shape=(volume_size, volume_size, volume_size),
            allow_offline_fallback=True,
            seed=42,
        )
        for seed in seed_list:
            print(
                f"[public_rock_xct] Running {rock_name} (source={manifest['source_kind']}) seed={seed} ..."
            )
            dm = PairedVolumeDataModule(
                high_path=str(hr_tif),
                degrade_from_hr=True,
                scale_factor=scale_factor,
                psf_sigma=0.85,
                degrade_noise_std=0.025,
                norm_mode="percentile",
                train_frac=0.60,
                val_frac=0.15,
                guard_slices_lr=1,
                batch_size=4,
                eval_batch_size=4,
                seed=seed,
            )
            dm.setup("fit")
            lr_test, hr_test = dm.test_volumes

            for mode, name in (("trilinear", "Trilinear-3D"), ("tricubic", "Tricubic-3D")):
                pred = upsample_3d(lr_test, scale_factors=scale_factor, mode=mode)  # type: ignore[arg-type]
                row: dict[str, object] = {"dataset": rock_name, "seed": seed, "method": name}
                row.update(evaluate_image_metrics(pred, hr_test))
                row.update(evaluate_petrophysical_metrics(pred, hr_test, pore_is_low=True))
                rows.append(row)

            pl.seed_everything(seed, workers=True)
            model = SRGAN(
                learning_rate=2e-4 if fast else 5e-5,
                adaptive_size=4 if fast else 16,
                super_res_factor=scale_factor,
                num_blocks=num_blocks,
                out_channels=out_channels,
                disc_features=disc_features,
                pretrain_epochs=max(1, epochs // 4),
                use_adversarial_loss=True,
                use_vgg_loss=False,
                output_dir=str(output_dir / "runs" / f"{rock_name}_s{seed}"),
                save_val_images=False,
            )
            trainer = pl.Trainer(
                default_root_dir=str(output_dir / "runs"),
                max_epochs=epochs,
                accelerator="auto",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(model, datamodule=dm)

            vols = super_resolve_volume_orthogonal(
                model.gen, lr_test, scale_factor=scale_factor, fusion="fourier"
            )
            for key, label, ax_idx in (
                ("yz", "OpenLSR-X-SingleAxis(yz)", 0),
                ("fused", "OpenLSR-X-3Axis-Fourier", 0),
            ):
                row = {"dataset": rock_name, "seed": seed, "method": label}
                row.update(evaluate_image_metrics(vols[key], hr_test))
                row.update(
                    evaluate_petrophysical_metrics(
                        vols[key], hr_test, pore_is_low=True, stack_axis=ax_idx
                    )
                )
                rows.append(row)

    csv_path = output_dir / "public_rock_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[public_rock_xct] Saved {len(rows)} rows to {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Open Rock Micro-CT SR Benchmark.")
    parser.add_argument("--output-dir", default="results/public_rock", help="Output directory.")
    parser.add_argument("--data-dir", default="data/benchmarks", help="Benchmark data cache.")
    parser.add_argument("--volume-size", type=int, default=128, help="Cubic HR side length.")
    parser.add_argument("--scale-factor", type=int, default=2, choices=[2, 4], help="SR factor.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds.")
    parser.add_argument("--fast", action="store_true", help="Rapid CPU smoke benchmark.")
    ns = parser.parse_args()

    run_public_rock_benchmark(
        output_dir=Path(ns.output_dir),
        data_dir=Path(ns.data_dir),
        volume_size=ns.volume_size,
        scale_factor=ns.scale_factor,
        epochs=ns.epochs,
        seeds=ns.seeds,
        fast=ns.fast,
    )


if __name__ == "__main__":
    main()
