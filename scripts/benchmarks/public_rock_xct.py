#!/usr/bin/env python3
"""Open Rock Micro-CT Super-Resolution Benchmark (Bentheimer Sandstone & Indiana Limestone)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from openlsrx.baselines import upsample_3d
from openlsrx.data import (
    PairedVolumeDataModule,
    fetch_or_generate_rock_sample,
    prepare_bentheimer_dualscan_benchmark,
)
from openlsrx.inference import super_resolve_volume_orthogonal
from openlsrx.metrics import (
    evaluate_image_metrics,
    evaluate_petrophysical_metrics,
    evaluate_spectral_metrics,
    evaluate_transport_metrics,
    precompute_reference_transport,
)
from openlsrx.models import SRGAN


def run_public_rock_benchmark(
    output_dir: Path,
    data_dir: Path = Path("data/benchmarks"),
    volume_size: int = 128,
    scale_factor: int = 2,
    epochs: int = 120,
    seeds: list[int] | None = None,
    fast: bool = False,
    allow_synthetic_fallback: bool | None = None,
    dualscan: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = seeds or ([42] if fast else [42, 43, 44])
    fallback = bool(fast) if allow_synthetic_fallback is None else bool(allow_synthetic_fallback)
    eff_scale = 3 if dualscan else scale_factor
    voxel_um = 6.0 if dualscan else 1.0

    if fast:
        volume_size = 48 if not dualscan else 72
        epochs = 3
        num_blocks = 2
        out_channels = 16
        disc_features: list[int] | None = [16, 32]
    else:
        num_blocks = 6 if dualscan else 12
        out_channels = 48 if dualscan else 64
        disc_features = None

    rows: list[dict[str, object]] = []
    dataset_names = (
        ("bentheimer_dualscan_core1", "bentheimer_dualscan_core2")
        if dualscan
        else ("bentheimer_sandstone", "indiana_limestone")
    )
    for rock_name in dataset_names:
        lr_tif: Path | None = None
        if dualscan:
            lr_side = min(75, max(18, volume_size // 3))
            lr_tif, hr_tif, manifest = prepare_bentheimer_dualscan_benchmark(
                dataset_name=rock_name,
                dest_dir=data_dir,
                lr_shape=(lr_side, lr_side, lr_side),
                allow_offline_fallback=fallback,
                prefer_offline=fast,
                seed=42,
            )
        else:
            hr_tif, manifest = fetch_or_generate_rock_sample(
                dataset_name=rock_name,
                dest_dir=data_dir,
                shape=(volume_size, volume_size, volume_size),
                allow_offline_fallback=fallback,
                prefer_offline=fast,
                seed=42,
            )
        src_kind = str(manifest.get("source_kind", "unknown"))
        is_exp = int(bool(manifest.get("is_experimental_xct", False)))
        for seed in seed_list:
            print(f"[public_rock_xct] Running {rock_name} (source={src_kind}) seed={seed} ...")
            dm = PairedVolumeDataModule(
                high_path=str(hr_tif),
                low_path=str(lr_tif) if lr_tif is not None else None,
                degrade_from_hr=(not dualscan),
                scale_factor=eff_scale,
                psf_sigma=0.85,
                degrade_noise_std=0.025,
                norm_mode="percentile",
                cross_scan_calibration="quantile" if dualscan else "none",
                calibration_psf_sigma=1.0 if dualscan else None,
                train_frac=0.60,
                val_frac=0.15,
                guard_slices_lr=1,
                train_axes=("yz", "xz", "xy") if dualscan else ("yz",),
                context_slices=3 if dualscan else 1,
                batch_size=4,
                eval_batch_size=4,
                seed=seed,
            )
            dm.setup("fit")
            lr_test, hr_test = dm.test_volumes
            ref_transport = precompute_reference_transport(
                hr_test, pore_is_low=True, voxel_size=voxel_um
            )

            for mode, name in (("trilinear", "Trilinear-3D"), ("tricubic", "Tricubic-3D")):
                pred = upsample_3d(lr_test, scale_factors=eff_scale, mode=mode)  # type: ignore[arg-type]
                row: dict[str, object] = {
                    "dataset": rock_name,
                    "source_kind": src_kind,
                    "is_experimental_xct": is_exp,
                    "seed": seed,
                    "method": name,
                }
                row.update(evaluate_image_metrics(pred, hr_test))
                row.update(evaluate_spectral_metrics(pred, hr_test, voxel_size_um=voxel_um))
                row.update(evaluate_petrophysical_metrics(pred, hr_test, pore_is_low=True))
                row.update(
                    evaluate_transport_metrics(
                        pred,
                        hr_test,
                        pore_is_low=True,
                        voxel_size=voxel_um,
                        precomputed_true=ref_transport,
                    )
                )
                rows.append(row)

            pl.seed_everything(seed, workers=True)
            run_dir = output_dir / "runs" / f"{rock_name}_s{seed}"
            model = SRGAN(
                backbone="swinir" if (dualscan or eff_scale == 3) else "srgan",
                context_slices=3 if dualscan else 1,
                learning_rate=3e-4 if dualscan else (2e-4 if fast else 5e-5),
                adaptive_size=4 if fast else 8,
                super_res_factor=eff_scale,
                num_blocks=num_blocks,
                out_channels=out_channels,
                disc_features=disc_features,
                pretrain_epochs=max(1, epochs // 4),
                use_adversarial_loss=(not dualscan),
                use_vgg_loss=False,
                output_dir=str(run_dir),
                save_val_images=False,
            )
            ckpt_cb: ModelCheckpoint | None = None
            callbacks: list[pl.Callback] = []
            if not fast:
                ckpt_cb = ModelCheckpoint(
                    dirpath=str(run_dir / "checkpoints"),
                    filename="best",
                    monitor="val/psnr",
                    mode="max",
                    save_top_k=1,
                )
                callbacks.append(ckpt_cb)
            trainer = pl.Trainer(
                default_root_dir=str(output_dir / "runs"),
                max_epochs=epochs,
                accelerator="auto",
                devices=1,
                logger=False,
                enable_checkpointing=(not fast),
                callbacks=callbacks,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(model, datamodule=dm)
            if ckpt_cb is not None and ckpt_cb.best_model_path:
                ckpt = torch.load(
                    ckpt_cb.best_model_path, map_location=model.device, weights_only=False
                )
                model.load_state_dict(ckpt["state_dict"], strict=False)

            vols = super_resolve_volume_orthogonal(
                model.gen,
                lr_test,
                scale_factor=eff_scale,
                fusion="confidence_fourier" if dualscan else "fourier",
                context_slices=3 if dualscan else 1,
            )
            for key, label, ax_idx in (
                ("yz", "OpenLSR-X-SingleAxis(yz)", 0),
                ("fused", "OpenLSR-X-3Axis-Fourier", 0),
            ):
                row = {
                    "dataset": rock_name,
                    "source_kind": src_kind,
                    "is_experimental_xct": is_exp,
                    "seed": seed,
                    "method": label,
                }
                row.update(evaluate_image_metrics(vols[key], hr_test))
                row.update(evaluate_spectral_metrics(vols[key], hr_test, voxel_size_um=voxel_um))
                row.update(
                    evaluate_petrophysical_metrics(
                        vols[key], hr_test, pore_is_low=True, stack_axis=ax_idx
                    )
                )
                row.update(
                    evaluate_transport_metrics(
                        vols[key],
                        hr_test,
                        pore_is_low=True,
                        voxel_size=voxel_um,
                        precomputed_true=ref_transport,
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
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (defaults to results/public_rock, or results/public_rock_dualscan with --dualscan).",
    )
    parser.add_argument("--data-dir", default="data/benchmarks", help="Benchmark data cache.")
    parser.add_argument("--volume-size", type=int, default=128, help="Cubic HR side length.")
    parser.add_argument("--scale-factor", type=int, default=2, choices=[2, 3, 4], help="SR factor.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds.")
    parser.add_argument("--fast", action="store_true", help="Rapid CPU smoke benchmark.")
    parser.add_argument(
        "--dualscan",
        action="store_true",
        help="Run physical 18 um -> 6 um (3x) Bentheimer dual-scan benchmark (Zenodo 5542624).",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow synthetic fallback if live Zenodo download fails during a full campaign.",
    )
    ns = parser.parse_args()
    resolved_out = (
        Path(ns.output_dir)
        if ns.output_dir is not None
        else Path("results/public_rock_dualscan" if ns.dualscan else "results/public_rock")
    )

    run_public_rock_benchmark(
        output_dir=resolved_out,
        data_dir=Path(ns.data_dir),
        volume_size=ns.volume_size,
        scale_factor=ns.scale_factor,
        epochs=ns.epochs,
        seeds=ns.seeds,
        fast=ns.fast,
        allow_synthetic_fallback=(True if ns.allow_synthetic_fallback else None),
        dualscan=bool(ns.dualscan),
    )


if __name__ == "__main__":
    main()
