#!/usr/bin/env python3
"""Diamond Light Source I13-2 Multi-Resolution Zn-13X Synchrotron XCT & Transport Study.

Benchmarks 3D interpolation, 2D+Fourier SRResNet, memory-matched Native 3D CNN (SR3DResNet),
classical SRGAN (1-axis vs 3-axis Fourier), and normalization-free RRDB-RaGAN (3-axis Fourier)
across image fidelity, 3D morphology/topology, and 3D Fickian diffusive tortuosity + drainage.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from openlsrx.baselines import upsample_3d
from openlsrx.data import (
    PairedVolumeDataModule,
    prepare_zn13x_benchmark,
    synthesize_zn13x_phantom_pair,
)
from openlsrx.inference import super_resolve_volume_orthogonal
from openlsrx.io import export_for_openimpala
from openlsrx.metrics import (
    evaluate_image_metrics,
    evaluate_petrophysical_metrics,
    evaluate_transport_metrics,
)
from openlsrx.models import (
    SRGAN,
    SR3DResNet,
    build_3d_subcube_dataloaders,
    super_resolve_volume_3d,
)


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _evaluate_method(
    dataset_name: str,
    pair: str,
    seed: int,
    method: str,
    pred_vol: np.ndarray,
    hr_test: np.ndarray,
    voxel_size_um: float,
    stack_axis: int = 0,
    gen_params: int = 0,
    train_sec: float = 0.0,
    infer_sec: float = 0.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset_name,
        "pair": pair,
        "seed": seed,
        "method": method,
        "gen_params": gen_params,
        "train_sec": round(train_sec, 4),
        "infer_sec": round(infer_sec, 4),
    }
    row.update(evaluate_image_metrics(pred_vol, hr_test))
    row.update(
        evaluate_petrophysical_metrics(
            pred_vol,
            hr_test,
            pore_is_low=True,
            voxel_size=voxel_size_um,
            stack_axis=stack_axis,
        )
    )
    row.update(
        evaluate_transport_metrics(
            pred_vol,
            hr_test,
            pore_is_low=True,
            voxel_size=voxel_size_um,
            drainage_steps=10,
        )
    )
    return row


def run_zn13x_study(
    output_dir: Path,
    data_dir: Path = Path("data/benchmarks"),
    pairs: list[str] | None = None,
    lr_size: int = 48,
    epochs: int = 120,
    seeds: list[int] | None = None,
    raw_h5_dir: Path | None = None,
    fast: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_list = pairs or (["2x"] if fast else ["2x", "8x"])
    seed_list = seeds or ([42] if fast else [42, 43, 44])

    if fast:
        lr_size = 20
        epochs = 3
        srgan_blocks = 2
        rrdb_blocks = 2
        out_channels = 16
        growth_channels = 8
        disc_features: list[int] | None = [16, 32]
        c3d_channels = 12
        c3d_blocks = 2
        c3d_samples = 8
    else:
        srgan_blocks = 10
        rrdb_blocks = 8
        out_channels = 64
        growth_channels = 32
        disc_features = [32, 64, 128, 128]
        c3d_channels = 28
        c3d_blocks = 4
        c3d_samples = 48

    rows: list[dict[str, object]] = []

    for pair in pair_list:
        pair_lit: Literal["2x", "8x"] = "2x" if pair == "2x" else "8x"
        scale = 2 if pair_lit == "2x" else 8
        cur_lr_size = max(12, lr_size // 2) if (pair_lit == "8x" and fast) else lr_size
        lr_tif, hr_tif, manifest = prepare_zn13x_benchmark(
            pair=pair_lit,
            dest_dir=data_dir,
            lr_size=cur_lr_size,
            raw_h5_dir=raw_h5_dir,
            allow_offline_fallback=True,
            seed=42,
        )
        dataset_name = f"zn13x_{pair_lit}"
        voxel_um = float(manifest["high_resolution_um"])  # type: ignore[arg-type]

        for seed in seed_list:
            print(
                f"[zn13x_study] Running {dataset_name} ({scale}x, source={manifest['source_kind']}) seed={seed} ..."
            )
            dm = PairedVolumeDataModule(
                low_path=str(lr_tif),
                high_path=str(hr_tif),
                scale_factor=scale,
                norm_mode="none",
                train_frac=0.65,
                val_frac=0.20,
                guard_slices_lr=1,
                patch_size_lr=min(16, cur_lr_size),
                batch_size=4 if scale == 2 else 2,
                eval_batch_size=4 if scale == 2 else 2,
                seed=seed,
            )
            dm.setup("fit")
            lr_train, hr_train = dm.splits_3d["train"]
            lr_val, hr_val = dm.splits_3d["val"]

            if manifest["source_kind"] == "offline_calibrated_zn13x_phantom":
                lr_test, hr_test = synthesize_zn13x_phantom_pair(
                    pair=pair_lit,
                    lr_shape=(cur_lr_size, cur_lr_size, cur_lr_size),
                    seed=seed + 1000,
                )
            else:
                lr_test, hr_test = dm.test_volumes

            # 1) 3D Tricubic baseline
            t0 = time.perf_counter()
            pred_tricubic = upsample_3d(lr_test, scale_factors=scale, mode="tricubic")
            infer_tricubic = time.perf_counter() - t0
            rows.append(
                _evaluate_method(
                    dataset_name,
                    pair_lit,
                    seed,
                    "Tricubic-3D",
                    pred_tricubic,
                    hr_test,
                    voxel_um,
                    infer_sec=infer_tricubic,
                )
            )

            # 2) SRResNet (2D + 3-Axis Fourier, generator-only ablation)
            pl.seed_everything(seed, workers=True)
            srresnet = SRGAN(
                learning_rate=2e-4 if fast else 1e-4,
                adaptive_size=4 if fast else 8,
                super_res_factor=scale,
                num_blocks=srgan_blocks,
                out_channels=out_channels,
                disc_features=disc_features,
                use_adversarial_loss=False,
                use_vgg_loss=False,
                output_dir=str(output_dir / "runs" / f"{dataset_name}_s{seed}_srresnet"),
                save_val_images=False,
            )
            trainer_srres = pl.Trainer(
                default_root_dir=str(output_dir / "runs"),
                max_epochs=epochs,
                accelerator="auto",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            t0 = time.perf_counter()
            trainer_srres.fit(srresnet, datamodule=dm)
            train_srres = time.perf_counter() - t0

            t0 = time.perf_counter()
            srres_vols = super_resolve_volume_orthogonal(
                srresnet.gen, lr_test, scale_factor=scale, fusion="fourier"
            )
            infer_srres = time.perf_counter() - t0
            rows.append(
                _evaluate_method(
                    dataset_name,
                    pair_lit,
                    seed,
                    "SRResNet-3Axis-Fourier",
                    srres_vols["fused"],
                    hr_test,
                    voxel_um,
                    gen_params=_count_params(srresnet.gen),
                    train_sec=train_srres,
                    infer_sec=infer_srres,
                )
            )

            # 3) Native 3D CNN baseline (SR3DResNet)
            pl.seed_everything(seed, workers=True)
            sr3d = SR3DResNet(
                learning_rate=2e-4,
                super_res_factor=scale,
                out_channels=c3d_channels,
                num_blocks=c3d_blocks,
            )
            tr_3d_loader, va_3d_loader = build_3d_subcube_dataloaders(
                lr_train,
                hr_train,
                lr_val,
                hr_val,
                scale_factor=scale,
                patch_size_lr=min(12, int(min(lr_train.shape))),
                train_samples=c3d_samples,
                val_samples=max(2, c3d_samples // 4),
                batch_size=2 if scale == 2 else 1,
                seed=seed,
            )
            trainer_3d = pl.Trainer(
                default_root_dir=str(output_dir / "runs"),
                max_epochs=epochs,
                accelerator="auto",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            t0 = time.perf_counter()
            trainer_3d.fit(sr3d, train_dataloaders=tr_3d_loader, val_dataloaders=va_3d_loader)
            train_3d = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred_3d = super_resolve_volume_3d(
                sr3d.gen,
                lr_test,
                scale_factor=scale,
                patch_size_lr=min(16, cur_lr_size),
                overlap_lr=4,
            )
            infer_3d = time.perf_counter() - t0
            rows.append(
                _evaluate_method(
                    dataset_name,
                    pair_lit,
                    seed,
                    "SR3DResNet-Native3D",
                    pred_3d,
                    hr_test,
                    voxel_um,
                    gen_params=_count_params(sr3d.gen),
                    train_sec=train_3d,
                    infer_sec=infer_3d,
                )
            )

            # 4 & 5) Classical OpenLSR-X SRGAN (Single-Axis yz vs 3-Axis Fourier)
            pl.seed_everything(seed, workers=True)
            srgan = SRGAN(
                learning_rate=2e-4 if fast else 1e-4,
                adaptive_size=4 if fast else 8,
                super_res_factor=scale,
                num_blocks=srgan_blocks,
                out_channels=out_channels,
                disc_features=disc_features,
                pretrain_epochs=max(1, epochs // 3),
                use_adversarial_loss=True,
                use_vgg_loss=False,
                output_dir=str(output_dir / "runs" / f"{dataset_name}_s{seed}_srgan"),
                save_val_images=False,
            )
            trainer_srgan = pl.Trainer(
                default_root_dir=str(output_dir / "runs"),
                max_epochs=epochs,
                accelerator="auto",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            t0 = time.perf_counter()
            trainer_srgan.fit(srgan, datamodule=dm)
            train_srgan = time.perf_counter() - t0

            t0 = time.perf_counter()
            srgan_vols = super_resolve_volume_orthogonal(
                srgan.gen, lr_test, scale_factor=scale, fusion="fourier"
            )
            infer_srgan = time.perf_counter() - t0
            rows.append(
                _evaluate_method(
                    dataset_name,
                    pair_lit,
                    seed,
                    "OpenLSR-X-SingleAxis(yz)",
                    srgan_vols["yz"],
                    hr_test,
                    voxel_um,
                    stack_axis=0,
                    gen_params=_count_params(srgan.gen),
                    train_sec=train_srgan,
                    infer_sec=infer_srgan / 3.0,
                )
            )
            rows.append(
                _evaluate_method(
                    dataset_name,
                    pair_lit,
                    seed,
                    "OpenLSR-X-3Axis-Fourier",
                    srgan_vols["fused"],
                    hr_test,
                    voxel_um,
                    stack_axis=0,
                    gen_params=_count_params(srgan.gen),
                    train_sec=train_srgan,
                    infer_sec=infer_srgan,
                )
            )

            # 6) Normalization-free RRDB-RaGAN + 3-Axis Fourier Fusion
            pl.seed_everything(seed, workers=True)
            rrdb_ragan = SRGAN(
                learning_rate=2e-4 if fast else 1e-4,
                adaptive_size=4 if fast else 8,
                super_res_factor=scale,
                num_blocks=rrdb_blocks,
                out_channels=out_channels,
                growth_channels=growth_channels,
                backbone="rrdb",
                gan_loss_type="ragan",
                output_activation="clamp",
                pixel_loss_type="l1",
                disc_features=disc_features,
                pretrain_epochs=max(1, epochs // 3),
                use_adversarial_loss=True,
                use_vgg_loss=False,
                output_dir=str(output_dir / "runs" / f"{dataset_name}_s{seed}_rrdb_ragan"),
                save_val_images=False,
            )
            trainer_rrdb = pl.Trainer(
                default_root_dir=str(output_dir / "runs"),
                max_epochs=epochs,
                accelerator="auto",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            t0 = time.perf_counter()
            trainer_rrdb.fit(rrdb_ragan, datamodule=dm)
            train_rrdb = time.perf_counter() - t0

            t0 = time.perf_counter()
            rrdb_vols = super_resolve_volume_orthogonal(
                rrdb_ragan.gen, lr_test, scale_factor=scale, fusion="fourier"
            )
            infer_rrdb = time.perf_counter() - t0
            rows.append(
                _evaluate_method(
                    dataset_name,
                    pair_lit,
                    seed,
                    "OpenLSR-X-RRDB-RaGAN-3Axis-Fourier",
                    rrdb_vols["fused"],
                    hr_test,
                    voxel_um,
                    stack_axis=0,
                    gen_params=_count_params(rrdb_ragan.gen),
                    train_sec=train_rrdb,
                    infer_sec=infer_rrdb,
                )
            )

            if seed == seed_list[0]:
                export_for_openimpala(
                    rrdb_vols["fused"],
                    output_dir=output_dir / "openimpala_ready",
                    name=f"{dataset_name}_rrdb_ragan_s{seed}",
                    pore_is_low=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    csv_path = output_dir / "zn13x_synchrotron_results.csv"
    json_path = output_dir / "zn13x_synchrotron_results.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"[zn13x_study] Saved {len(rows)} benchmark rows to {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Diamond I13-2 Zn-13X Synchrotron SR Study.")
    parser.add_argument("--output-dir", default="results/zn13x_study", help="Output directory.")
    parser.add_argument("--data-dir", default="data/benchmarks", help="Benchmark cache directory.")
    parser.add_argument(
        "--pairs",
        nargs="+",
        choices=["2x", "8x"],
        default=None,
        help="I13-2 magnification pairs to evaluate.",
    )
    parser.add_argument("--lr-size", type=int, default=48, help="LR cubic side length.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds.")
    parser.add_argument("--raw-h5-dir", type=str, default=None, help="Optional raw I13-2 HDF5 dir.")
    parser.add_argument("--fast", action="store_true", help="Rapid CPU smoke benchmark.")
    ns = parser.parse_args()

    run_zn13x_study(
        output_dir=Path(ns.output_dir),
        data_dir=Path(ns.data_dir),
        pairs=ns.pairs,
        lr_size=ns.lr_size,
        epochs=ns.epochs,
        seeds=ns.seeds,
        raw_h5_dir=Path(ns.raw_h5_dir) if ns.raw_h5_dir else None,
        fast=ns.fast,
    )


if __name__ == "__main__":
    main()
