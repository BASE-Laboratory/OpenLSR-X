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
from typing import Any, Literal

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from openlsrx.baselines import upsample_3d
from openlsrx.data import (
    PairedVolumeDataModule,
    prepare_zn13x_benchmark,
    synthesize_zn13x_phantom_pair,
)
from openlsrx.inference import fuse_orthogonal_volumes, super_resolve_volume_orthogonal
from openlsrx.inference.orthogonal import AxisName
from openlsrx.io import export_for_openimpala
from openlsrx.metrics import (
    compute_tortuosity_uncertainty_bounds,
    evaluate_image_metrics,
    evaluate_petrophysical_metrics,
    evaluate_spectral_metrics,
    evaluate_transport_metrics,
    evaluate_triphase_zn13x_metrics,
    precompute_reference_transport,
)
from openlsrx.models import (
    SRGAN,
    SR3DResNet,
    build_3d_subcube_dataloaders,
    super_resolve_volume_3d,
)


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _fit_with_best_checkpoint(
    model: pl.LightningModule,
    root_dir: Path,
    run_tag: str,
    epochs: int,
    fast: bool,
    monitor: str = "val/psnr",
    **fit_kwargs: Any,
) -> float:
    """Fit ``model`` and restore the best validation PSNR checkpoint in full runs."""
    callbacks: list[pl.Callback] = []
    ckpt_cb: ModelCheckpoint | None = None
    if not fast:
        ckpt_dir = root_dir / "checkpoints" / run_tag
        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor=monitor,
            mode="max",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(ckpt_cb)

    trainer = pl.Trainer(
        default_root_dir=str(root_dir),
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=(not fast),
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    t0 = time.perf_counter()
    trainer.fit(model, **fit_kwargs)
    elapsed = time.perf_counter() - t0

    if ckpt_cb is not None and ckpt_cb.best_model_path:
        ckpt = torch.load(ckpt_cb.best_model_path, map_location=model.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
    return elapsed


def _evaluate_method(
    dataset_name: str,
    pair: str,
    seed: int,
    method: str,
    pred_vol: np.ndarray,
    hr_test: np.ndarray,
    voxel_size_um: float,
    source_kind: str = "unknown",
    is_experimental_xct: bool = False,
    precomputed_true: dict[str, Any] | None = None,
    stack_axis: int = 0,
    gen_params: int = 0,
    train_sec: float = 0.0,
    infer_sec: float = 0.0,
    uncertainty_std: np.ndarray | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset_name,
        "pair": pair,
        "source_kind": source_kind,
        "is_experimental_xct": int(bool(is_experimental_xct)),
        "seed": seed,
        "method": method,
        "gen_params": gen_params,
        "train_sec": round(train_sec, 4),
        "infer_sec": round(infer_sec, 4),
    }
    row.update(evaluate_image_metrics(pred_vol, hr_test))
    row.update(evaluate_spectral_metrics(pred_vol, hr_test, voxel_size_um=voxel_size_um))
    row.update(
        evaluate_petrophysical_metrics(
            pred_vol,
            hr_test,
            pore_is_low=True,
            voxel_size=voxel_size_um,
            stack_axis=stack_axis,
        )
    )
    row.update(evaluate_triphase_zn13x_metrics(pred_vol, hr_test))
    row.update(
        evaluate_transport_metrics(
            pred_vol,
            hr_test,
            pore_is_low=True,
            voxel_size=voxel_size_um,
            drainage_steps=10,
            precomputed_true=precomputed_true,
        )
    )
    if uncertainty_std is not None:
        unc_bounds = compute_tortuosity_uncertainty_bounds(
            pred_vol,
            uncertainty_std,
            true_vol=hr_test,
            pore_is_low=True,
            precomputed_true=precomputed_true,
        )
        row["mean_epistemic_std"] = unc_bounds["mean_epistemic_std"]
        row["d_eff_halfwidth"] = unc_bounds["d_eff_halfwidth"]
        row["tortuosity_halfwidth"] = unc_bounds["tortuosity_halfwidth"]
        row["uncertainty_error_spearman_r"] = unc_bounds["uncertainty_error_spearman_r"]
    else:
        row["mean_epistemic_std"] = None
        row["d_eff_halfwidth"] = None
        row["tortuosity_halfwidth"] = None
        row["uncertainty_error_spearman_r"] = None
    return row


def run_zn13x_study(
    output_dir: Path,
    data_dir: Path = Path("data/benchmarks"),
    pairs: list[str] | None = None,
    lr_size: int = 48,
    volume_size: int | None = None,
    epochs: int = 120,
    seeds: list[int] | None = None,
    raw_h5_dir: Path | None = None,
    fast: bool = False,
    allow_synthetic_fallback: bool | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_list = pairs or (["2x"] if fast else ["2x", "8x"])
    seed_list = seeds or ([42] if fast else [42, 43, 44])
    fallback = bool(fast) if allow_synthetic_fallback is None else bool(allow_synthetic_fallback)

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
        base_lr = 2e-4
    else:
        # Parameter-balanced regimes (~0.35M-1.15M generator params across 2D/2.5D/3D families)
        srgan_blocks = 8
        rrdb_blocks = 4
        out_channels = 48
        growth_channels = 24
        disc_features = [32, 64, 128, 128]
        c3d_channels = 32
        c3d_blocks = 5
        c3d_samples = 48
        base_lr = 1.5e-4

    rows: list[dict[str, object]] = []
    runs_root = output_dir / "runs"

    for pair in pair_list:
        pair_lit: Literal["2x", "8x"] = "2x" if pair == "2x" else "8x"
        scale = 2 if pair_lit == "2x" else 8
        if fast:
            cur_lr_size = max(12, lr_size // 2) if pair_lit == "8x" else lr_size
        elif volume_size is not None:
            cur_lr_size = max(12, int(volume_size) // scale)
        else:
            cur_lr_size = lr_size
        lr_tif, hr_tif, manifest = prepare_zn13x_benchmark(
            pair=pair_lit,
            dest_dir=data_dir,
            lr_size=cur_lr_size,
            raw_h5_dir=raw_h5_dir,
            allow_offline_fallback=fallback,
            prefer_offline=fast,
            seed=42,
        )
        dataset_name = f"zn13x_{pair_lit}"
        voxel_um = float(manifest["high_resolution_um"])  # type: ignore[arg-type]
        src_kind = str(manifest.get("source_kind", "unknown"))
        is_exp = bool(manifest.get("is_experimental_xct", False))

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

            # Solve ground-truth HR 3D Fickian diffusion & drainage ONCE per seed
            ref_transport = precompute_reference_transport(
                hr_test,
                pore_is_low=True,
                voxel_size=voxel_um,
                drainage_steps=10,
            )

            # 1) 3D Tricubic baseline (cell-centered grid_mode=True)
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
                    source_kind=src_kind,
                    is_experimental_xct=is_exp,
                    precomputed_true=ref_transport,
                    infer_sec=infer_tricubic,
                )
            )

            # 2) SRResNet (2D L1 + 3-Axis Fourier, generator-only baseline)
            pl.seed_everything(seed, workers=True)
            srresnet = SRGAN(
                learning_rate=base_lr,
                adaptive_size=4 if fast else 8,
                super_res_factor=scale,
                num_blocks=srgan_blocks,
                out_channels=out_channels,
                pixel_loss_type="l1",
                use_adversarial_loss=False,
                use_vgg_loss=False,
                output_dir=str(runs_root / f"{dataset_name}_s{seed}_srresnet"),
                save_val_images=False,
            )
            train_srres = _fit_with_best_checkpoint(
                srresnet,
                runs_root,
                f"{dataset_name}_s{seed}_srresnet",
                epochs=epochs,
                fast=fast,
                datamodule=dm,
            )
            t0 = time.perf_counter()
            srres_vols = super_resolve_volume_orthogonal(
                srresnet.gen,
                lr_test,
                scale_factor=scale,
                fusion="fourier",
                return_uncertainty=True,
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
                    source_kind=src_kind,
                    is_experimental_xct=is_exp,
                    precomputed_true=ref_transport,
                    gen_params=_count_params(srresnet.gen),
                    train_sec=train_srres,
                    infer_sec=infer_srres,
                    uncertainty_std=srres_vols["uncertainty_std"],
                )
            )

            # 3) Native 3D CNN baseline (SR3DResNet with full O_h 3D augmentation)
            pl.seed_everything(seed, workers=True)
            sr3d = SR3DResNet(
                learning_rate=base_lr,
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
                augment=True,
                seed=seed,
            )
            train_3d = _fit_with_best_checkpoint(
                sr3d,
                runs_root,
                f"{dataset_name}_s{seed}_sr3d",
                epochs=epochs,
                fast=fast,
                train_dataloaders=tr_3d_loader,
                val_dataloaders=va_3d_loader,
            )
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
                    source_kind=src_kind,
                    is_experimental_xct=is_exp,
                    precomputed_true=ref_transport,
                    gen_params=_count_params(sr3d.gen),
                    train_sec=train_3d,
                    infer_sec=infer_3d,
                )
            )

            # 4 & 5) Classical OpenLSR-X SRGAN (Single-Axis yz vs 3-Axis Fourier)
            pl.seed_everything(seed, workers=True)
            srgan = SRGAN(
                learning_rate=base_lr,
                adaptive_size=4 if fast else 8,
                super_res_factor=scale,
                num_blocks=srgan_blocks,
                out_channels=out_channels,
                disc_features=disc_features,
                pretrain_epochs=max(1, epochs // 3),
                use_adversarial_loss=True,
                use_vgg_loss=False,
                output_dir=str(runs_root / f"{dataset_name}_s{seed}_srgan"),
                save_val_images=False,
            )
            train_srgan = _fit_with_best_checkpoint(
                srgan,
                runs_root,
                f"{dataset_name}_s{seed}_srgan",
                epochs=epochs,
                fast=fast,
                datamodule=dm,
            )
            t0 = time.perf_counter()
            srgan_vols = super_resolve_volume_orthogonal(
                srgan.gen,
                lr_test,
                scale_factor=scale,
                fusion="fourier",
                return_uncertainty=True,
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
                    source_kind=src_kind,
                    is_experimental_xct=is_exp,
                    precomputed_true=ref_transport,
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
                    source_kind=src_kind,
                    is_experimental_xct=is_exp,
                    precomputed_true=ref_transport,
                    stack_axis=0,
                    gen_params=_count_params(srgan.gen),
                    train_sec=train_srgan,
                    infer_sec=infer_srgan,
                    uncertainty_std=srgan_vols["uncertainty_std"],
                )
            )

            # 6) Normalization-free RRDB-RaGAN + 3-Axis Fourier Fusion
            pl.seed_everything(seed, workers=True)
            rrdb_ragan = SRGAN(
                learning_rate=base_lr,
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
                output_dir=str(runs_root / f"{dataset_name}_s{seed}_rrdb_ragan"),
                save_val_images=False,
            )
            train_rrdb = _fit_with_best_checkpoint(
                rrdb_ragan,
                runs_root,
                f"{dataset_name}_s{seed}_rrdb_ragan",
                epochs=epochs,
                fast=fast,
                datamodule=dm,
            )
            t0 = time.perf_counter()
            rrdb_vols = super_resolve_volume_orthogonal(
                rrdb_ragan.gen,
                lr_test,
                scale_factor=scale,
                fusion="fourier",
                return_uncertainty=True,
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
                    source_kind=src_kind,
                    is_experimental_xct=is_exp,
                    precomputed_true=ref_transport,
                    stack_axis=0,
                    gen_params=_count_params(rrdb_ragan.gen),
                    train_sec=train_rrdb,
                    infer_sec=infer_rrdb,
                    uncertainty_std=rrdb_vols["uncertainty_std"],
                )
            )

            # Shared multi-axis 2.5D DataModule for controlled SwinIR stepwise ablation ladder
            dm_25d = PairedVolumeDataModule(
                low_path=str(lr_tif),
                high_path=str(hr_tif),
                scale_factor=scale,
                context_slices=3,
                train_axes=("yz", "xz", "xy"),
                norm_mode="none",
                train_frac=0.65,
                val_frac=0.20,
                guard_slices_lr=1,
                patch_size_lr=min(16, cur_lr_size),
                batch_size=4 if scale == 2 else 2,
                eval_batch_size=4 if scale == 2 else 2,
                seed=seed,
            )

            # 7) Stepwise Ablation A: SwinIR-2.5D (Pure L1 + Standard Fourier; isolates 2.5D SwinIR backbone)
            pl.seed_everything(seed, workers=True)
            swin_l1 = SRGAN(
                learning_rate=base_lr,
                adaptive_size=4 if fast else 8,
                super_res_factor=scale,
                num_blocks=2 if fast else 4,
                out_channels=16 if fast else 48,
                context_slices=3,
                backbone="swinir",
                swin_window_size=4 if fast else 8,
                swin_num_heads=4,
                output_activation="clamp",
                pixel_loss_type="l1",
                use_adversarial_loss=False,
                use_vgg_loss=False,
                output_dir=str(runs_root / f"{dataset_name}_s{seed}_swinir_25d_l1"),
                save_val_images=False,
            )
            train_swin_l1 = _fit_with_best_checkpoint(
                swin_l1,
                runs_root,
                f"{dataset_name}_s{seed}_swinir_25d_l1",
                epochs=epochs,
                fast=fast,
                datamodule=dm_25d,
            )
            t0 = time.perf_counter()
            swin_l1_vols = super_resolve_volume_orthogonal(
                swin_l1.gen,
                lr_test,
                scale_factor=scale,
                fusion="fourier",
                return_uncertainty=True,
            )
            infer_swin_l1 = time.perf_counter() - t0
            rows.append(
                _evaluate_method(
                    dataset_name,
                    pair_lit,
                    seed,
                    "OpenLSR-X-SwinIR-2.5D-L1-Fourier",
                    swin_l1_vols["fused"],
                    hr_test,
                    voxel_um,
                    source_kind=src_kind,
                    is_experimental_xct=is_exp,
                    precomputed_true=ref_transport,
                    stack_axis=0,
                    gen_params=_count_params(swin_l1.gen),
                    train_sec=train_swin_l1,
                    infer_sec=infer_swin_l1,
                    uncertainty_std=swin_l1_vols["uncertainty_std"],
                )
            )

            # 8 & 9) Stepwise Ablations B & C: + Physics/Topology Supervision (Fourier vs Confidence-Fourier)
            pl.seed_everything(seed, workers=True)
            swin_phys = SRGAN(
                learning_rate=base_lr,
                adaptive_size=4 if fast else 8,
                super_res_factor=scale,
                num_blocks=2 if fast else 4,
                out_channels=16 if fast else 48,
                context_slices=3,
                backbone="swinir",
                swin_window_size=4 if fast else 8,
                swin_num_heads=4,
                output_activation="clamp",
                pixel_loss_type="l1",
                use_adversarial_loss=False,
                use_vgg_loss=False,
                conservation_loss_weight=0.25,
                conservation_psf_sigma=0.5,
                topology_loss_weight=0.05,
                gradient_loss_weight=0.10,
                ffl_loss_weight=0.05,
                output_dir=str(runs_root / f"{dataset_name}_s{seed}_swinir_25d_phys"),
                save_val_images=False,
            )
            train_swin = _fit_with_best_checkpoint(
                swin_phys,
                runs_root,
                f"{dataset_name}_s{seed}_swinir_25d_phys",
                epochs=epochs,
                fast=fast,
                datamodule=dm_25d,
            )
            t0 = time.perf_counter()
            swin_vols = super_resolve_volume_orthogonal(
                swin_phys.gen,
                lr_test,
                scale_factor=scale,
                fusion="confidence_fourier",
                return_uncertainty=True,
            )
            infer_swin = time.perf_counter() - t0

            # Fuse the exact same 3 orthogonal sweeps under standard Fourier to isolate ConfFourier cleanly
            t0_four = time.perf_counter()
            per_axis_swin: dict[AxisName, np.ndarray] = {
                "yz": swin_vols["yz"],
                "xz": swin_vols["xz"],
                "xy": swin_vols["xy"],
            }
            swin_phys_fourier = fuse_orthogonal_volumes(per_axis_swin, mode="fourier")
            infer_swin_fourier = infer_swin + (time.perf_counter() - t0_four)

            rows.append(
                _evaluate_method(
                    dataset_name,
                    pair_lit,
                    seed,
                    "OpenLSR-X-SwinIR-2.5D-Physics-Fourier",
                    swin_phys_fourier,
                    hr_test,
                    voxel_um,
                    source_kind=src_kind,
                    is_experimental_xct=is_exp,
                    precomputed_true=ref_transport,
                    stack_axis=0,
                    gen_params=_count_params(swin_phys.gen),
                    train_sec=train_swin,
                    infer_sec=infer_swin_fourier,
                    uncertainty_std=swin_vols["uncertainty_std"],
                )
            )
            rows.append(
                _evaluate_method(
                    dataset_name,
                    pair_lit,
                    seed,
                    "OpenLSR-X-SwinIR-2.5D-Physics-ConfFourier",
                    swin_vols["fused"],
                    hr_test,
                    voxel_um,
                    source_kind=src_kind,
                    is_experimental_xct=is_exp,
                    precomputed_true=ref_transport,
                    stack_axis=0,
                    gen_params=_count_params(swin_phys.gen),
                    train_sec=train_swin,
                    infer_sec=infer_swin,
                    uncertainty_std=swin_vols["uncertainty_std"],
                )
            )

            if seed == seed_list[0]:
                export_for_openimpala(
                    rrdb_vols["fused"],
                    output_dir=output_dir / "openimpala_ready",
                    name=f"{dataset_name}_rrdb_ragan_s{seed}",
                    pore_is_low=True,
                )
                export_for_openimpala(
                    swin_vols["fused"],
                    output_dir=output_dir / "openimpala_ready",
                    name=f"{dataset_name}_swinir_25d_phys_s{seed}",
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
    parser.add_argument(
        "--lr-size",
        type=int,
        default=48,
        help="LR cubic side length (default: 48).",
    )
    parser.add_argument(
        "--volume-size",
        type=int,
        default=None,
        help="Optional target HR cubic side length (overrides --lr-size as volume_size // scale).",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds.")
    parser.add_argument("--raw-h5-dir", type=str, default=None, help="Optional raw I13-2 HDF5 dir.")
    parser.add_argument("--fast", action="store_true", help="Rapid CPU smoke benchmark.")
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow synthetic fallback if live Zenodo/HDF5 loading fails during a full campaign.",
    )
    ns = parser.parse_args()

    run_zn13x_study(
        output_dir=Path(ns.output_dir),
        data_dir=Path(ns.data_dir),
        pairs=ns.pairs,
        lr_size=ns.lr_size,
        volume_size=ns.volume_size,
        epochs=ns.epochs,
        seeds=ns.seeds,
        raw_h5_dir=Path(ns.raw_h5_dir) if ns.raw_h5_dir else None,
        fast=ns.fast,
        allow_synthetic_fallback=(True if ns.allow_synthetic_fallback else None),
    )


if __name__ == "__main__":
    main()
