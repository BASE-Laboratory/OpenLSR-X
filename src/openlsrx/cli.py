"""Unified command-line interface for OpenLSR-X training, inference, and XCT evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytorch_lightning as pl
import tifffile
import torch
from pytorch_lightning.cli import LightningCLI

from .data import (
    AVAILABLE_PUBLIC_DATASETS,
    PairedVolumeDataModule,  # noqa: F401 - registered for LightningCLI subclass discovery
    SyntheticPoreSpyDataModule,  # noqa: F401 - registered for LightningCLI subclass discovery
    build_circular_fov_mask,
    fetch_or_generate_rock_sample,
    load_volume_any,
    normalize_volume,
    prepare_zn13x_benchmark,
)
from .inference import super_resolve_volume_orthogonal
from .inference.orthogonal import AxisName, FusionMode
from .io import export_for_openimpala
from .metrics import (
    evaluate_image_metrics,
    evaluate_petrophysical_metrics,
    evaluate_spectral_metrics,
    evaluate_transport_metrics,
    evaluate_triphase_zn13x_metrics,
)
from .models import SRGAN

ZN13X_BENCHMARK_KEYS = ("zn13x_2x", "zn13x_8x")


def _run_evaluate(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="openlsrx evaluate",
        description="Compute 2D/3D image-fidelity, spectral FSC, petrophysical, and transport metrics on XCT volumes.",
    )
    parser.add_argument(
        "--pred", required=True, help="Path to super-resolved TIFF/HDF5/NPY volume or 2D slice."
    )
    parser.add_argument(
        "--target", required=True, help="Path to ground-truth HR TIFF/HDF5/NPY volume or 2D slice."
    )
    parser.add_argument(
        "--threshold",
        choices=["otsu", "yen", "fixed"],
        default="otsu",
        help="Segmentation thresholding method (default: otsu).",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Threshold value when --threshold=fixed.",
    )
    parser.add_argument(
        "--pore-is-high",
        action="store_true",
        help="Set if pores have higher greyscale intensity than solid (e.g., local-thickness maps).",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=1.0, help="Isotropic voxel spacing (um)."
    )
    parser.add_argument(
        "--stack-axis", type=int, default=0, help="Stacking axis for anisotropy check."
    )
    parser.add_argument(
        "--include-spectral",
        action="store_true",
        help="Also compute 2D FRC / 3D FSC reference crossover (FSC=0.5) and spectral AUC.",
    )
    parser.add_argument(
        "--include-triphase",
        action="store_true",
        help="Also evaluate three-phase (Pore / Na-13X Core / Zn-13X Rim) chemical segmentation.",
    )
    parser.add_argument(
        "--include-transport",
        action="store_true",
        help="Also solve 3D Fickian diffusive tortuosity (D_eff/D_0) and capillary drainage curves.",
    )
    parser.add_argument(
        "--export-openimpala-dir",
        "--export-openimpala",
        dest="export_openimpala_dir",
        type=str,
        default=None,
        help="Optional directory to export segmented 8-bit TIFF + AMReX .inputs for OpenImpala.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output file path.")
    ns = parser.parse_args(list(argv))

    pred_vol = load_volume_any(ns.pred, preserve_2d=True)
    target_vol = load_volume_any(ns.target, preserve_2d=True)
    if pred_vol.shape != target_vol.shape:
        raise ValueError(
            f"Shape mismatch between --pred {pred_vol.shape} and --target {target_vol.shape}."
        )

    pore_is_low = not ns.pore_is_high
    img_metrics = evaluate_image_metrics(pred_vol, target_vol)
    petro_metrics = evaluate_petrophysical_metrics(
        pred_vol,
        target_vol,
        threshold_method=ns.threshold,
        fixed_threshold=ns.fixed_threshold,
        pore_is_low=pore_is_low,
        voxel_size=ns.voxel_size,
        stack_axis=ns.stack_axis,
    )
    combined: dict[str, Any] = {
        "image_metrics": img_metrics,
        "petrophysical_metrics": petro_metrics,
    }
    if ns.include_spectral:
        combined["spectral_metrics"] = evaluate_spectral_metrics(
            pred_vol, target_vol, voxel_size_um=ns.voxel_size
        )
    if ns.include_triphase:
        combined["triphase_zn13x_metrics"] = evaluate_triphase_zn13x_metrics(pred_vol, target_vol)
    if ns.include_transport and pred_vol.ndim == 3:
        combined["transport_metrics"] = evaluate_transport_metrics(
            pred_vol,
            target_vol,
            threshold_method=ns.threshold,
            fixed_threshold=ns.fixed_threshold,
            pore_is_low=pore_is_low,
            voxel_size=ns.voxel_size,
        )
    if ns.export_openimpala_dir and pred_vol.ndim == 3:
        tif_p, inp_p = export_for_openimpala(
            pred_vol,
            output_dir=ns.export_openimpala_dir,
            name=Path(ns.pred).stem,
            threshold_method="fixed",
            fixed_threshold=float(petro_metrics["threshold_true"]),
            pore_is_low=pore_is_low,
        )
        combined["openimpala_export"] = {
            "tiff_path": str(tif_p),
            "inputs_path": str(inp_p),
        }

    formatted = json.dumps(combined, indent=2)
    print(formatted)
    if ns.output:
        out_path = Path(ns.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(formatted + "\n", encoding="utf-8")
    return 0


def _run_export_openimpala(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="openlsrx export-openimpala",
        description="Export a 3D XCT volume as an uncompressed 8-bit TIFF + AMReX .inputs deck for OpenImpala.",
    )
    parser.add_argument("--volume", required=True, help="Path to 3D TIFF/HDF5/NPY volume.")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for OpenImpala files."
    )
    parser.add_argument("--name", default=None, help="Sample base name (defaults to volume stem).")
    parser.add_argument(
        "--threshold",
        choices=["otsu", "yen", "fixed"],
        default="otsu",
        help="Thresholding algorithm for greyscale inputs (default: otsu).",
    )
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--pore-is-high", action="store_true")
    parser.add_argument("--direction", choices=["x", "y", "z"], default="x")
    ns = parser.parse_args(list(argv))

    vol = load_volume_any(ns.volume)
    sample_name = ns.name or Path(ns.volume).stem
    tif_path, inputs_path = export_for_openimpala(
        vol,
        output_dir=ns.output_dir,
        name=sample_name,
        threshold_method=ns.threshold,
        fixed_threshold=ns.fixed_threshold,
        pore_is_low=(not ns.pore_is_high),
        direction=ns.direction,
    )
    print(f"[OpenImpala] geometry: {tif_path} | inputs: {inputs_path}")
    return 0


def _run_download_benchmarks(argv: Sequence[str]) -> int:
    all_choices = sorted([*AVAILABLE_PUBLIC_DATASETS.keys(), *ZN13X_BENCHMARK_KEYS])
    parser = argparse.ArgumentParser(
        prog="openlsrx download-benchmarks",
        description="Fetch open micro-CT rock & Diamond I13-2 Zn-13X synchrotron benchmark volumes.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(AVAILABLE_PUBLIC_DATASETS.keys()),
        choices=all_choices,
        help="Benchmark datasets to prepare.",
    )
    parser.add_argument(
        "--dest",
        "--output-dir",
        dest="dest",
        default="data/benchmarks",
        help="Destination directory (default: data/benchmarks).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Cubic HR side length for rock benchmarks, or LR side length for zn13x_* if <= 64 (default: 128).",
    )
    parser.add_argument(
        "--raw-h5-dir",
        type=str,
        default=None,
        help="Optional local directory containing raw Diamond I13-2 16906*_recon.h5 or tomo_p4_tomopy_recon.h5 files.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Synthesize deterministic calibrated digital-rock/synchrotron phantoms offline without network fetches.",
    )
    parser.add_argument(
        "--no-offline-fallback",
        action="store_true",
        help="Fail instead of synthesizing calibrated digital-rock/synchrotron phantoms if offline.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    ns = parser.parse_args(list(argv))

    shape = (ns.size, ns.size, ns.size)
    for name in ns.datasets:
        if name in ZN13X_BENCHMARK_KEYS:
            pair: Literal["2x", "8x"] = "2x" if name == "zn13x_2x" else "8x"
            scale = 2 if pair == "2x" else 8
            lr_size = max(16, ns.size // scale) if ns.size >= 64 else max(16, ns.size)
            lr_tif, hr_tif, manifest = prepare_zn13x_benchmark(
                pair=pair,
                dest_dir=ns.dest,
                lr_size=lr_size,
                raw_h5_dir=ns.raw_h5_dir,
                allow_offline_fallback=(not ns.no_offline_fallback),
                prefer_offline=bool(ns.offline),
                seed=ns.seed,
            )
            print(
                f"[{name}] ready at {lr_tif} & {hr_tif} "
                f"(source={manifest['source_kind']}, hr_sha256={str(manifest['hr_sha256'])[:12]}...)"
            )
        else:
            tif_path, manifest = fetch_or_generate_rock_sample(
                dataset_name=name,
                dest_dir=ns.dest,
                shape=shape,
                allow_offline_fallback=(not ns.no_offline_fallback),
                prefer_offline=bool(ns.offline),
                seed=ns.seed,
            )
            print(
                f"[{name}] ready at {tif_path} "
                f"(source={manifest['source_kind']}, sha256={str(manifest['sha256'])[:12]}...)"
            )
    return 0


def _run_download_zn13x(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="openlsrx download-zn13x",
        description="Prepare Diamond Light Source I13-2 multi-magnification Zn-13X paired synchrotron volumes.",
    )
    parser.add_argument(
        "--pair",
        choices=["2x", "8x", "all"],
        default="all",
        help="Magnification pair(s) to prepare: 2x, 8x, or all (default: all).",
    )
    parser.add_argument(
        "--dest",
        "--output-dir",
        dest="dest",
        default="data/benchmarks",
        help="Destination directory (default: data/benchmarks).",
    )
    parser.add_argument(
        "--lr-size",
        type=int,
        default=32,
        help="Cubic LR side length in voxels (default: 32).",
    )
    parser.add_argument("--raw-h5-dir", type=str, default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Synthesize deterministic calibrated Zn-13X phantoms offline without network fetches.",
    )
    parser.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Allow calibrated synthetic Zn-13X fallback when offline (enabled by default unless --no-offline-fallback).",
    )
    parser.add_argument("--no-offline-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    ns = parser.parse_args(list(argv))

    pairs: list[Literal["2x", "8x"]] = (
        ["2x", "8x"] if ns.pair == "all" else [cast(Literal["2x", "8x"], ns.pair)]
    )
    allow_fb = bool(ns.synthetic_fallback) or (not ns.no_offline_fallback)
    for pair in pairs:
        lr_tif, hr_tif, manifest = prepare_zn13x_benchmark(
            pair=pair,
            dest_dir=ns.dest,
            lr_size=ns.lr_size,
            raw_h5_dir=ns.raw_h5_dir,
            allow_offline_fallback=allow_fb,
            prefer_offline=bool(ns.offline),
            seed=ns.seed,
        )
        print(
            f"[zn13x_{pair}] ready at {lr_tif} & {hr_tif} "
            f"(source={manifest['source_kind']}, hr_sha256={str(manifest['hr_sha256'])[:12]}...)"
        )
    return 0


def _run_predict_volume(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="openlsrx predict-volume",
        description="Run 3-axis orthogonal 2D/2.5D-to-3D XCT super-resolution and epistemic uncertainty export.",
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        dest="checkpoint",
        required=True,
        help="Path to trained SRGAN/RRDB/SwinIR .ckpt file.",
    )
    parser.add_argument("--input", required=True, help="Path to 3D LR TIFF/HDF5/NPY volume.")
    parser.add_argument(
        "--output", required=True, help="Output path for super-resolved 3D BigTIFF volume."
    )
    parser.add_argument(
        "--uncertainty-output",
        default=None,
        help="Optional output path for 3-axis epistemic uncertainty std map (.tiff).",
    )
    parser.add_argument(
        "--fusion",
        choices=["mean", "median", "fourier", "confidence", "confidence_fourier"],
        default="confidence_fourier",
        help="Multi-axis volumetric fusion operator (default: confidence_fourier).",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        choices=["yz", "xz", "xy"],
        default=["yz", "xz", "xy"],
        help="Orthogonal sweep axes (default: yz xz xy).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Slice inference batch size.")
    parser.add_argument(
        "--norm-mode",
        choices=["percentile", "minmax", "none", "fixed"],
        default="percentile",
        help="Input LR intensity normalization mode (default: percentile).",
    )
    parser.add_argument(
        "--vmin", type=float, default=None, help="Lower bound when --norm-mode=fixed."
    )
    parser.add_argument(
        "--vmax", type=float, default=None, help="Upper bound when --norm-mode=fixed."
    )
    parser.add_argument("--circular-fov-mask", action="store_true")
    parser.add_argument("--h5-dataset-key", default="4-TomopyRecon-tomo/data")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (e.g., 'cpu' or 'cuda:0'; auto-detected by default).",
    )
    ns = parser.parse_args(list(argv))

    dev = (
        torch.device(ns.device)
        if ns.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = SRGAN.load_from_checkpoint(ns.checkpoint, map_location=dev)
    model.eval().to(dev)

    raw_lr = load_volume_any(ns.input, h5_dataset_key=ns.h5_dataset_key)
    fov = build_circular_fov_mask(raw_lr.shape) if ns.circular_fov_mask else None
    lr_norm = normalize_volume(
        raw_lr,
        mode=ns.norm_mode,
        vmin=ns.vmin,
        vmax=ns.vmax,
        mask=fov,
    )

    axes = [cast(AxisName, ax) for ax in ns.axes]
    ortho = super_resolve_volume_orthogonal(
        generator=model.gen,
        low_volume=lr_norm,
        scale_factor=model.super_res_factor,
        axes=axes,
        fusion=cast(FusionMode, ns.fusion),
        batch_size=ns.batch_size,
        device=dev,
        context_slices=model.context_slices,
        return_uncertainty=(ns.uncertainty_output is not None),
    )

    out_path = Path(ns.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fused = np.ascontiguousarray(ortho["fused"], dtype=np.float32)
    tifffile.imwrite(out_path, fused, bigtiff=True, photometric="minisblack")

    if ns.uncertainty_output is not None and "uncertainty_std" in ortho:
        unc_path = Path(ns.uncertainty_output)
        unc_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            unc_path,
            np.ascontiguousarray(ortho["uncertainty_std"], dtype=np.float32),
            bigtiff=True,
            photometric="minisblack",
        )
    print(f"[predict-volume] fused shape={fused.shape} saved to {out_path}")
    return 0


class OpenLSRXLightningCLI(LightningCLI):
    """LightningCLI binding ``SRGAN`` to any ``LightningDataModule`` subclass via YAML ``class_path``."""


def main(args: Sequence[str] | None = None) -> None:
    """Entry point for ``openlsrx`` and ``python -m openlsrx``."""
    raw_args = list(sys.argv[1:] if args is None else args)
    if raw_args and raw_args[0] == "evaluate":
        raise SystemExit(_run_evaluate(raw_args[1:]))
    if raw_args and raw_args[0] == "export-openimpala":
        raise SystemExit(_run_export_openimpala(raw_args[1:]))
    if raw_args and raw_args[0] == "download-benchmarks":
        raise SystemExit(_run_download_benchmarks(raw_args[1:]))
    if raw_args and raw_args[0] == "download-zn13x":
        raise SystemExit(_run_download_zn13x(raw_args[1:]))
    if raw_args and raw_args[0] == "predict-volume":
        raise SystemExit(_run_predict_volume(raw_args[1:]))

    OpenLSRXLightningCLI(
        model_class=SRGAN,
        datamodule_class=pl.LightningDataModule,
        subclass_mode_data=True,
        args=raw_args if args is not None else None,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
