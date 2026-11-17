"""Unified command-line interface for OpenLSR-X training, inference, and XCT evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from .data import (
    AVAILABLE_PUBLIC_DATASETS,
    PairedVolumeDataModule,  # noqa: F401 - registered for LightningCLI subclass discovery
    SyntheticPoreSpyDataModule,  # noqa: F401 - registered for LightningCLI subclass discovery
    fetch_or_generate_rock_sample,
    load_volume_any,
    prepare_zn13x_benchmark,
)
from .io import export_for_openimpala
from .metrics import (
    evaluate_image_metrics,
    evaluate_petrophysical_metrics,
    evaluate_transport_metrics,
)
from .models import SRGAN

ZN13X_BENCHMARK_KEYS = ("zn13x_2x", "zn13x_8x")


def _run_evaluate(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="openlsrx evaluate",
        description="Compute 2D/3D image-fidelity, petrophysical, and transport metrics on XCT volumes.",
    )
    parser.add_argument(
        "--pred", required=True, help="Path to super-resolved TIFF/HDF5/NPY volume."
    )
    parser.add_argument(
        "--target", required=True, help="Path to ground-truth HR TIFF/HDF5/NPY volume."
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
    parser.add_argument("--voxel-size", type=float, default=1.0, help="Isotropic voxel spacing.")
    parser.add_argument(
        "--stack-axis", type=int, default=0, help="Stacking axis for anisotropy check."
    )
    parser.add_argument(
        "--include-transport",
        action="store_true",
        help="Also solve 3D Fickian diffusive tortuosity (D_eff/D_0) and capillary drainage curves.",
    )
    parser.add_argument(
        "--export-openimpala-dir",
        type=str,
        default=None,
        help="Optional directory to export segmented 8-bit TIFF + AMReX .inputs for OpenImpala.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output file path.")
    ns = parser.parse_args(list(argv))

    pred_vol = load_volume_any(ns.pred)
    target_vol = load_volume_any(ns.target)
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
        help="Optional local directory containing raw Diamond I13-2 16906*_recon.h5 files.",
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
                seed=ns.seed,
            )
            print(
                f"[{name}] ready at {tif_path} "
                f"(source={manifest['source_kind']}, sha256={str(manifest['sha256'])[:12]}...)"
            )
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

    OpenLSRXLightningCLI(
        model_class=SRGAN,
        datamodule_class=pl.LightningDataModule,
        subclass_mode_data=True,
        args=raw_args if args is not None else None,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
