"""Unified command-line interface for OpenLSR-X training, inference, and XCT evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from .data import (
    AVAILABLE_PUBLIC_DATASETS,
    PairedVolumeDataModule,  # noqa: F401 - registered for LightningCLI subclass discovery
    SyntheticPoreSpyDataModule,  # noqa: F401 - registered for LightningCLI subclass discovery
    fetch_or_generate_rock_sample,
    load_volume_any,
)
from .metrics import evaluate_image_metrics, evaluate_petrophysical_metrics
from .models import SRGAN


def _run_evaluate(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="openlsrx evaluate",
        description="Compute 2D/3D image-fidelity and porous-media petrophysical metrics on XCT volumes.",
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
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output file path.")
    ns = parser.parse_args(list(argv))

    pred_vol = load_volume_any(ns.pred)
    target_vol = load_volume_any(ns.target)
    if pred_vol.shape != target_vol.shape:
        raise ValueError(
            f"Shape mismatch between --pred {pred_vol.shape} and --target {target_vol.shape}."
        )

    img_metrics = evaluate_image_metrics(pred_vol, target_vol)
    petro_metrics = evaluate_petrophysical_metrics(
        pred_vol,
        target_vol,
        threshold_method=ns.threshold,
        fixed_threshold=ns.fixed_threshold,
        pore_is_low=(not ns.pore_is_high),
        voxel_size=ns.voxel_size,
        stack_axis=ns.stack_axis,
    )
    combined = {"image_metrics": img_metrics, "petrophysical_metrics": petro_metrics}
    formatted = json.dumps(combined, indent=2)
    print(formatted)
    if ns.output:
        out_path = Path(ns.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(formatted + "\n", encoding="utf-8")
    return 0


def _run_download_benchmarks(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="openlsrx download-benchmarks",
        description="Fetch open micro-CT rock benchmark volumes (with offline digital-rock fallback).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(AVAILABLE_PUBLIC_DATASETS.keys()),
        choices=sorted(AVAILABLE_PUBLIC_DATASETS.keys()),
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
        help="Cubic HR volume side length (default: 128).",
    )
    parser.add_argument(
        "--no-offline-fallback",
        action="store_true",
        help="Fail instead of synthesizing calibrated digital-rock phantoms if offline.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    ns = parser.parse_args(list(argv))

    shape = (ns.size, ns.size, ns.size)
    for name in ns.datasets:
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
