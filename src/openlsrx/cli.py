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
from basecoat import Volume, read_ome_zarr, write_ome_zarr
from pytorch_lightning.cli import LightningCLI

from .data import (
    AVAILABLE_DUALSCAN_DATASETS,
    AVAILABLE_PUBLIC_DATASETS,
    NormMode,
    PairedVolumeDataModule,  # noqa: F401 - registered for LightningCLI subclass discovery
    SyntheticPoreSpyDataModule,  # noqa: F401 - registered for LightningCLI subclass discovery
    apply_cross_scan_calibration,
    build_circular_fov_mask,
    fetch_or_generate_rock_sample,
    load_volume_any,
    normalize_volume,
    prepare_bentheimer_dualscan_benchmark,
    prepare_zn13x_benchmark,
)
from .inference import super_resolve_volume_orthogonal
from .inference.orthogonal import AxisName, FusionMode
from .io import export_for_openimpala, export_volume_ome_zarr
from .metrics import (
    evaluate_image_metrics,
    evaluate_petrophysical_metrics,
    evaluate_spectral_metrics,
    evaluate_transport_metrics,
    evaluate_triphase_zn13x_metrics,
)
from .models import SRGAN, SR3DResNet, super_resolve_volume_3d

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
    all_choices = sorted(
        [
            *AVAILABLE_PUBLIC_DATASETS.keys(),
            *AVAILABLE_DUALSCAN_DATASETS.keys(),
            *ZN13X_BENCHMARK_KEYS,
        ]
    )
    parser = argparse.ArgumentParser(
        prog="openlsrx download-benchmarks",
        description="Fetch open micro-CT rock, physical dual-scan Bentheimer, & Diamond I13-2 Zn-13X benchmark volumes.",
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
        default=None,
        help=(
            "Optional cubic side override. Defaults to 128 HR for single-scan rocks, "
            "75 LR (225 HR) for Bentheimer dual-scan, and 48 LR for zn13x_*."
        ),
    )
    parser.add_argument(
        "--raw-h5-dir",
        type=str,
        default=None,
        help="Optional local directory containing raw Diamond I13-2 16906*_recon.h5 or tomo_p4_tomopy_recon.h5 files.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Optional local directory containing Zenodo 5542624 16-bit .raw dual-scan files.",
    )
    parser.add_argument(
        "--certified-regions",
        type=str,
        default=None,
        help=(
            "Optional path to a MultimodalBenchmark certified_regions.yaml or "
            "SimilarityTransform3D JSON/YAML file for Zn-13X 7-DOF registration."
        ),
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

    rock_side = 128 if ns.size is None else int(ns.size)
    shape = (rock_side, rock_side, rock_side)
    for name in ns.datasets:
        if name in ZN13X_BENCHMARK_KEYS:
            pair: Literal["2x", "8x"] = "2x" if name == "zn13x_2x" else "8x"
            scale = 2 if pair == "2x" else 8
            if ns.size is None:
                lr_size = 48
            else:
                lr_size = max(16, ns.size // scale) if ns.size >= 64 else max(16, ns.size)
            lr_tif, hr_tif, manifest = prepare_zn13x_benchmark(
                pair=pair,
                dest_dir=ns.dest,
                lr_size=lr_size,
                raw_h5_dir=ns.raw_h5_dir,
                certified_regions_path=ns.certified_regions,
                allow_offline_fallback=(not ns.no_offline_fallback),
                prefer_offline=bool(ns.offline),
                seed=ns.seed,
            )
            prov_tag = (
                ", provisional_8bit_preview=True"
                if manifest.get("provisional_8bit_preview")
                else ""
            )
            print(
                f"[{name}] ready at {lr_tif} & {hr_tif} "
                f"(source={manifest['source_kind']}{prov_tag}, hr_sha256={str(manifest['hr_sha256'])[:12]}...)"
            )
        elif name in AVAILABLE_DUALSCAN_DATASETS:
            spec_ds = AVAILABLE_DUALSCAN_DATASETS[name]
            if ns.size is None:
                lr_side = min(spec_ds.low_raw_shape[0], 75)
            else:
                lr_side = (
                    min(spec_ds.low_raw_shape[0], max(12, ns.size))
                    if ns.size <= spec_ds.low_raw_shape[0]
                    else min(spec_ds.low_raw_shape[0], max(12, ns.size // spec_ds.scale_factor))
                )
            lr_tif, hr_tif, manifest = prepare_bentheimer_dualscan_benchmark(
                dataset_name=name,
                dest_dir=ns.dest,
                lr_shape=(lr_side, lr_side, lr_side),
                raw_dir=ns.raw_dir,
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
        default=48,
        help="Cubic LR side length in voxels (default: 48).",
    )
    parser.add_argument("--raw-h5-dir", type=str, default=None)
    parser.add_argument(
        "--certified-regions",
        type=str,
        default=None,
        help=(
            "Optional path to a MultimodalBenchmark certified_regions.yaml or "
            "SimilarityTransform3D JSON/YAML file for 7-DOF registration."
        ),
    )
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
            certified_regions_path=ns.certified_regions,
            allow_offline_fallback=allow_fb,
            prefer_offline=bool(ns.offline),
            seed=ns.seed,
        )
        prov_tag = (
            ", provisional_8bit_preview=True" if manifest.get("provisional_8bit_preview") else ""
        )
        print(
            f"[zn13x_{pair}] ready at {lr_tif} & {hr_tif} "
            f"(source={manifest['source_kind']}{prov_tag}, hr_sha256={str(manifest['hr_sha256'])[:12]}...)"
        )
    return 0


def _detect_checkpoint_model_type(ckpt_path: str, dev: torch.device) -> Literal["srgan", "sr3d"]:
    raw_ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state_dict = raw_ckpt.get("state_dict", {}) if isinstance(raw_ckpt, dict) else {}
    for k in state_dict:
        t = state_dict[k]
        if isinstance(t, torch.Tensor) and t.ndim == 5:
            return "sr3d"
    hparams = raw_ckpt.get("hyper_parameters", {}) if isinstance(raw_ckpt, dict) else {}
    if isinstance(hparams, dict) and "loss_type" in hparams and "backbone" not in hparams:
        return "sr3d"
    return "srgan"


def _run_predict_volume(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="openlsrx predict-volume",
        description="Run 3-axis orthogonal 2D/2.5D-to-3D or sliding-window 3D XCT super-resolution.",
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        dest="checkpoint",
        required=True,
        help="Path to trained SRGAN/RRDB/SwinIR/SR3DResNet .ckpt file.",
    )
    parser.add_argument("--input", required=True, help="Path to 3D LR TIFF/HDF5/NPY volume.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path for super-resolved 3D BigTIFF volume (required unless --output-dir is set).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory (writes <output-dir>/predicted_stack.tiff when --output is omitted).",
    )
    parser.add_argument(
        "--uncertainty-output",
        default=None,
        help="Optional output path for 3-axis orthogonal view-disagreement std map (.tiff).",
    )
    parser.add_argument(
        "--save-uncertainty",
        action="store_true",
        help="Write orthogonal view-disagreement std map (<output-dir>/uncertainty_std_stack.tiff or alongside --output).",
    )
    parser.add_argument(
        "--model-type",
        choices=["auto", "srgan", "sr3d"],
        default="auto",
        help="Checkpoint architecture type (default: auto).",
    )
    parser.add_argument(
        "--fusion",
        choices=["mean", "median", "fourier", "confidence", "confidence_fourier"],
        default="confidence_fourier",
        help="Multi-axis volumetric fusion operator for 2D/2.5D models (default: confidence_fourier).",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        choices=["yz", "xz", "xy"],
        default=["yz", "xz", "xy"],
        help="Orthogonal sweep axes for 2D/2.5D models (default: yz xz xy).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Slice inference batch size.")
    parser.add_argument(
        "--norm-mode",
        choices=["percentile", "minmax", "none", "fixed"],
        default=None,
        help=(
            "Input LR intensity normalization mode override. Defaults to checkpoint train-fitted "
            "bounds when available, otherwise 'percentile'."
        ),
    )
    parser.add_argument(
        "--recompute-input-norm",
        action="store_true",
        help="Ignore checkpoint train-fitted normalization/calibration and normalize --input fresh.",
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
        "--voxel-size-um",
        type=float,
        default=None,
        help="Output HR isotropic voxel size in micrometres (auto-derived from calibrated .ome.zarr LR input when omitted).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (e.g., 'cpu' or 'cuda:0'; auto-detected by default).",
    )
    ns = parser.parse_args(list(argv))

    if ns.output is None and ns.output_dir is None:
        parser.error("Provide either --output <path.tiff> or --output-dir <dir>.")
    out_path = (
        Path(ns.output) if ns.output is not None else (Path(ns.output_dir) / "predicted_stack.tiff")
    )
    unc_path: Path | None = None
    if ns.uncertainty_output is not None:
        unc_path = Path(ns.uncertainty_output)
    elif ns.save_uncertainty:
        unc_dir = Path(ns.output_dir) if ns.output_dir is not None else out_path.parent
        unc_path = unc_dir / "uncertainty_std_stack.tiff"

    dev = (
        torch.device(ns.device)
        if ns.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    resolved_type: Literal["srgan", "sr3d"] = (
        _detect_checkpoint_model_type(ns.checkpoint, dev)
        if ns.model_type == "auto"
        else cast(Literal["srgan", "sr3d"], ns.model_type)
    )
    model_obj: SRGAN | SR3DResNet
    if resolved_type == "sr3d":
        model_obj = SR3DResNet.load_from_checkpoint(ns.checkpoint, map_location=dev)
    else:
        model_obj = SRGAN.load_from_checkpoint(ns.checkpoint, map_location=dev)
    model_obj.eval().to(dev)

    raw_lr = load_volume_any(ns.input, h5_dataset_key=ns.h5_dataset_key)
    in_path_obj = Path(ns.input)
    out_voxel_um: float | tuple[float, ...] | None = ns.voxel_size_um
    if (
        out_voxel_um is None
        and in_path_obj.is_dir()
        and in_path_obj.name.lower().endswith((".zarr", ".ome.zarr"))
    ):
        lr_bc = read_ome_zarr(in_path_obj, level=0)
        if lr_bc.is_calibrated:
            sf = float(model_obj.super_res_factor)
            out_voxel_um = tuple(s / sf for s in lr_bc.spacing_in("um"))

    saved_rad = (
        None if ns.recompute_input_norm else getattr(model_obj, "_saved_radiometric_state", None)
    )

    if isinstance(saved_rad, dict) and ns.norm_mode is None and ns.vmin is None and ns.vmax is None:
        use_fov = bool(ns.circular_fov_mask or saved_rad.get("circular_fov_mask", False))
        fov = build_circular_fov_mask(raw_lr.shape) if use_fov else None
        saved_mode = str(saved_rad.get("norm_mode", "percentile"))
        b_lr = saved_rad.get("norm_bounds_lr")
        if saved_mode == "none":
            lr_norm = normalize_volume(raw_lr, mode="none", mask=fov)
        elif isinstance(b_lr, (list, tuple)) and len(b_lr) == 2:
            lr_norm = normalize_volume(
                raw_lr, mode="fixed", vmin=float(b_lr[0]), vmax=float(b_lr[1]), mask=fov
            )
        else:
            lr_norm = normalize_volume(raw_lr, mode="percentile", mask=fov)
        cal_params = saved_rad.get("cross_scan_calibration_params")
        if isinstance(cal_params, dict) and str(cal_params.get("mode", "none")) != "none":
            lr_norm = apply_cross_scan_calibration(lr_norm, cal_params, mask=fov)
    else:
        eff_mode = cast(NormMode, ns.norm_mode if ns.norm_mode is not None else "percentile")
        fov = build_circular_fov_mask(raw_lr.shape) if ns.circular_fov_mask else None
        lr_norm = normalize_volume(
            raw_lr,
            mode=eff_mode,
            vmin=ns.vmin,
            vmax=ns.vmax,
            mask=fov,
        )

    out_is_zarr = out_path.name.lower().endswith((".zarr", ".ome.zarr"))

    def _save_aux_output(arr: np.ndarray, dest_path: Path) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.name.lower().endswith((".zarr", ".ome.zarr")):
            vol_u = (
                Volume.from_array(arr, voxel_size_um=out_voxel_um)
                if out_voxel_um is not None
                else Volume.uncalibrated(arr)
            )
            write_ome_zarr(vol_u, dest_path, overwrite=True)
        else:
            tifffile.imwrite(dest_path, arr, bigtiff=True, photometric="minisblack")

    if isinstance(model_obj, SR3DResNet):
        pred_3d = super_resolve_volume_3d(
            generator_3d=model_obj.gen,
            low_volume=lr_norm,
            scale_factor=int(round(float(model_obj.super_res_factor))),
            device=dev,
        )
        fused = np.ascontiguousarray(pred_3d, dtype=np.float32)
        if out_is_zarr:
            export_volume_ome_zarr(
                fused,
                out_path,
                voxel_size_um=out_voxel_um,
                include_pore_labels=True,
                overwrite=True,
            )
        else:
            _save_aux_output(fused, out_path)
        if unc_path is not None:
            _save_aux_output(np.zeros_like(fused, dtype=np.float32), unc_path)
    else:
        axes = [cast(AxisName, ax) for ax in ns.axes]
        want_unc = unc_path is not None or out_is_zarr
        ortho = super_resolve_volume_orthogonal(
            generator=model_obj.gen,
            low_volume=lr_norm,
            scale_factor=model_obj.super_res_factor,
            axes=axes,
            fusion=cast(FusionMode, ns.fusion),
            batch_size=ns.batch_size,
            device=dev,
            context_slices=model_obj.context_slices,
            return_uncertainty=want_unc,
        )
        fused = np.ascontiguousarray(ortho["fused"], dtype=np.float32)
        sig_ortho = (
            np.ascontiguousarray(ortho["uncertainty_std"], dtype=np.float32)
            if "uncertainty_std" in ortho
            else None
        )
        if out_is_zarr:
            export_volume_ome_zarr(
                fused,
                out_path,
                voxel_size_um=out_voxel_um,
                include_pore_labels=True,
                sigma_ortho=sig_ortho,
                overwrite=True,
            )
        else:
            _save_aux_output(fused, out_path)
        if unc_path is not None and sig_ortho is not None:
            _save_aux_output(sig_ortho, unc_path)
    print(f"[predict-volume] fused shape={fused.shape} saved to {out_path}")
    return 0


class OpenLSRXLightningCLI(LightningCLI):
    """LightningCLI binding ``SRGAN`` (default) or ``SR3DResNet`` to any ``LightningDataModule``."""


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
