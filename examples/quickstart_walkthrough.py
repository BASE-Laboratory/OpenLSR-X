#!/usr/bin/env python3
"""Self-contained end-to-end OpenLSR-X walkthrough and overview figure generator.

Demonstrates:
1. 3D porous-media XCT volume synthesis and realistic PSF + noise degradation.
2. Leakage-free 3D slab splitting across train / val / test.
3. Compact 2D SRGAN training with L1 generator warmup + adversarial fine-tuning.
4. 3D Tricubic vs Single-Axis (yz) vs 3-Axis Orthogonal Fourier Fusion on an
   independent isotropic 3D test cube.
5. Quantitative image-fidelity & petrophysical evaluation + publication figure export.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.ndimage import distance_transform_edt

from openlsrx import (
    SRGAN,
    SyntheticPoreSpyDataModule,
    evaluate_image_metrics,
    evaluate_petrophysical_metrics,
    generate_porespy_volume,
    super_resolve_volume_orthogonal,
    upsample_3d,
)
from openlsrx.metrics.petrophysics import segment_pore_space, two_point_correlation_radial


def run_walkthrough(
    output_figure: Path,
    extra_copy: Path | None = None,
    output_json: Path | None = None,
    volume_size: int = 64,
    epochs: int = 16,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Execute the end-to-end walkthrough and save the multi-panel overview figure."""
    pl.seed_everything(seed, workers=True)

    dm = SyntheticPoreSpyDataModule(
        volume_size=volume_size,
        scale_factor=2,
        feature_regime="low_feature",
        noise_regime="high_noise",
        patch_size_lr=24,
        batch_size=4,
        guard_slices_lr=1,
        seed=seed,
    )
    dm.setup()

    model = SRGAN(
        learning_rate=1e-3,
        adaptive_size=6,
        super_res_factor=2,
        num_blocks=4,
        out_channels=32,
        disc_features=[32, 32, 64, 64],
        pixel_loss_type="l1",
        pretrain_epochs=max(2, (3 * epochs) // 4),
        use_vgg_loss=False,
        use_adversarial_loss=True,
        adversarial_weight=5e-4,
        save_val_images=False,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=dm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.gen.to(device).eval()

    # Evaluate on an independent, strictly unseen isotropic 3D test cube (seed + 1000)
    lr_test, hr_test, _ = generate_porespy_volume(
        shape=volume_size,
        feature_regime="low_feature",
        noise_regime="high_noise",
        scale_factor=2,
        seed=seed + 1000,
    )

    pred_tricubic = upsample_3d(lr_test, scale_factors=2, mode="tricubic")
    ortho = super_resolve_volume_orthogonal(
        model.gen,
        lr_test,
        scale_factor=2,
        axes=("yz", "xz", "xy"),
        fusion="fourier",
        batch_size=8,
        device=device,
    )
    pred_yz = ortho["yz"]
    pred_fourier = ortho["fused"]

    methods = {
        "3D Tricubic": pred_tricubic,
        "Single-Axis (yz)": pred_yz,
        "3-Axis Fourier Fusion": pred_fourier,
    }
    summary: dict[str, dict[str, float]] = {}
    for name, vol in methods.items():
        img_m = evaluate_image_metrics(vol, hr_test)
        petro_m = evaluate_petrophysical_metrics(vol, hr_test, threshold_method="otsu")
        summary[name] = {**img_m, **petro_m}

    # Render publication-style 2-row overview figure
    lr_nn = upsample_3d(lr_test, scale_factors=2, mode="nearest")
    mid_y = hr_test.shape[1] // 2  # Through-plane (xz) cross-section highlights slice-stacking

    fig = plt.figure(figsize=(14.0, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1.0])

    panels = [
        ("LR Input (2x)", lr_nn[:, mid_y, :]),
        ("3D Tricubic", pred_tricubic[:, mid_y, :]),
        ("Single-Axis SR (yz)", pred_yz[:, mid_y, :]),
        ("3-Axis Fourier Fusion", pred_fourier[:, mid_y, :]),
        ("HR Ground Truth", hr_test[:, mid_y, :]),
    ]
    for col, (title, slc) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(slc, cmap="gray", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x (voxels)", fontsize=8)
        if col == 0:
            ax.set_ylabel("Through-plane z (voxels)", fontsize=8)
        ax.tick_params(labelsize=7)

    # Bottom-left (cols 0-1): Inscribed pore-radius distribution via EDT
    ax_psd = fig.add_subplot(gs[1, 0:2])
    hr_mask, thr = segment_pore_space(hr_test, method="otsu")
    dt_gt = distance_transform_edt(hr_mask)
    r_gt = dt_gt[hr_mask]
    bins = np.linspace(0.5, max(float(r_gt.max(initial=6.0)), 6.0), 14)
    centers = 0.5 * (bins[:-1] + bins[1:])
    gt_hist, _ = np.histogram(r_gt, bins=bins, density=True)
    ax_psd.plot(centers, gt_hist, "k-", lw=2.0, label="HR Ground Truth")
    colors = {
        "3D Tricubic": "#6c757d",
        "Single-Axis (yz)": "#e76f51",
        "3-Axis Fourier Fusion": "#1d6fb8",
    }
    styles = {"3D Tricubic": "--", "Single-Axis (yz)": "-.", "3-Axis Fourier Fusion": "-"}
    for name, vol in methods.items():
        m_mask, _ = segment_pore_space(vol, method="fixed", fixed_value=thr)
        r_pred = distance_transform_edt(m_mask)[m_mask]
        if r_pred.size > 0:
            hist, _ = np.histogram(r_pred, bins=bins, density=True)
            ax_psd.plot(
                centers, hist, linestyle=styles[name], color=colors[name], lw=1.7, label=name
            )
    ax_psd.set_title("Inscribed Pore-Radius PDF P(r)", fontsize=10, fontweight="bold")
    ax_psd.set_xlabel("Pore Radius r (voxels)", fontsize=8.5)
    ax_psd.set_ylabel("Probability Density", fontsize=8.5)
    ax_psd.legend(fontsize=7.5, frameon=False)
    ax_psd.grid(alpha=0.25)

    # Bottom-middle (cols 2-3): Radial two-point correlation S2(r)
    ax_s2 = fig.add_subplot(gs[1, 2:4])
    s2_gt = two_point_correlation_radial(hr_mask)
    radii_gt = np.arange(len(s2_gt))
    ax_s2.plot(radii_gt, s2_gt, "k-", lw=2.0, label="HR Ground Truth")
    for name, vol in methods.items():
        m_mask, _ = segment_pore_space(vol, method="fixed", fixed_value=thr)
        s2_p = two_point_correlation_radial(m_mask)
        ax_s2.plot(
            np.arange(len(s2_p)),
            s2_p,
            linestyle=styles[name],
            color=colors[name],
            lw=1.7,
            label=name,
        )
    ax_s2.set_title("Radial Two-Point Correlation S2(r)", fontsize=10, fontweight="bold")
    ax_s2.set_xlabel("Lag Distance r (voxels)", fontsize=8.5)
    ax_s2.set_ylabel("S2(r)", fontsize=8.5)
    ax_s2.grid(alpha=0.25)

    # Bottom-right (col 4): Through-plane gradient anisotropy error
    ax_ani = fig.add_subplot(gs[1, 4])
    labels = ["Tricubic", "1-Axis\n(yz)", "3-Axis\nFourier"]
    ani_vals = [
        summary["3D Tricubic"]["through_plane_anisotropy_error"],
        summary["Single-Axis (yz)"]["through_plane_anisotropy_error"],
        summary["3-Axis Fourier Fusion"]["through_plane_anisotropy_error"],
    ]
    bars = ax_ani.bar(labels, ani_vals, color=["#6c757d", "#e76f51", "#1d6fb8"], width=0.6)
    ax_ani.set_title(
        "Stacking Anisotropy\nError (lower is better)", fontsize=9.5, fontweight="bold"
    )
    ax_ani.set_ylabel("|Δ(∇z / ∇xy)|", fontsize=8.5)
    ax_ani.tick_params(labelsize=7.5)
    ax_ani.set_ylim(0.0, max(ani_vals) * 1.18 + 1e-4)
    ax_ani.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, ani_vals, strict=True):
        ax_ani.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + max(ani_vals) * 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=220)
    plt.close(fig)

    if extra_copy is not None:
        extra_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_figure, extra_copy)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenLSR-X quickstart walkthrough.")
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("docs/assets/quickstart_overview.png"),
        help="Destination path for the rendered overview figure.",
    )
    parser.add_argument(
        "--paper-figure",
        type=Path,
        default=Path("paper/figures/joss_overview.png"),
        help="Optional copy path for the JOSS manuscript figure.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--volume-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_walkthrough(
        output_figure=args.output_figure,
        extra_copy=args.paper_figure,
        output_json=args.output_json,
        volume_size=args.volume_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    for method, metrics in summary.items():
        print(
            f"{method:24s} | PSNR={metrics['psnr']:.2f} dB | SSIM={metrics['ssim']:.4f} "
            f"| |Δφ|={metrics['porosity_abs_error']:.4f} "
            f"| AnisotropyErr={metrics['through_plane_anisotropy_error']:.4f}"
        )


if __name__ == "__main__":
    main()
