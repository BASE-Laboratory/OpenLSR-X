#!/usr/bin/env python3
"""Generate publication-ready LaTeX tables, Markdown summaries, and 300-DPI figures."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt

from openlsrx.metrics import segment_pore_space

KEY_METRICS: list[tuple[str, str, bool]] = [
    ("psnr", "PSNR (dB) $\\uparrow$", True),
    ("ssim_3d_triplanar", "3D SSIM $\\uparrow$", True),
    ("porosity_abs_error", "$|\\Delta \\phi|$ $\\downarrow$", False),
    ("pore_dice", "Pore Dice $\\uparrow$", True),
    ("pore_radius_wasserstein", "PSD $W_1$ $\\downarrow$", False),
    ("through_plane_anisotropy_error", "Aniso. Err. $\\downarrow$", False),
]


def _aggregate_csv(
    csv_path: Path, group_col: str
) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = str(row[group_col])
            m = str(row["method"])
            for col, _, _ in KEY_METRICS:
                if col in row and row[col] != "":
                    buckets[(g, m)][col].append(float(row[col]))

    summary: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for key, metric_lists in buckets.items():
        summary[key] = {
            col: (float(np.mean(vals)), float(np.std(vals))) for col, vals in metric_lists.items()
        }
    return summary


def write_latex_and_markdown_tables(
    csv_path: Path,
    group_col: str,
    out_tex: Path,
    out_md: Path,
    caption: str,
    label: str,
) -> None:
    agg = _aggregate_csv(csv_path, group_col=group_col)
    groups = sorted({g for g, _ in agg})

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    # LaTeX booktabs table
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{ll" + "c" * len(KEY_METRICS) + "}",
        "\\toprule",
        "Dataset / Regime & Method & " + " & ".join(hdr for _, hdr, _ in KEY_METRICS) + " \\\\",
        "\\midrule",
    ]

    md_headers = ["Dataset / Regime", "Method"] + [
        hdr.replace("$\\uparrow$", "(↑)").replace("$\\downarrow$", "(↓)").replace("$", "")
        for _, hdr, _ in KEY_METRICS
    ]
    md_lines = [
        f"### {caption}",
        "",
        "| " + " | ".join(md_headers) + " |",
        "| " + " | ".join(["---"] * len(md_headers)) + " |",
    ]

    for g_idx, g in enumerate(groups):
        methods = [m for grp, m in agg if grp == g]
        # Find best method per metric within group
        best_method: dict[str, str] = {}
        for col, _, higher_better in KEY_METRICS:
            scored = [(m, agg[(g, m)][col][0]) for m in methods if col in agg[(g, m)]]
            if scored:
                best_method[col] = (
                    max(scored, key=lambda x: x[1])[0]
                    if higher_better
                    else min(scored, key=lambda x: x[1])[0]
                )

        g_tex = g.replace("_", "\\_")
        for m in methods:
            m_tex = m.replace("_", "\\_")
            tex_cells = [g_tex, m_tex]
            md_cells = [g, m]
            for col, _, _ in KEY_METRICS:
                mean, std = agg[(g, m)][col]
                cell_str = f"{mean:.4f}" if std < 1e-9 else f"{mean:.4f} $\\pm$ {std:.4f}"
                md_str = f"{mean:.4f}" if std < 1e-9 else f"{mean:.4f} ± {std:.4f}"
                if best_method.get(col) == m:
                    cell_str = f"\\textbf{{{cell_str}}}"
                    md_str = f"**{md_str}**"
                tex_cells.append(cell_str)
                md_cells.append(md_str)
            tex_lines.append(" & ".join(tex_cells) + " \\\\")
            md_lines.append("| " + " | ".join(md_cells) + " |")
        if g_idx + 1 < len(groups):
            tex_lines.append("\\midrule")

    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    out_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def plot_visual_comparison(example_dir: Path, out_fig: Path) -> None:
    """Create a multi-panel orthogonal slice, error-map, and pore-radius distribution figure."""
    hr = tifffile.imread(example_dir / "hr_test.tif")
    tric = tifffile.imread(example_dir / "tricubic_test.tif")
    sr_yz = tifffile.imread(example_dir / "srgan_yz_test.tif")
    sr_fused = tifffile.imread(example_dir / "srgan_fused_test.tif")

    # Inspect an orthogonal xy plane (through-plane relative to yz training!)
    mid_w = hr.shape[2] // 2
    panels = [
        ("Tricubic 3D (xy)", tric[:, :, mid_w]),
        ("OpenLSR-X Single-Axis yz (xy)", sr_yz[:, :, mid_w]),
        ("OpenLSR-X 3-Axis Fourier (xy)", sr_fused[:, :, mid_w]),
        ("Ground Truth HR (xy)", hr[:, :, mid_w]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), constrained_layout=True)
    for idx, (title, img) in enumerate(panels):
        ax_top = axes[0, idx]
        ax_top.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax_top.set_title(title, fontsize=10, fontweight="bold")
        ax_top.axis("off")

    # Bottom row: 3 absolute error maps + 1 inscribed pore-radius distribution comparison
    for idx, (label, vol_pred) in enumerate(
        [
            ("|Error| Tricubic 3D", tric[:, :, mid_w]),
            ("|Error| Single-Axis yz", sr_yz[:, :, mid_w]),
            ("|Error| 3-Axis Fourier", sr_fused[:, :, mid_w]),
        ]
    ):
        err = np.abs(vol_pred - hr[:, :, mid_w])
        im = axes[1, idx].imshow(err, cmap="magma", vmin=0.0, vmax=0.35)
        axes[1, idx].set_title(f"{label} (MAE={err.mean():.3f})", fontsize=9.5)
        axes[1, idx].axis("off")
        fig.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.03)

    ax_psd = axes[1, 3]
    for name, vol, color, style in (
        ("Ground Truth HR", hr, "#111111", "-"),
        ("Tricubic 3D", tric, "#d95f02", "--"),
        ("OpenLSR-X 3-Axis", sr_fused, "#1b9e77", "-."),
    ):
        mask, _ = segment_pore_space(vol, pore_is_low=True)
        radii = distance_transform_edt(mask)[mask]
        if radii.size > 0:
            bins = np.linspace(0.5, max(float(radii.max()), 3.0), 20)
            hist, edges = np.histogram(radii, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax_psd.plot(centers, hist, label=name, color=color, linestyle=style, linewidth=1.8)

    ax_psd.set_title("Pore Inscribed Radius PDF", fontsize=9.5, fontweight="bold")
    ax_psd.set_xlabel("Inscribed Radius (voxels)")
    ax_psd.set_ylabel("Probability Density")
    ax_psd.legend(fontsize=8)
    ax_psd.grid(alpha=0.25)

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300)
    fig.savefig(out_fig.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manuscript LaTeX tables and figures.")
    parser.add_argument(
        "--porespy-csv",
        default="results/porespy_study/porespy_study_results.csv",
        help="Path to PoreSpy study CSV.",
    )
    parser.add_argument(
        "--rock-csv",
        default="results/public_rock/public_rock_results.csv",
        help="Path to public rock benchmark CSV.",
    )
    parser.add_argument(
        "--example-volumes",
        default="results/porespy_study/example_volumes",
        help="Directory containing saved example 3D TIFF volumes.",
    )
    parser.add_argument("--out-dir", default="results/paper_artifacts", help="Output directory.")
    ns = parser.parse_args()

    out_dir = Path(ns.out_dir)
    ps_csv = Path(ns.porespy_csv)
    if ps_csv.is_file():
        write_latex_and_markdown_tables(
            csv_path=ps_csv,
            group_col="regime",
            out_tex=out_dir / "tables" / "porespy_benchmark.tex",
            out_md=out_dir / "tables" / "porespy_benchmark.md",
            caption="3D XCT Super-Resolution Benchmark across Factorial PoreSpy Regimes.",
            label="tab:porespy_benchmark",
        )
        print(f"[make_paper_outputs] Wrote PoreSpy tables to {out_dir / 'tables'}")

    rk_csv = Path(ns.rock_csv)
    if rk_csv.is_file():
        write_latex_and_markdown_tables(
            csv_path=rk_csv,
            group_col="dataset",
            out_tex=out_dir / "tables" / "public_rock_benchmark.tex",
            out_md=out_dir / "tables" / "public_rock_benchmark.md",
            caption="3D Micro-CT Super-Resolution on Bentheimer Sandstone and Indiana Limestone.",
            label="tab:rock_benchmark",
        )
        print(f"[make_paper_outputs] Wrote Rock XCT tables to {out_dir / 'tables'}")

    ex_dir = Path(ns.example_volumes)
    if (ex_dir / "hr_test.tif").is_file():
        fig_path = out_dir / "figures" / "orthogonal_sr_comparison.png"
        plot_visual_comparison(ex_dir, fig_path)
        print(f"[make_paper_outputs] Wrote visual comparison figure to {fig_path}")


if __name__ == "__main__":
    main()
