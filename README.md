# OpenLSR-X

[![CI](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/ci.yml/badge.svg)](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/ci.yml)
[![Docs](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/docs.yml/badge.svg)](https://base-laboratory.github.io/OpenLSR-X/)
[![JOSS Paper](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/release.yml/badge.svg)](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/release.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

**OpenLSR-X** is a reproducible Python framework and benchmark suite for **physics- and topology-guided 2D/2.5D-to-3D Super-Resolution of X-Ray Computed Tomography (XCT)** porous-media scans using shifted-window Transformers (`SwinIR`), Residual-in-Residual Dense Blocks (`RRDB` / `RaGAN`), SRGAN, and **epistemic confidence-weighted 3-axis Fourier fusion**. Originally developed during the PhD research of **Calum Green** (Imperial College London & Diamond Light Source) and maintained by **BASE Laboratory**, OpenLSR-X bridges deep-learning super-resolution with quantitative **synchrotron beamline metrology, reactive multi-phase segmentation, and uncertainty-propagated 3D diffusive transport**.

---

## Highlights

- **2D & `2.5D` Training → Uncertainty-Aware Isotropic 3D Reconstruction:** Train memory-efficient `1`-slice or `2k + 1` multi-slice (`2.5D`) generators (`SwinIR`, `RRDB` + `RaGAN`, `SRGAN`/`SRResNet`) alongside native `SR3DResNet` baselines across integer and fractional synchrotron optical magnifications (`2x`, `2.5x`, `4x`), and fuse orthogonal sweeps (`yz + xz + xy`) via **Confidence-Weighted Directional Fourier Fusion** (`confidence_fourier`) with zero-overhead voxel-wise epistemic uncertainty $\sigma_{\text{ortho}}(\mathbf{x})$.
- **Physics- & Topology-Preserving Objectives:** Combines Beer–Lambert **X-ray attenuation conservation** (`AttenuationConservationLoss`), differentiable **3D pore-throat skeleton preservation** (`SoftCLDiceLoss`), 2D/3D **Sobel phase-boundary sharpness** (`SobelGradientLoss`), and **Focal Frequency Loss** (`FocalFrequencyLoss`).
- **Leakage-Free 3D Spatial Splitting:** Enforces strictly disjoint 3D sub-volume slabs (with configurable guard margins and per-axis boundary-safe `2.5D` reflect padding) across `train`, `val`, and `test`.
- **Synchrotron Spectral, Petrophysical & Diffusive Transport Metrology:** Evaluates **3D Fourier Shell Correlation (FSC)** & **2D Fourier Ring Correlation (FRC)** (`1/2`-bit and `0.143` effective resolution in $\mu\text{m}$), tri-phase `Zn-13X` reactive core-rim segmentation (`Macropore` / `Na-13X` / `Zn-13X`), porosity/Euler/surface-area metrics, quasi-static drainage capillary pressure $P_c(S_w)$, and **3D steady-state Laplace diffusive tortuosity** ($D_{\text{eff}} / D_0$, $\tau$) with epistemic uncertainty intervals $[D_{\text{eff}}^-, D_{\text{eff}}^+]$ and direct **OpenImpala** export.
- **Out-of-the-Box Benchmarks & Open Synchrotron Profiles:** Ships with deterministic **Factorial PoreSpy** benchmarks, **Open Rock Micro-CT** (Bentheimer Sandstone & Indiana Limestone), and automated Zenodo staging + calibrated phantoms for the open Diamond Light Source I13-2 multi-resolution `Zn-13X` dataset ([*Scientific Data*, 2025](https://doi.org/10.1038/s41597-025-04605-9)).

---

## Quickstart

### 1. Installation (`uv`)

```bash
git clone https://github.com/BASE-Laboratory/OpenLSR-X.git
cd OpenLSR-X
uv venv --python 3.11
uv pip install -e ".[dev,docs]"
```

### 2. Train, Test, and Predict via Unified CLI

Train the physics- and topology-guided `2.5D SwinIR` configuration on Diamond Light Source I13-2 `Zn-13X` (`2x`) or synthetic PoreSpy media:

```bash
openlsrx download-zn13x --synthetic-fallback
openlsrx fit -c configs/i13-xct/zn13x_2x_swinir_25d_physics.yaml
```

Prepare open rock micro-CT benchmark stacks and train on Bentheimer Sandstone:

```bash
openlsrx download-benchmarks
openlsrx fit -c configs/labxct/bs_config.yaml
```

Evaluate image-fidelity, petrophysical, 3D FSC spectral, tri-phase `Zn-13X`, and 3D diffusive transport metrics (+ OpenImpala export):

```bash
openlsrx evaluate \
  --pred predicted_stack.tiff \
  --target high_res_stack.tiff \
  --voxel-size 0.8125 \
  --include-transport \
  --include-spectral \
  --include-triphase \
  --export-openimpala outputs/openimpala_ready \
  --output metrics.json
```

---

## Reproducing Manuscript Benchmarks

Run the Factorial PoreSpy, Open Rock Micro-CT, and 7-arm Diamond I13-2 `Zn-13X` studies and generate publication-ready LaTeX `booktabs` tables and 300-DPI comparison figures:

```bash
# Rapid CPU smoke verification (~2 minutes)
python scripts/benchmarks/porespy_study.py --fast
python scripts/benchmarks/public_rock_xct.py --fast
python scripts/benchmarks/zn13x_synchrotron_study.py --fast
python scripts/benchmarks/aggregate_campaign.py --inputs outputs --output-dir outputs/campaign_summary

# Full multi-seed GPU campaign (see docs/npj-study-design.md)
python scripts/benchmarks/zn13x_synchrotron_study.py --volume-size 256 --epochs 300 --seeds 42 43 44
python scripts/benchmarks/aggregate_campaign.py --inputs outputs --output-dir outputs/campaign_summary
```

---

## Repository Structure

```text
OpenLSR-X/
├── configs/                  # Validated LightningCLI YAML configs (base, porespy, labxct, i13-xct)
├── docs/                     # MkDocs user, dataset, benchmark, API, and npj study design docs
├── paper/                    # JOSS manuscript (paper.md) and bibliography (paper.bib)
├── scripts/
│   ├── benchmarks/           # PoreSpy, Rock XCT & I13-2 Zn-13X runners + multi-seed campaign aggregator
│   └── hpc/                  # SLURM and SGE ('Young') GPU batch templates
├── src/openlsrx/
│   ├── baselines/            # 2D & 3D classical interpolation baselines (Nearest, Trilinear, Tricubic, Lanczos)
│   ├── data/                 # PoreSpy, Rock XCT, DLS I13-2 Zn-13X loaders & 2.5D leakage-free 3D splitting
│   ├── inference/            # Single-axis, 3-axis Fourier, and epistemic confidence-weighted 3D fusion
│   ├── io/                   # OpenImpala 3D TIFF + .inputs deck exporter
│   ├── losses/               # Attenuation conservation, soft-clDice topology, Sobel, FFL, RaGAN & VGGLoss
│   ├── metrics/              # 2D/3D PSNR/SSIM, 3D FSC / 2D FRC, tri-phase petrophysics & 3D Laplace transport
│   ├── models/               # SwinIRGenerator, RRDBGenerator, SRGAN/SRResNet, and native SR3DResNet
│   └── cli.py                # Unified `openlsrx` CLI entrypoint
└── tests/                    # 71 unit, analytical physics/spectral/transport, Lightning & CLI tests (>90% cov)
```

---

## Citation & License

OpenLSR-X is released under the [BSD 3-Clause License](LICENSE). If you use OpenLSR-X in academic work, please cite the software via [`CITATION.cff`](CITATION.cff) and Ledig et al. (CVPR 2017).

Maintained by **BASE Laboratory**, building on the PhD work of **Calum Green**. Contributions are welcome—see [`CONTRIBUTING.md`](CONTRIBUTING.md).
