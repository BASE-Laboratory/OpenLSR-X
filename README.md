# OpenLSR-X

[![CI](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/ci.yml/badge.svg)](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/ci.yml)
[![Docs](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/docs.yml/badge.svg)](https://base-laboratory.github.io/OpenLSR-X/)
[![JOSS Paper](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/release.yml/badge.svg)](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/release.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

**OpenLSR-X** is a reproducible Python framework and benchmark suite for **physics- and topology-guided 2D/2.5D-to-3D Super-Resolution of X-Ray Computed Tomography (XCT)** porous-media scans using shifted-window Transformers (`SwinIR`), Residual-in-Residual Dense Blocks (`RRDB` / `RaGAN`), SRGAN, and **confidence-weighted 3-axis Fourier fusion**. Originally developed during the PhD research of **Calum Green** (Imperial College London & Diamond Light Source) and maintained by **BASE Laboratory**, OpenLSR-X bridges deep-learning super-resolution with quantitative **synchrotron beamline metrology, reactive multi-phase segmentation, and uncertainty-propagated 3D diffusive transport**.

---

## Highlights

- **Cell-Centered 2D & `2.5D` Training → Isotropic 3D Reconstruction:** Train memory-efficient `1`-slice or native-LR-strided `2k + 1` multi-slice (`2.5D`) generators (`SwinIR`, `RRDB` + `RaGAN`, `SRGAN`/`SRResNet`) alongside native `SR3DResNet` baselines across integer and fractional synchrotron optical magnifications (`2x`, `2.5x`, `4x`, `8x`), and fuse cell-centered (`grid_mode=True`) orthogonal sweeps (`yz + xz + xy`) via **Confidence-Weighted Directional Fourier Fusion** (`confidence_fourier`) with zero-overhead voxel-wise **orthogonal view-disagreement uncertainty** $\sigma_{\text{ortho}}(\mathbf{x})$ ($n=3$ views).
- **Physics- & Topology-Guided Objectives:** Combines **linear-attenuation ($\mu$) downsampling consistency** (`AttenuationConservationLoss`, with `2.5D` through-plane $z$-PSF curvature correction), differentiable **bimodal pore-throat skeleton preservation** (`SoftCLDiceLoss`), 2D/3D **Sobel phase-boundary sharpness** (`SobelGradientLoss`), and **Focal Frequency Loss** (`FocalFrequencyLoss`).
- **Split-Isolated 3D Spatial Splitting & Normalization:** Partitions 3D volumes into disjoint sub-volume slabs (separated by PSF guard margins) *before* fitting intensity normalization bounds on `train` only and *before* per-split synthetic PSF degradation.
- **Synchrotron Spectral, Petrophysical & Diffusive Transport Metrology:** Evaluates **3D Fourier Shell Correlation (FSC)** & **2D Fourier Ring Correlation (FRC)** (`FSC = 0.5` prediction-vs-reference crossover frequency & spectral AUC), ground-truth-thresholded porosity/Euler/surface-area metrics, tri-phase `Zn-13X` reactive core-rim segmentation (`Macropore` / `Na-13X` / `Zn-13X`), quasi-static drainage capillary pressure $S_{\text{nwp}}(P_c)$, and residual-verified **3D steady-state Laplace diffusive tortuosity** ($D_{\text{eff}} / D_0$, $\tau$) with view-disagreement sensitivity bounds $[D_{\text{eff}}^-, D_{\text{eff}}^+]$ and direct **OpenImpala** export.
- **Provenance-Verified Open Benchmarks:** Ships with deterministic **Factorial PoreSpy** benchmarks, **Open Rock Micro-CT** (Bentheimer Sandstone from Zenodo `5542624` & calibrated carbonate phantom), and coaxial sub-voxel-registered Zenodo staging (`10.5281/zenodo.14245751`) + explicit provenance tracking (`source_kind`, `is_experimental_xct`) for the Diamond Light Source I13-2 multi-resolution `Zn-13X` dataset ([*Scientific Data*, 2025](https://doi.org/10.1038/s41597-025-04605-9)).

---

## Quickstart

### 1. Installation (`uv`)

```bash
git clone https://github.com/BASE-Laboratory/OpenLSR-X.git
cd OpenLSR-X
uv venv --python 3.11
uv pip install -e ".[dev,docs]"
```

### 2. Train, Reconstruct 3D Volumes, and Evaluate via Unified CLI

Stage the Diamond Light Source I13-2 `Zn-13X` (`2x` & `8x`) benchmark (pass `--offline` on air-gapped nodes for explicit synthetic phantoms) and train the multi-axis `2.5D SwinIR` configuration:

```bash
openlsrx download-zn13x --dest data/benchmarks
openlsrx fit -c configs/i13-xct/zn13x_2x_swinir_25d_physics.yaml
```

Run standalone 3-axis orthogonal `confidence_fourier` super-resolution + view-disagreement uncertainty export on any 3D LR volume:

```bash
openlsrx predict-volume \
  --ckpt lightning_logs/version_0/checkpoints/last.ckpt \
  --input data/benchmarks/zn13x_2x_lr.tif \
  --output-dir results/zn13x_2x_sr \
  --fusion confidence_fourier \
  --save-uncertainty
```

Prepare open rock micro-CT benchmark stacks and train on Bentheimer Sandstone:

```bash
openlsrx download-benchmarks --dest data/benchmarks
openlsrx fit -c configs/labxct/bs_config.yaml
```

Evaluate image-fidelity, petrophysical, 3D FSC spectral, tri-phase `Zn-13X`, and 3D diffusive transport metrics (+ OpenImpala export):

```bash
openlsrx evaluate \
  --pred results/zn13x_2x_sr/zn13x_2x_lr_sr_confidence_fourier.tif \
  --target data/benchmarks/zn13x_2x_hr.tif \
  --voxel-size 0.8125 \
  --include-transport \
  --include-spectral \
  --include-triphase \
  --export-openimpala-dir results/openimpala_ready \
  --output results/metrics.json
```

---

## Reproducing Manuscript Benchmarks

Run the Factorial PoreSpy, Open Rock Micro-CT, and 9-arm Diamond I13-2 `Zn-13X` studies and generate publication-ready LaTeX `booktabs` tables and comparison figures (production runs fail loudly if real Zenodo/HDF5 data are unavailable unless `--allow-synthetic-fallback` is explicitly passed):

```bash
# Rapid offline CPU smoke verification (~2 minutes; explicitly labeled synthetic phantoms)
python scripts/benchmarks/porespy_study.py --fast
python scripts/benchmarks/public_rock_xct.py --fast
python scripts/benchmarks/zn13x_synchrotron_study.py --fast
python scripts/benchmarks/aggregate_campaign.py --inputs results --output-dir results/campaign_summary

# Full multi-seed GPU campaign on real Zenodo / beamline HDF5 data (see docs/npj-study-design.md)
python scripts/benchmarks/zn13x_synchrotron_study.py --lr-size 48 --epochs 120 --seeds 42 43 44
python scripts/benchmarks/aggregate_campaign.py --inputs results --output-dir results/campaign_summary
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
│   └── hpc/                  # Isambard-AI GH200 SLURM array, generic SLURM, and SGE ('Young') templates
├── src/openlsrx/
│   ├── baselines/            # Cell-centered 2D & 3D classical interpolation baselines
│   ├── data/                 # PoreSpy, Rock XCT, DLS I13-2 Zn-13X loaders & split-isolated 2.5D datasets
│   ├── inference/            # Cell-centered single-axis, 3-axis Fourier, and confidence-weighted 3D fusion
│   ├── io/                   # OpenImpala 3D TIFF + AMReX .inputs deck exporter
│   ├── losses/               # Attenuation consistency, soft-clDice topology, Sobel, FFL & VGGLoss
│   ├── metrics/              # 2D/3D PSNR/SSIM, 3D FSC / 2D FRC, HR-locked petrophysics & 3D Laplace transport
│   ├── models/               # SwinIRGenerator, RRDBGenerator, SRGAN/SRResNet, and native SR3DResNet
│   └── cli.py                # Unified `openlsrx` CLI entrypoint
└── tests/                    # Comprehensive unit, analytical physics/spectral/transport & red-team regression suite
```

---

## Citation & License

OpenLSR-X is released under the [BSD 3-Clause License](LICENSE). If you use OpenLSR-X in academic work, please cite the software via [`CITATION.cff`](CITATION.cff) and Ledig et al. (CVPR 2017).

Maintained by **BASE Laboratory**, building on the PhD work of **Calum Green**. Contributions are welcome—see [`CONTRIBUTING.md`](CONTRIBUTING.md).
