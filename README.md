# OpenLSR-X

[![CI](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/ci.yml/badge.svg)](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/ci.yml)
[![Docs](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/docs.yml/badge.svg)](https://base-laboratory.github.io/OpenLSR-X/)
[![JOSS Paper](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/release.yml/badge.svg)](https://github.com/BASE-Laboratory/OpenLSR-X/actions/workflows/release.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

**OpenLSR-X** is a reproducible Python framework and benchmark suite for **2D-to-3D Super-Resolution of X-Ray Computed Tomography (XCT)** porous-media scans using Generative Adversarial Networks (SRGAN) and multi-axis 3D Fourier fusion. Originally developed during the PhD research of **Calum Green** (Imperial College London & Diamond Light Source) and maintained by **BASE Laboratory**, OpenLSR-X bridges deep-learning super-resolution with quantitative **petrophysical and topological validation**.

---

## Highlights

- **2D Training → Isotropic 3D Reconstruction:** Train memory-efficient 2D slice models (with reflect padding, `PixelNorm`, generator warmup, and optional VGG-19 perceptual loss) and reconstruct artifact-free 3D volumes via **3-Axis Orthogonal Fourier Fusion** (`yz + xz + xy`).
- **Leakage-Free 3D Spatial Splitting:** Enforces strictly disjoint 3D sub-volume slabs (with configurable guard margins) across `train`, `val`, and `test` so orthogonal evaluation slices never share voxels with training slices.
- **Petrophysical & Topological Metrics:** Evaluates not only 2D/3D PSNR and tri-planar SSIM, but also segmented **porosity error** ($|\Delta\phi|$), **pore Dice/IoU**, **specific surface area** ($S_v$), **3D Euler characteristic density** ($\chi_V$), **inscribed pore-radius Wasserstein-1 distance**, **two-point correlation** $S_2(r)$, and **through-plane gradient anisotropy**.
- **Out-of-the-Box Benchmarks & Open Synchrotron Profiles:** Ships with deterministic **Factorial PoreSpy** benchmarks, an **Open Rock Micro-CT** runner (Bentheimer Sandstone & Indiana Limestone), and NeXus/HDF5 profiles for the open Diamond Light Source I13-2 multi-resolution `Zn-13X` dataset ([*Scientific Data*, 2025](https://doi.org/10.1038/s41597-025-04605-9)).

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

Run training out of the box on a synthetic 3D PoreSpy porous-media volume:

```bash
openlsrx fit -c configs/base/config.yaml
```

Prepare open rock micro-CT benchmark stacks and train on Bentheimer Sandstone:

```bash
openlsrx download-benchmarks
openlsrx fit -c configs/labxct/bs_config.yaml
```

Predict an orthogonal super-resolved 3D BigTIFF stack from a trained checkpoint:

```bash
openlsrx predict -c configs/labxct/bs_config_predict.yaml --ckpt_path path/to/checkpoint.ckpt
```

Evaluate image-fidelity and petrophysical XCT metrics on any pair of 3D TIFF/HDF5 volumes:

```bash
openlsrx evaluate --pred predicted_stack.tiff --target high_res_stack.tiff --output metrics.json
```

---

## Reproducing Manuscript Benchmarks

Run the Factorial PoreSpy and Open Rock Micro-CT studies and generate publication-ready LaTeX `booktabs` tables and 300-DPI comparison figures:

```bash
# Rapid CPU smoke verification (< 2 minutes)
python scripts/benchmarks/porespy_study.py --fast
python scripts/benchmarks/public_rock_xct.py --fast
python scripts/benchmarks/make_paper_outputs.py

# Full multi-seed GPU benchmark
python scripts/benchmarks/porespy_study.py --volume-size 256 --epochs 300 --use-vgg
python scripts/benchmarks/public_rock_xct.py --volume-size 256 --epochs 300
python scripts/benchmarks/make_paper_outputs.py
```

---

## Repository Structure

```text
OpenLSR-X/
├── configs/                  # Validated LightningCLI YAML configs (base, porespy, labxct, i13-xct)
├── docs/                     # MkDocs user, dataset, benchmark, and API documentation
├── paper/                    # JOSS manuscript (paper.md) and bibliography (paper.bib)
├── scripts/
│   ├── benchmarks/           # End-to-end PoreSpy & Rock XCT study runners + LaTeX/figure builder
│   └── hpc/                  # SLURM and SGE ('Young') GPU batch templates
├── src/openlsrx/
│   ├── baselines/            # 2D & 3D classical interpolation baselines (Nearest, Trilinear, Tricubic, Lanczos)
│   ├── data/                 # Synthetic PoreSpy, TIFF/HDF5 volume loaders, and leakage-free 3D splitting
│   ├── inference/            # 2D-to-3D single-axis and 3-axis orthogonal Fourier stack fusion
│   ├── losses/               # PixelNorm, FP16-safe custom sigmoid, and checkpoint-clean VGGLoss
│   ├── metrics/              # 2D/3D PSNR/SSIM and porous-media petrophysical metrics
│   ├── models/               # ConvBlock, ResidualBlock, Generator, Discriminator, and SRGAN/SRResNet
│   └── cli.py                # Unified `openlsrx` CLI entrypoint
└── tests/                    # Unit, analytical phantom, end-to-end Lightning, and YAML schema tests
```

---

## Citation & License

OpenLSR-X is released under the [BSD 3-Clause License](LICENSE). If you use OpenLSR-X in academic work, please cite the software via [`CITATION.cff`](CITATION.cff) and Ledig et al. (CVPR 2017).

Maintained by **BASE Laboratory**, building on the PhD work of **Calum Green**. Contributions are welcome—see [`CONTRIBUTING.md`](CONTRIBUTING.md).
