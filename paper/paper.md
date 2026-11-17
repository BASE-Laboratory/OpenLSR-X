---
title: "OpenLSR-X: A Python Framework for 2D-to-3D Orthogonal Super-Resolution and Petrophysical Validation in X-Ray Computed Tomography"
tags:
  - Python
  - PyTorch
  - X-ray computed tomography
  - porous media
  - super-resolution
  - petrophysics
authors:
  - name: Calum Green
    affiliation: 1
  - name: BASE Laboratory Contributors
    affiliation: 1
affiliations:
  - name: BASE Laboratory
    index: 1
date: 17 November 2026
bibliography: paper.bib
---

# Summary

**OpenLSR-X** is an open-source Python package built on PyTorch [@paszke2019pytorch] and PyTorch Lightning [@falcon2019lightning] for super-resolving three-dimensional X-ray computed tomography (XCT) scans of porous materials using two-dimensional generative adversarial networks (SRGANs) [@ledig2017photo] combined with three-dimensional orthogonal Fourier fusion. Alongside training and inference pipelines supporting synthetic porous media [@gostick2019porespy], laboratory micro-CT stacks, and synchrotron NeXus/HDF5 reconstructions, OpenLSR-X provides a quantitative evaluation suite measuring both classical image fidelity and domain-critical petrophysical properties—including segmented porosity, specific surface area, Euler connectivity, inscribed pore-radius distributions, and two-point spatial correlations.

# Statement of Need

Three-dimensional X-ray micro-CT is a foundational characterization technique across hydrogeology, carbon storage, battery electrodes, catalysis, and rock mechanics [@blunt2013pore]. However, XCT instruments enforce an inherent trade-off between sample field of view (FOV) and voxel resolution: scans broad enough to capture a representative elementary volume (REV) frequently under-resolve sub-micron pore throats, biasing multiphase flow and transport simulations [@wang2019deep].

While 3D convolutional and generative super-resolution models can recover fine-scale pore geometry [@janssens2020computed], training fully 3D adversarial networks on gigavoxel tomograms requires substantial GPU memory and registered 3D low/high-resolution scan pairs that are expensive to acquire and align. Conversely, applying a 2D slice-wise super-resolution network along a single tomographic axis introduces through-plane discontinuities ("venetian blind" stacking artifacts) and anisotropic frequency distortion. Furthermore, general-purpose computer-vision toolboxes evaluate super-resolution almost exclusively via PSNR and SSIM, which correlate poorly with topological pore-throat preservation.

OpenLSR-X fills this gap by providing porous-media researchers with:

1. **Memory-efficient 2D-to-3D super-resolution** that trains on 2D slices from spatially disjoint 3D slabs and reconstructs isotropic 3D volumes via directional 3D Fourier fusion across orthogonal (`yz`, `xz`, `xy`) planes.
2. **Leakage-free 3D benchmark infrastructure** that prevents cross-plane train/test voxel intersection when evaluating orthogonal generalization.
3. **Integrated petrophysical diagnostics** that directly test whether super-resolved volumes preserve porosity, interfacial area, topology, and pore-size distributions relative to ground truth and 3D interpolation baselines.

# Architecture and Orthogonal 3D Fusion

OpenLSR-X adapts the residual sub-pixel generator and discriminator architecture of @ledig2017photo specifically for single-channel quantitative XCT attenuation data:

- **Artifact-suppressed convolutions:** Reflection padding (`padding_mode="reflect"`) eliminates bright-rim zero-padding boundary artifacts across cropped sub-slices, while channel-wise feature vector normalization (`PixelNorm`) [@karras2018progressive] stabilizes adversarial training under small batch sizes without cross-sample batch-normalization drift.
- **Controlled ablation and warmup:** A unified `SRGAN` `LightningModule` supports generator-only `SRResNet` training ($L_1$/MSE), scheduled generator warmup epochs prior to discriminator activation, and ImageNet-normalized VGG-19 perceptual losses whose frozen weights are isolated from checkpoint files.
- **3D Orthogonal Fourier Fusion:** Given a low-resolution 3D scan $V_{\text{LR}} \in \mathbb{R}^{d \times h \times w}$ and scale factor $s$, OpenLSR-X generates three candidate reconstructions $\hat{V}_{yz}, \hat{V}_{xz}, \hat{V}_{xy} \in \mathbb{R}^{sd \times sh \times sw}$ by applying the 2D generator across each orthogonal plane and fuses their 3D discrete Fourier transforms weighted by each view's super-resolved in-plane spatial wavevectors.

# Petrophysical Validation and Reproducible Benchmarks

To support reproducible methods comparisons without proprietary scan dependencies, OpenLSR-X ships with two automated benchmark suites (`scripts/benchmarks/`):

1. **Factorial PoreSpy Study:** Evaluates single-scale and dual-porosity 3D synthetic media generated with PoreSpy [@gostick2019porespy] across controlled detector blur and noise regimes, comparing 3D Nearest, Trilinear, and Tricubic B-spline interpolation against `SRResNet`, single-axis `OpenLSR-X`, and 3-axis Fourier-fused `OpenLSR-X`.
2. **Open Rock Micro-CT Study:** Evaluates super-resolution on Bentheimer Sandstone and bimodal Indiana Limestone micro-CT benchmarks, exporting publication-ready `booktabs` LaTeX tables and multi-panel error and pore-radius distribution figures.

# Acknowledgements

OpenLSR-X originates from the doctoral research of Calum Green (including synchrotron XCT experiments conducted at Diamond Light Source Beamline I13) and is maintained and extended by BASE Laboratory.

# References
