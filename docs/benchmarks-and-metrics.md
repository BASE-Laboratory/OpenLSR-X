# Benchmarks & Petrophysical Metrics

Standard computer-vision metrics (PSNR and 2D SSIM) can favor overly smooth reconstructions that pinch off narrow pore throats and corrupt permeability predictions. OpenLSR-X pairs image-fidelity metrics with **porous-media petrophysical and topological diagnostics**.

---

## Implemented Metrics

### 1. Image-Fidelity Metrics (`openlsrx.metrics.image`)
- **PSNR (dB):** Volumetric Peak Signal-to-Noise Ratio.
- **2D Slice SSIM & 3D Tri-Planar SSIM:** Structural Similarity averaged across `yz`, `xz`, and `xy` planes.
- **MS-SSIM, MAE, and RMSE:** Multi-scale structural similarity (for slices $\ge 161$ px) and pixelwise error norms.

### 2. Petrophysical & Topological Metrics (`openlsrx.metrics.petrophysics`)
- **Porosity Error ($|\Delta\phi|$):** Absolute and relative error in segmented pore volume fraction under Otsu, Yen, or reference-locked HR thresholds.
- **Pore-Phase Dice & IoU:** Voxel-wise overlap of the segmented pore network.
- **Specific Surface Area Error ($S_v$):** Interfacial pore-grain surface area per bulk volume extracted via 3D marching cubes.
- **Euler Characteristic Density ($\chi_V$):** Topological connectivity of the 3D pore space per unit volume.
- **Inscribed Pore-Radius Wasserstein-1 Distance ($W_1$):** Earth-mover distance between exact 3D Euclidean distance transform (EDT) pore-size distributions.
- **Two-Point Correlation RMSE ($S_2(r)$):** Discrepancy in FFT-computed radially averaged two-point probability functions.
- **Through-Plane Anisotropy Error:** Measures slice-stacking gradient discontinuities along the non-trained axis relative to ground-truth structural anisotropy.

---

## 3D Orthogonal Fourier Stack Fusion

Given an isotropic low-resolution 3D scan $V_{\text{LR}} \in \mathbb{R}^{d \times h \times w}$ and a 2D generator $G_\theta$ trained only on slices from a single plane:

1. OpenLSR-X generates three candidate super-resolved 3D volumes ($\hat{V}_{yz}, \hat{V}_{xz}, \hat{V}_{xy} \in \mathbb{R}^{D \times H \times W}$) by cubic-interpolating along each respective slicing axis and applying $G_\theta$ across the two orthogonal in-plane dimensions.
2. In 3D Fourier space, each candidate volume is weighted proportionally to its two super-resolved in-plane wavevector components:
   $$W_a(\mathbf{k}) = \frac{\sum_{j \neq a} k_j^2 + \epsilon}{2\,(k_x^2 + k_y^2 + k_z^2) + 3\epsilon}$$
   suppressing low-pass interpolation blur and slice-stacking artifacts along the non-super-resolved axis of each individual pass.

---

## Running the Reproducible Benchmark Suite

1. **Factorial PoreSpy Study** (`Low_Noise`/`High_Noise` × `Low_Feature`/`High_Feature`):
   ```bash
   python scripts/benchmarks/porespy_study.py --output-dir results/porespy_study
   ```
2. **Open Rock Micro-CT Study** (`bentheimer_sandstone` & `indiana_limestone`):
   ```bash
   python scripts/benchmarks/public_rock_xct.py --output-dir results/public_rock
   ```
3. **Generate LaTeX Tables & Figures:**
   ```bash
   python scripts/benchmarks/make_paper_outputs.py --out-dir results/paper_artifacts
   ```
   Outputs `booktabs`-formatted `.tex` tables, `.md` summary tables, and 300-DPI `.png`/`.pdf` orthogonal comparison figures with pore-radius distributions.
