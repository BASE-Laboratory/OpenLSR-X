# Benchmarks, Petrophysics & Pore-Scale Transport

Standard computer-vision metrics (PSNR and 2D SSIM) can favor overly smooth reconstructions that pinch off narrow pore throats and corrupt permeability and diffusivity predictions. OpenLSR-X pairs image-fidelity metrics with **porous-media petrophysical, topological, and 3D pore-scale transport physics diagnostics**.

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

### 3. 3D Pore-Scale Transport Physics (`openlsrx.metrics.transport`)
- **Directional Effective Diffusivity ($D_{\text{eff}}/D_0$) & Tortuosity ($\tau_z, \tau_y, \tau_x$):** Solves steady-state 3D Fickian Laplace diffusion ($\nabla \cdot \nabla c = 0$) across all three Cartesian directions on the 6-connected percolating pore network using a cell-centered finite-volume discretization matching `OpenImpala` and `TauFactor`.
- **Formation Factor ($F = (D_{\text{eff}}/D_0)^{-1}$) & Directional Transport Anisotropy:** Detects artificial slice-stacking bottlenecks where pores percolate in-plane (`yz`) but pinch off along the through-plane stacking axis (`x`).
- **Simulated Capillary Drainage Curve RMSE ($S_{\text{nwp}}(P_c)$):** Simulates inlet-connected morphological mercury intrusion / drainage via `PoreSpy` to quantify throat-entry capillary pressure errors.

---

## Direct `OpenImpala` Simulation Bridge (`openlsrx.io.openimpala`)

For full-volume HPC transport simulations on `AMReX` grids using [`BASE-Laboratory/OpenImpala`](https://github.com/BASE-Laboratory/OpenImpala), export any super-resolved volume directly from the CLI or Python API:

```bash
openlsrx export-openimpala \
  --volume outputs/predicted_xy/predicted_stack.tiff \
  --output-dir outputs/openimpala_ready \
  --name zn13x_sr \
  --direction x
```

This writes both an uncompressed 8-bit phase-labeled 3D TIFF (`zn13x_sr.tif`) and a ready-to-run AMReX `ParmParse` input deck (`zn13x_sr_openimpala.inputs`). You can also combine transport evaluation and export in a single step via `openlsrx evaluate --include-transport --export-openimpala-dir outputs/openimpala_ready`.

---

## 3D Orthogonal Fourier Stack Fusion & Model Backbones

Given an isotropic low-resolution 3D scan $V_{\text{LR}} \in \mathbb{R}^{d \times h \times w}$ and a 2D generator $G_\theta$ trained only on slices from a single plane:

1. OpenLSR-X generates three candidate super-resolved 3D volumes ($\hat{V}_{yz}, \hat{V}_{xz}, \hat{V}_{xy} \in \mathbb{R}^{D \times H \times W}$) by cubic-interpolating along each respective slicing axis and applying $G_\theta$ across the two orthogonal in-plane dimensions.
2. In 3D Fourier space, each candidate volume is weighted proportionally to its two super-resolved in-plane wavevector components:
   $$W_a(\mathbf{k}) = \frac{\sum_{j \neq a} k_j^2 + \epsilon}{2\,(k_x^2 + k_y^2 + k_z^2) + 3\epsilon}$$
   suppressing low-pass interpolation blur and slice-stacking artifacts along the non-super-resolved axis of each individual pass.

OpenLSR-X includes three neural model families for controlled ablation:
- **Classical `SRGAN` / `SRResNet`** (`backbone="srgan"`): Residual blocks with optional pixel/instance/batch normalization.
- **Normalization-Free `RRDBGenerator` + `RaGAN`** (`backbone="rrdb"`, `gan_loss_type="ragan"`): Preserves absolute synchrotron X-ray attenuation contrasts and supports arbitrary fractional beamline magnification ratios (e.g., $2.5\times$ for $0.8125 \to 0.325\ \mu\text{m}$ and $5\times$ for $1.625 \to 0.325\ \mu\text{m}$).
- **Native 3D CNN Baseline (`SR3DResNet`)**: Volumetric 3D residual network with raised-cosine sliding-window inference (`super_resolve_volume_3d`) for benchmarking GPU memory footprint vs. 2D + 3-Axis Fourier Fusion.

---

## Running the Reproducible Benchmark Suite & HPC Campaign

1. **Factorial PoreSpy Study** (`Low_Noise`/`High_Noise` × `Low_Feature`/`High_Feature`):
   ```bash
   python scripts/benchmarks/porespy_study.py --output-dir results/porespy_study
   ```
2. **Open Rock Micro-CT Study** (`bentheimer_sandstone` & `indiana_limestone`):
   ```bash
   python scripts/benchmarks/public_rock_xct.py --output-dir results/public_rock
   ```
3. **Diamond Light Source I13-2 Multi-Resolution `Zn-13X` Synchrotron & Transport Study** ($2\times$ & $8\times$):
   ```bash
   python scripts/benchmarks/zn13x_synchrotron_study.py --output-dir results/zn13x_study
   ```
4. **Isambard-AI (`GH200`) Multi-Seed SLURM Array & Campaign Aggregator:**
   ```bash
   sbatch scripts/hpc/isambard_ai_array.slurm
   python scripts/benchmarks/aggregate_campaign.py --inputs results/campaign --output-dir results/campaign_summary
   ```
   Generates multi-seed `mean ± std` `booktabs` LaTeX tables (`campaign_table.tex`), Markdown summaries (`campaign_summary.md`), and PSNR-vs-transport Pareto plots (`campaign_tradeoffs.png`).
