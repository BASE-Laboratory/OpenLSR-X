# Benchmarks, Petrophysics & Pore-Scale Transport

Standard computer-vision metrics (PSNR and 2D SSIM) can favor overly smooth reconstructions that pinch off narrow pore throats and corrupt permeability and diffusivity predictions. OpenLSR-X pairs image-fidelity metrics with **porous-media petrophysical, topological, and 3D pore-scale transport physics diagnostics**.

---

## Implemented Metrics

### 1. Image-Fidelity Metrics (`openlsrx.metrics.image`)
- **PSNR (dB):** Volumetric Peak Signal-to-Noise Ratio.
- **2D Slice SSIM & 3D Tri-Planar SSIM:** Structural Similarity averaged across `yz`, `xz`, and `xy` planes.
- **MS-SSIM, MAE, and RMSE:** Multi-scale structural similarity (for slices $\ge 161$ px) and pixelwise error norms.

### 2. Spectral Agreement Metrics (`openlsrx.metrics.spectral`)
- **3D Fourier Shell Correlation (FSC) & 2D Fourier Ring Correlation (FRC):** Hann-apodized, window-weighted zero-DC radial spectral correlation (`rfftn`-multiplicity weighted, excluding $k=0$).
- **Prediction-vs-Reference `FSC = 0.5` Crossover (`fsc_cutoff_050_cyc_vox`, `fsc_half_pitch_050_um`) & Spectral AUC (`fsc_auc`):** Reports the spatial frequency where prediction-to-reference correlation drops below `0.5`—a blur-sensitive agreement metric appropriate for full-reference super-resolution (unlike cryo-EM `0.143` half-map thresholds, which assume two independent noisy reconstructions).

### 3. Petrophysical & Topological Metrics (`openlsrx.metrics.petrophysics`)
- **Reference-Locked Thresholding (`threshold_true`):** Headline petrophysical and transport metrics apply the ground-truth HR threshold (`otsu`, `yen`, or `fixed`) to both `pred_vol` and `true_vol` so greyscale gain/offset bias is exposed rather than hidden (with independent `*_auto_otsu` diagnostics reported separately).
- **Porosity Error ($|\Delta\phi|$), Pore-Phase Dice & IoU:** Segmented pore volume fraction error and voxel-wise overlap.
- **Specific Surface Area Error ($S_v$) & Euler Characteristic Density ($\chi_V$):** Interfacial pore-grain surface area via 3D marching cubes and topological connectivity per unit volume.
- **Inscribed Pore-Radius Wasserstein-1 Distance ($W_1$) & Non-Periodic Two-Point Correlation RMSE ($S_2(r)$):** Exact 3D Euclidean distance transform pore-size distributions and 8-octant zero-padded non-periodic `rfftn` radial autocorrelation.
- **Through-Plane Anisotropy Error:** Measures slice-stacking gradient discontinuities along the non-trained axis relative to ground-truth structural anisotropy.

### 4. 3D Pore-Scale Transport Physics (`openlsrx.metrics.transport`)
- **Residual-Verified Directional Effective Diffusivity ($D_{\text{eff}}/D_0$) & Tortuosity ($\tau_z, \tau_y, \tau_x$):** Solves steady-state 3D Fickian Laplace diffusion ($\nabla \cdot \nabla c = 0$) across all three Cartesian directions on the 6-connected percolating pore network using a cell-centered finite-volume discretization matching `OpenImpala` and `TauFactor`, verifying `cg` convergence (`info == 0`), relative residual norm, inlet/outlet flux conservation, and variational bounds ($0 \le D_{\text{eff}}/D_0 \le \phi_{\text{perc}}$, $\tau \ge 1$).
- **View-Disagreement Transport Sensitivity Bounds (`compute_tortuosity_uncertainty_bounds`):** Evaluates `[D_eff^-, D_eff^+]` under $\hat{V} \pm k\sigma_{\text{ortho}}$ alongside voxel-wise Spearman rank correlation between $\sigma_{\text{ortho}}$ and $|\hat{V} - V_{\text{HR}}|$.
- **Simulated Capillary Drainage Curve RMSE ($S_{\text{nwp}}(P_c)$):** Simulates inlet-connected morphological drainage via `PoreSpy` to quantify throat-entry capillary pressure errors.

---

## Direct `OpenImpala` Simulation Bridge (`openlsrx.io.openimpala`)

For full-volume HPC transport simulations on `AMReX` grids using [`BASE-Laboratory/OpenImpala`](https://github.com/BASE-Laboratory/OpenImpala), export any super-resolved volume directly from the CLI or Python API:

```bash
openlsrx export-openimpala \
  --volume outputs/predicted_orthogonal/predicted_stack.tiff \
  --output-dir outputs/openimpala_ready \
  --name zn13x_sr \
  --direction x
```

This writes both an uncompressed 8-bit phase-labeled 3D TIFF (`zn13x_sr.tif`) and a ready-to-run AMReX `ParmParse` input deck (`zn13x_sr_openimpala.inputs`). You can also combine transport evaluation and export in a single step via `openlsrx evaluate --include-transport --export-openimpala-dir outputs/openimpala_ready`.

---

## Cell-Centered 3D Orthogonal Fourier Fusion & Model Backbones

Given an isotropic low-resolution 3D scan $V_{\text{LR}} \in \mathbb{R}^{d \times h \times w}$ and a 2D or `2.5D` generator $G_\theta$:

1. OpenLSR-X generates three cell-centered candidate super-resolved 3D volumes ($\hat{V}_{yz}, \hat{V}_{xz}, \hat{V}_{xy} \in \mathbb{R}^{D \times H \times W}$) via `zoom_cell_centered` (`grid_mode=True`) along each slicing axis and native-LR-strided `2.5D` context windows across the two orthogonal in-plane dimensions.
2. In 3D Fourier space (`rfftn` in streaming `float32`/`complex64`), each candidate volume is weighted proportionally to its two super-resolved in-plane wavevector components (`mode="fourier"`), optionally preceded by spatial Winsorization of single-view out-of-plane blur outliers (`mode="confidence_fourier"`).

OpenLSR-X includes four neural model families for controlled ablation:
- **Shifted-Window `SwinIRGenerator` (`2D` / `2.5D`)** (`backbone="swinir"`): Windowed self-attention with native-LR-strided multi-slice context (`context_slices=3`) and multi-axis training (`train_axes=("yz", "xz", "xy")`).
- **Classical `SRGAN` / `SRResNet`** (`backbone="srgan"`): Residual blocks with optional pixel/instance/batch normalization.
- **Normalization-Free `RRDBGenerator` + `RaGAN`** (`backbone="rrdb"`, `gan_loss_type="ragan"`): Preserves greyscale X-ray attenuation contrasts and supports arbitrary fractional beamline magnification ratios (e.g., $2.5\times$ and $5\times$).
- **Native 3D CNN Baseline (`SR3DResNet`)**: Volumetric 3D residual network with 48-element $O_h$ cubic symmetry augmentation and raised-cosine sliding-window inference (`super_resolve_volume_3d`).

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
