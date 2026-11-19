# Analytical & Visual Verification Suite

This page documents the reproducible verification suite included in `scripts/verification/run_verification_suite.py` and enforced in CI via `tests/test_analytical_verification.py`. Each core function in `openlsrx` is checked against **closed-form analytical ground truth**, **explicitly labeled 3D phantoms**, **real 6 µm Bentheimer Sandstone micro-CT scans (`Zenodo 5542624`) under controlled synthetic 2× degradation**, and **physical independent dual-scan `18 µm → 6 µm` (`3×`) Bentheimer Sandstone micro-CT pairs (`Zenodo 5542624`)**.

To regenerate every figure and the machine-readable report (`docs/assets/verification/verification_summary.json`) locally:

```bash
# Fast closed-form analytical pytest checks (~30 s on CPU)
pytest tests/test_analytical_verification.py

# Full visual + multi-seed Bentheimer (synthetic 2x + physical dual-scan 3x) verification suite
python scripts/verification/run_verification_suite.py \
  --output-dir docs/assets/verification \
  --include-bentheimer \
  --include-bentheimer-dualscan
```

---

## 1. Data Generation, Cell-Centered Alignment, Split Isolation & Registration (`openlsrx.data`)

![Subsystem 1 — Cell-centered alignment, PoreSpy generation, 2.5D strided slice extraction, PSF-guarded spatial splitting, and sub-voxel 3D registration](assets/verification/vr_01_data_and_alignment.png)

### 1.1 `zoom_cell_centered` — Eliminating Half-Block Depth Shift

Block-averaging an HR grid of spacing `Δx_HR = 1` by factor `s` places coarse voxel `k` at coordinate `(k + 0.5)*s - 0.5`. Legacy `scipy.ndimage.zoom(..., grid_mode=False)` pins corner voxels (`0` and `N - 1`), introducing a spatially varying depth shift of up to `±(s - 1)/2` HR voxels. `zoom_cell_centered` (`openlsrx.data.transforms`) uses `grid_mode=True` with boundary-preserving `'nearest'` extension so coarse and fine voxel centers remain aligned across `2×`, `3×`, `4×`, and `8×`.

| Scale Factor `s` | Analytical Corner-Pinned Limit `(s - 1) / 2` | Legacy `grid_mode=False` Max Shift (HR vox) | `zoom_cell_centered` Max Shift (HR vox) |
| :---: | :---: | :---: | :---: |
| `2×` | `0.5000` | `±0.4683` | **`±0.000004`** (`4e-6`) |
| `3×` | `1.0000` | `±0.9032` | **`±0.000004`** (`4e-6`) |
| `4×` | `1.5000` | `±1.3095` | **`±0.000008`** (`8e-6`) |
| `8×` | `3.5000` | `±2.6111` | **`±0.000023`** (`2.3e-5`) |

### 1.2 `generate_porespy_volume` and `degrade_xct_volume`

`generate_porespy_volume` synthesizes correlated 3D two-phase porous media (`float32` in `[0, 1]`), and `degrade_xct_volume` applies 3D Gaussian PSF blur, exact `s³` block-averaging, and detector noise (`(64, 64, 64)` HR → `(32, 32, 32)` LR at `scale_factor=2`).

### 1.3 `extract_paired_slices_from_subvolume` and `build_multi_axis_slice_dataset`

Extracts cell-centered `(LR, HR)` slice pairs across `("yz", "xz", "xy")`. In `2.5D` mode (`context_slices = 2k + 1`), neighboring slices are sampled at native coarse-grid spacing (`±Δz_LR`) rather than cubically blended sub-voxel steps, producing `lr_ctx` of shape `(160, 3, 32, 32)` and `hr_slc` of shape `(160, 64, 64)` on a `(16, 32, 32)` LR / `(32, 64, 64)` HR slab.

### 1.4 `minimum_psf_guard_slices_lr`, `split_volume_spatially`, `compute_normalization_bounds`, and `fit_cross_scan_calibration`

Enforces 3D slab isolation by purging `ceil((4 * psf_sigma + 2) / s)` LR slices (`2` LR / `4` HR slices at `psf_sigma=1.0, s=2`) at both `train|val` and `val|test` boundaries (`Train=36`, `Guard=4`, `Val=8`, `Guard=4`, `Test=12` HR slices), fits percentile normalization bounds (`[0.1360, 0.8566]`) strictly on `train`, and optionally fits an `affine` or monotonic `quantile` cross-scan radiometric map (`fit_cross_scan_calibration`) strictly on `train` so mutating held-out `test` voxels changes fitted knot values by `0.0`.

### 1.5 `register_multires_pair_3d` and Provenance-Guarded Loaders

- **Half-LR-voxel 3D shift recovery:** Injecting a `(+3, -2, +4)` HR voxel translation (`+1.5, -1.0, +2.0` LR voxels at `2×`) into a `Zn-13X` phantom pair, `register_multires_pair_3d` recovers the exact `(-3, +2, -4)` HR shift and raises `triplanar_ssim` from `0.2176` to **`0.7330`**.
- **Strict offline provenance:** Calling `prepare_zn13x_benchmark(..., prefer_offline=True)` or `prepare_bentheimer_dualscan_benchmark(..., prefer_offline=True)` sets `source_kind = "offline_calibrated_*"`, `is_experimental_xct = False`, and null Zenodo DOIs/URLs.

---

## 2. Topology, Gradient, Spectral & Approximate Conservation Losses (`openlsrx.losses`)

![Subsystem 2 — Soft skeletonization, SoftCLDiceLoss vs L1 on severed pore throats, SobelGradientLoss, FocalFrequencyLoss, and approximate slice/2.5D conservation loss](assets/verification/vr_02_losses_diagnostics.png)

### 2.1 `soft_skeletonize` and `SoftCLDiceLoss` — Detecting Severed Pore Throats

Blocking a narrow throat (`23 / 4096` pixels, `0.56%` of slice area) between two pore chambers disconnects flow (`D_eff = 0`) while modifying far fewer pixels than mild uniform noise:

| Perturbation Applied to Ground Truth | Modified Pixels | Pixel `L1` Loss | `SoftCLDiceLoss` | Connectivity |
| :--- | :---: | :---: | :---: | :---: |
| Mild uniform noise (`σ = 0.025`), throat open | `4096 / 4096` (`100%`) | `0.01254` | **`0.00724`** | Connected |
| Severed central throat (`23` pixels blocked) | `23 / 4096` (`0.56%`) | `0.00449` (`2.8×` lower `L1`) | **`0.40745`** (**`56.3×` higher**) | Disconnected |

### 2.2 `AttenuationConservationLoss`, `SobelGradientLoss`, and `FocalFrequencyLoss`

!!! note "Approximate Slice/2.5D Consistency vs. Exact 3D Conservation"
    `AttenuationConservationLoss` operates on 2D HR slice predictions paired with `1`-slice (`2D`) or `3`-slice (`2.5D`) LR inputs; it is therefore an **approximate slice/slab consistency regularizer**, not an exact 3D tomographic conservation constraint. When `2.5D` context (`C >= 3`) is provided, a local discrete `z`-Laplacian term (`L_{z-1} - 2 L_z + L_{z+1}`) reduces second-order through-plane PSF curvature bias on smooth profiles (`0.04412` → `0.02406`, `-45.5%` on a quadratic `z` ramp), whereas exact 3D block-average conservation only holds on full 3D volumes under matched LR/HR intensity calibration.

Across Gaussian blur widths `σ ∈ [0, 3.0]` HR voxels, `SobelGradientLoss`, `FocalFrequencyLoss`, `SoftCLDiceLoss`, and `AttenuationConservationLoss` all evaluate to `0.0` at `σ = 0` and increase strictly monotonically with degradation width.

---

## 3. Baselines, Neural Backbones & Orthogonal Fourier Fusion (`openlsrx.baselines`, `openlsrx.models`, `openlsrx.inference`)

### 3.1 Multi-Backbone Comparison on Synthetic & Calibrated 3D Phantoms (`2×`)

![Subsystem 3 — Multi-dataset super-resolution verification on Synthetic PoreSpy, Calibrated Indiana Limestone Phantom, and Calibrated 3-Phase Zn-13X Phantom](assets/verification/vr_03_models_and_fusion.png)

All three datasets in Figure 3 are **synthetic or calibrated 3D phantoms** (`8–10` CPU training epochs on the PSF-guarded `train` slab, evaluated on the unseen `test` slab):

- **Dataset A (`Synthetic PoreSpy 3D Phantom`, `generate_porespy_volume`):** Connected two-phase blobs (`16 × 48 × 48` HR test slab).
- **Dataset B (`Calibrated Indiana Limestone Phantom`, `source_kind="calibrated_carbonate_phantom"`):** Bimodal vuggy carbonate phantom (`PSNR`: `Nearest 20.03 dB` → `Tricubic 21.06 dB` → `SwinIR-2.5D-3Axis 22.09 dB`).
- **Dataset C (`Calibrated 3-Phase Zn-13X Phantom`, `source_kind="offline_calibrated_zn13x_phantom"`, `is_experimental_xct=False`):** `32³ → 64³` 3-phase phantom (`Tricubic` → `SwinIR-2.5D-3Axis`: `PSNR` `15.46 → 16.62 dB`, 3-phase macro-Dice `0.5576 → 0.6721`, macropore Dice `0.4833 → 0.7548`).

| Method (`Dataset A: Synthetic PoreSpy 3D`) | PSNR (dB) ↑ | 3D Triplanar SSIM ↑ | Pore Dice ↑ | Abs. Porosity Err ↓ | Through-Plane Anisotropy Err ↓ | Optical MTF_0.5 (cyc/vox) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `upsample_3d(..., "nearest")` | `15.19` | `0.6309` | `0.7905` | `0.1122` | `0.0487` | `0.1757` |
| `upsample_3d(..., "tricubic")` | `16.23` | `0.6958` | `0.8231` | `0.1162` | `0.0363` | `0.1876` |
| `SR3DResNet` (Native 3D CNN) | `16.48` | `0.7182` | `0.8134` | `0.1175` | `0.0469` | `0.1962` |
| `SRGAN` (`SRResNet-3Axis`) | `16.19` | `0.7334` | `0.8233` | `0.0912` | `0.0924` | `0.2265` |
| `SRGAN` (`RRDB-3Axis`) | **`18.34`** | **`0.8175`** | **`0.8897`** | **`0.0705`** | `0.0174` | **`0.2603`** |
| `super_resolve_volume_single_axis` (`SwinIR yz`) | `18.25` | `0.8123` | `0.8846` | `0.0765` | `0.0465` | `0.2522` |
| `super_resolve_volume_orthogonal` (`SwinIR-2.5D-3Axis`) | `18.11` | `0.8064` | `0.8810` | `0.0782` | **`0.0015`** (`31.0×` lower) | `0.2504` |

### 3.2 Real `6 µm` Bentheimer Micro-CT (`Zenodo 5542624`) Under Synthetic `2×` Degradation

![Real 6 um Bentheimer Micro-CT with Synthetic 2x Degradation](assets/verification/vr_05_bentheimer_synth_deg.png)

!!! important "Provenance & Scope of the Synthetic-Degradation Bentheimer Check"
    - **Real HR scan, synthetic `2×` LR degradation:** The `6 µm` HR volumes (`Core1_Subvol1_LR.tif` and `Core2_Subvol2_LR.tif`, `Zenodo 5542624`, Jackson et al., 2021 — named `_LR` in their `6 µm → 2 µm` study) are real experimental X-ray micro-CT scans, whereas the `12 µm` LR inputs here are generated via `degrade_xct_volume` (`degrade_from_hr=True`, `psf_sigma=1.0`, `noise_std=0.02`). A `(0, 0, 0)` HR-voxel shift on this pair therefore verifies **zero algorithmic coordinate drift** through synthetic degradation and cell-centered fusion, rather than physical registration of two independent scans.
    - **Sample size & convergence:** The table below reports `mean ± SD` across `3` seeds (`42, 43, 44`) on two distinct `96 × 80 × 80` Bentheimer cores (`22 × 80 × 80` held-out test slabs) after a compact `8`-epoch CPU training budget (`~35 s/run`).

| Core (`Zenodo 5542624`, Real `6 µm` HR + Synthetic `2×` LR, `n = 3` seeds) | Tricubic PSNR (dB) | `SwinIR-3Axis` PSNR (dB) | Tricubic `D_eff` Rel. Err (%) | Single-Axis `yz` `D_eff` Rel. Err (%) | `3-Axis` `D_eff` Rel. Err (%) | Voxel `σ_ortho` vs Error Spearman `r_s` | `±1 σ_ortho` Envelope Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bentheimer_sandstone` (Core 1 Subvol 1, `D_eff = 0.0396`) | `23.48 ± 0.00` | `25.71 ± 0.14` | `17.44 ± 0.40%` | `14.59 ± 1.23%` | **`14.08 ± 1.53%`** | `+0.264 ± 0.014` (weak positive) | `0 / 3` |
| `bentheimer_laminated` (Core 2 Subvol 2, `D_eff = 0.0765`) | `25.00 ± 0.03` | `26.99 ± 0.11` | `6.51 ± 0.14%` | **`4.05 ± 1.54%`** | `4.10 ± 2.13%` | `+0.227 ± 0.013` (weak positive) | `3 / 3` |

Across all `6/6` core-seed runs, both `Single-Axis SwinIR (yz)` and `3-Axis SwinIR-2.5D` reduce `D_eff` relative error below `Tricubic-3D` (`3-Axis` further improves over single-axis `yz` on all `3/3` Core 1 runs and matches `yz` within `1 SD` on Core 2, even though `3-Axis` PSNR is `0.1–0.3 dB` lower than single-axis `yz`). Inter-view standard deviation `σ_ortho` (`n = 3` orthogonal sweeps) exhibits a **weak but consistently positive** Spearman rank correlation (`r_s = 0.23–0.26`) with voxelwise error `|SR - HR|`, and its `±1 σ_ortho` threshold shift brackets true `D_eff` in `3/6` short-epoch runs (`0/3` on Core 1 where `8`-epoch bias exceeds `1 σ_ortho`, and `3/3` on Core 2).

### 3.3 Physical Independent Dual-Scan `18 µm → 6 µm` (`3×`) Bentheimer Micro-CT (`Zenodo 5542624`)

![Physical Independent Dual-Scan Bentheimer Micro-CT (Zenodo 5542624, True 18 um -> 6 um, 3x)](assets/verification/vr_06_bentheimer_dualscan_3x.png)

To verify `openlsrx` on **two physically independent X-ray micro-CT acquisitions** at an odd integer magnification (`3×`, corresponding to a `27×` voxel-volume upsampling ratio), `prepare_bentheimer_dualscan_benchmark` loads the MD5-verified 16-bit little-endian `.raw` scan pairs from `Zenodo 5542624` (`Core1_Subvol1` and `Core2_Subvol2`, `75³` at `18 µm` paired with `225³` at `6 µm`; `source_kind="downloaded_zenodo_5542624_dualscan"`, `is_experimental_xct=True`) and slices each linear `uint16 / 65535.0` pair on the integer HR voxel grid to exact `(0, 0, 0)` alignment (`Core 1` raw `[-1, 0, 0]` → staged `[0, 0, 0]`, `Core 2` `[0, 0, 0]`).

- **3D registration diagnostics:** On `(36, 32, 32)` LR → `(108, 96, 96)` HR subvolumes, `register_multires_pair_3d(..., scale_factor=3)` confirms sub-voxel hardware alignment (`hr_shift_zyx = [-1, 0, 0]`, `hr_triplanar_ssim = 0.6501` on Core 1; `hr_shift_zyx = [0, 0, 0]`, `hr_triplanar_ssim = 0.6124` on Core 2; `registration_quality_ok = True`).
- **Split-isolated cross-scan radiometric calibration:** Independent polychromatic full-core (`18 µm`) vs. interior-ROI (`6 µm`) reconstructions exhibit non-linear beam-hardening greyscale shifts. Fitting `cross_scan_calibration="quantile"` strictly on the `train` slab lifts held-out `test` (`27 × 96 × 96` HR) `Tricubic-3D` PSNR by **`+1.46 dB`** (`14.90 → 16.36 dB`, 3D SSIM `0.3547 → 0.4502`) on Core 1 and **`+0.70 dB`** (`16.52 → 17.22 dB`, 3D SSIM `0.3758 → 0.4452`) on Core 2 with zero `val`/`test` leakage.
- **Native `PixelShuffle(3)` `SwinIR-2.5D` (`10` CPU epochs, `n = 3` seeds):** Training `SwinIR-2.5D` (`ArbitraryScaleUpsample(scale_factor=3)` using native `nn.PixelShuffle(3)`) further improves unseen `test` PSNR, 3D SSIM, and pore Dice over calibrated `Tricubic-3D` across all `6/6` runs (`16.55 ± 0.01 dB` on Core 1; `17.45 ± 0.07 dB` and `0.0039 ± 0.0035` porosity error on Core 2), while orthogonal view disagreement `σ_ortho` correlates positively with true voxel error `|SR - HR|` (**`r_s = +0.383 ± 0.007`** on Core 1, **`+0.278 ± 0.008`** on Core 2).

| Core (`Zenodo 5542624`, True `18 µm → 6 µm` `3×` Dual-Scan, `n = 3` seeds, `10` CPU epochs) | Raw Tricubic PSNR (dB) | Calibrated Tricubic PSNR (dB) | `SwinIR-3Axis` PSNR (dB) | Cal. Tricubic → `3-Axis` 3D SSIM | Cal. Tricubic → `3-Axis` Pore Dice | Cal. Tricubic → `3-Axis` `D_eff` Rel. Err (%) | `σ_ortho` vs Error `r_s` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bentheimer_dualscan_core1` (`D_eff = 0.0392`, `27×96×96` HR) | `14.90 ± 0.00` | `16.36 ± 0.00` | **`16.55 ± 0.01`** | `0.4502` → **`0.4637 ± 0.0016`** | `0.7658` → **`0.7735 ± 0.0008`** | `70.19%` → `68.58 ± 0.92%` (`yz`: `67.32 ± 0.69%`) | **`+0.383 ± 0.007`** |
| `bentheimer_dualscan_core2` (`D_eff = 0.0638`, `27×96×96` HR) | `16.52 ± 0.00` | `17.22 ± 0.00` | **`17.45 ± 0.07`** | `0.4452` → **`0.4629 ± 0.0025`** | `0.7947` → **`0.7991 ± 0.0007`** | **`6.59%`** → `14.67 ± 3.21%` (`yz`: `14.38 ± 3.35%`) | **`+0.278 ± 0.008`** |

!!! warning "Restrained Interpretation of Short-CPU-Epoch `3×` Dual-Scan Transport"
    At `18 µm → 6 µm` (`27×` voxel-volume upsampling) on thin `27 × 96 × 96` (`162 µm` thick) slabs under a `10`-epoch CPU verification budget, sub-`18 µm` pore throats near percolation on Core 1 remain heavily blurred (`67.3–68.6%` `D_eff` error vs. `70.2%` Tricubic), while on Core 2 smooth calibrated `Tricubic-3D` happens to cancel partial-volume throat dilation/erosion at the HR Otsu threshold (`6.59%` `D_eff` error vs. `14.67%` for `10`-epoch `SwinIR-3Axis`, even though `SwinIR-3Axis` improves PSNR, 3D SSIM, pore Dice, and cuts bulk porosity error `4.2×` from `0.0163` to `0.0039`). This contrast illustrates why `OpenLSR-X` evaluates 3D Fickian diffusivity directly rather than treating PSNR, SSIM, or bulk porosity as surrogates for pore-throat transport; full GPU training on full `75³ → 225³` dual-scan volumes is configured in `configs/labxct/bs_dualscan_3x_swinir_25d.yaml`.

---

## 4. Closed-Form Analytical Verification of Physics & Spectral Metrics (`openlsrx.metrics`, `openlsrx.io`)

![Subsystem 4 — Closed-form analytical verification of 3D Fickian diffusion, zero-flux dead-end cavities, optical MTF vs shell FSC, non-periodic S2(r), 3D Euler characteristic, and bimodal capillary drainage](assets/verification/vr_04_analytical_physics_benchmarks.png)

### 4.1 `solve_steady_state_diffusion_axis` and `compute_directional_tortuosity`

| Analytical Test Geometry | Porosity `φ` | Exact `D_eff` | Solver `D_eff` | Exact `τ` | Solver `τ` | `solver_converged` | Rel. Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Open Bulk Space (`32³`) | `1.0000` | `1.000000` | **`0.999999`** | `1.0000` | **`1.0000`** | `True` | **`0.0001%`** |
| 2. Parallel Straight Ducts along `z` | `0.2500` | `0.250000` (`D_x=0`) | **`0.250000`** (`D_x=0`) | `1.0000` | **`1.0000`** | `True` | **`0.0001%`** |
| 3. Straight Trunk + Dead-End Cavity | `0.1266` | `0.06250` (1D trunk) | **`0.06267`** | `2.0260` | **`2.0206`** | `True` | **`0.27%`** |
| 4. Maxwell Insulating Sphere (`R = 6 / 40³`) | `0.9858` | `0.97878` | **`0.97663`** | `1.0071` | **`1.0093`** | `True` | **`0.22%`** |
| 5. Maxwell Insulating Sphere (`R = 9 / 40³`) | `0.9514` | `0.92879` | **`0.92401`** | `1.0243` | **`1.0296`** | `True` | **`0.52%`** |
| 6. Maxwell Insulating Sphere (`R = 11 / 40³`) | `0.9123` | `0.87391` | **`0.86690`** | `1.0439` | **`1.0523`** | `True` | **`0.80%`** |

In Case 3 (`Trunk + Dead-End Cavity`), the attached dead-end cavity carries zero steady-state flux (`∇c = 0` inside the pocket) and inlet/outlet flux conservation holds to **`|J_in - J_out| / J_mean = 2.61e-9`**.

### 4.2 `compute_fsc_3d` — Effective Optical MTF vs. Shell FSC

Convolving a 3D white-noise volume with an isotropic Gaussian PSF yields the exact analytical modulation transfer function `H(k) = exp(-2 * pi^2 * sigma^2 * k^2)` with half-amplitude cutoff `k_0.5 = sqrt(ln(2) / 2) / (pi * sigma)`. Because shell `FSC(k)` normalizes by both shell magnitudes, any positive scalar blur leaves noiseless `FSC(k) == 1`, whereas `effective_mtf` recovers `H(k)` directly:

| Gaussian Blur `σ` (vox) | Exact `k_0.5` (`cyc/vox`) | Measured `cutoff_freq_mtf_050` | Full-Curve `MTF_eff` RMSE | Noiseless Shell `cutoff_freq_050` (`FSC`) |
| :---: | :---: | :---: | :---: | :---: |
| `0.8` | `0.2342` | **`0.2348`** (`0.26%` err) | **`0.00298`** | `0.5000` (insensitive to pure blur) |
| `1.4` | `0.1339` | **`0.1338`** (`0.07%` err) | **`0.00527`** | `0.5000` (insensitive to pure blur) |
| `2.2` | `0.0852` | **`0.0853`** (`0.12%` err) | **`0.01096`** | `0.3904` (triggers only near float floor) |

### 4.3 `compute_euler_density`, `two_point_correlation_radial`, `compute_drainage_curve`, and `export_for_openimpala`

- **3D Euler Characteristic (`compute_euler_density`):** Solid 3D sphere (`R = 10` in `48³`, exact `χ = +1`) measures **`+1.0000`**; solid 3D torus (`R = 12, r = 4.5` in `48³`, exact `χ = 0`) measures **`0.0000`**.
- **Zero-padded non-periodic `S_2(r)` (`two_point_correlation_radial`):** Uncorrelated 3D Bernoulli medium (`φ = 0.2979`) matches exact `S_2(0) = φ = 0.2979` and `S_2(r >= 1) = φ^2 = 0.0888` with **`RMSE = 0.00030`**.
- **Capillary drainage (`compute_drainage_curve`):** Parallel cylindrical tubes (`R_1 = 3`, `R_2 = 7` voxels) exhibit sharp saturation steps at exact Young–Laplace entry proxies `2 / R_2 = 0.286` and `2 / R_1 = 0.667`.
- **OpenImpala export (`export_for_openimpala`):** Writes a verified `uint8` 3D multi-page TIFF (`98,890` bytes) and companion AMReX `.inputs` deck (`eb2.geom_type = "tiff"`, `torteff.phase = 1`, `torteff.direction = "z"`).
