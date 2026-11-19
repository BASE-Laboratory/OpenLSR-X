# Datasets & Configuration

OpenLSR-X provides two composable PyTorch Lightning `DataModule` classes selectable in YAML via `data.class_path`:

1. `openlsrx.data.SyntheticPoreSpyDataModule` — Deterministic 3D porous-media generation across noise and feature-scale regimes.
2. `openlsrx.data.PairedVolumeDataModule` — File-based loader supporting 3D multi-page `.tif`/`.tiff` stacks, directories of 2D slices, `.npy` arrays, and synchrotron `.h5`/`.hdf5`/`.nxs` files.

---

## Split-Isolated 3D Spatial Splitting & Normalization

When evaluating 2D/`2.5D`-trained models on orthogonal 3D XCT planes, naive random 2D slice splits leak structural information across axes because every `xy` or `xz` plane intersects all `yz` slices of the same cube, and whole-volume intensity normalization or PSF blurring leaks `test` statistics into `train`.

Both DataModules partition the 3D volume along `split_axis` into **strictly non-overlapping 3D slabs** (`train`, `val`, `test`) separated by `guard_slices_lr` buffer slices *before* fitting intensity normalization bounds (on `train` only), *before* fitting optional cross-scan radiometric calibration (`cross_scan_calibration: "none" | "affine" | "quantile"` mapping `LR_train` to PSF-matched `HR_train` with zero `val`/`test` leakage), and *before* applying per-split synthetic PSF degradation, then extract cell-centered (`grid_mode=True`) 2D or native-LR-strided `2.5D` slice pairs.

---

## Dataset Profiles Included in `configs/`

| Config Path | Target Dataset | Description |
| :--- | :--- | :--- |
| `configs/base/config.yaml` | Synthetic PoreSpy | Quickstart $128^3$ dual-porosity attenuation benchmark |
| `configs/base/srresnet_ablation.yaml` | Synthetic PoreSpy | Non-adversarial L1/CAWR generator ablation |
| `configs/porespy/Low_Noise/*.yaml` | Factorial PoreSpy | Low acquisition noise × Low/High feature frequency (including `ps_rrdb_ragan_config.yaml`) |
| `configs/porespy/High_Noise/*.yaml` | Factorial PoreSpy | High acquisition noise × Low/High feature frequency |
| `configs/labxct/bs_config.yaml` | Bentheimer Sandstone | Well-sorted quartz sandstone (~22% intergranular porosity, Zenodo `5542624`, synthetic $2\times$ LR) |
| `configs/labxct/bs_dualscan_3x_swinir_25d.yaml` | Bentheimer Dual-Scan ($3\times$) | Physical independent $18\ \mu\text{m} \to 6\ \mu\text{m}$ ($75^3 \to 225^3$, Zenodo `5542624`) `2.5D SwinIR` with `PixelShuffle(3)` + `train`-fitted quantile calibration |
| `configs/labxct/il_config.yaml` | Indiana Limestone | Calibrated bimodal vuggy + microporous carbonate phantom (~18% total porosity) |
| `configs/i13-xct/certified_regions.yaml` | Diamond I13 (`Zn13X` 7-DOF) | Canonical Stage 0B 7-DOF similarity transforms & calibrated voxel spacings from `BASE-Laboratory/MultimodalBenchmark` |
| `configs/i13-xct/i13_xct_config.yaml` | Diamond I13 (`Zn13X` legacy) | Legacy nominal-scale NeXus slicing template (`openlsrx download-zn13x --raw-h5-dir` preferred for 7-DOF alignment) |
| `configs/i13-xct/zn13x_2x_1625_to_08125.yaml` | Diamond I13 (`Zn13X` $2\times$) | Experimental $4\times$ ($1.625\ \mu\text{m}$) $\to 8\times$ ($0.8125\ \mu\text{m}$) 7-DOF similarity-registered slab (`scale = 0.98640`) |
| `configs/i13-xct/zn13x_2x_swinir_25d_physics.yaml` | Diamond I13 (`Zn13X` $2\times$) | Multi-axis `2.5D SwinIR` + physics/topology losses + `confidence_fourier` fusion on 7-DOF registered slabs |
| `configs/i13-xct/zn13x_8x_2600_to_0325.yaml` | Diamond I13 (`Zn13X` $8\times$) | Experimental $2.5\times$ ($2.600\ \mu\text{m}$) $\to 20\times$ ($0.325\ \mu\text{m}$) chained 3-rung 7-DOF slab (`scale = 0.987968`) |

---

## Preparing Open Rock & Diamond I13-2 `Zn-13X` Benchmarks

Run the built-in fetcher to prepare `bentheimer_sandstone`, `indiana_limestone`, physical $18\ \mu\text{m} \to 6\ \mu\text{m}$ ($3\times$) Bentheimer dual-scan pairs (`bentheimer_dualscan_core1`, `bentheimer_dualscan_core2`), and **7-DOF similarity-registered** Diamond Light Source I13-2 `Zn-13X` ($2\times$ & $8\times$) 3D reference volumes with cryptographic checksum verification and SHA-256 provenance manifests:

```bash
# Prepare laboratory rock benchmarks (Bentheimer from Zenodo 5542624 & calibrated carbonate phantom, default 128^3)
openlsrx download-benchmarks --dest data/benchmarks

# Prepare physical 18 um -> 6 um (3x, default 75^3 -> 225^3) dual-scan Bentheimer pairs from Zenodo 5542624
openlsrx download-benchmarks --datasets bentheimer_dualscan_core1 bentheimer_dualscan_core2 --dest data/benchmarks

# Prepare Diamond Light Source I13-2 Zn-13X synchrotron pairs using MultimodalBenchmark 7-DOF transforms
openlsrx download-zn13x --dest data/benchmarks --lr-size 48 [--raw-h5-dir /path/to/savu_h5] [--certified-regions configs/i13-xct/certified_regions.yaml]
```

For the 16-bit Zenodo `5542624` dual-scan `.raw` volumes, `prepare_bentheimer_dualscan_benchmark` verifies MD5 checksums, resolves any integer HR-grid axial offset (`Core1_Subvol1` has a $-1$ HR-voxel / $-6\ \mu\text{m}$ raw offset between the $333^3$ LR and $1000^3$ HR subvolumes) by exact integer HR-voxel slicing (`hr_shift_zyx == [0, 0, 0]`, zero interpolation), and scales raw `uint16` intensities linearly by `1 / 65535.0` without whole-volume percentile clipping so `PairedVolumeDataModule` fits both percentile bounds and `cross_scan_calibration` strictly on the `train` slab. Cache validation checks requested size, alignment schema (`multimodal_benchmark_7dof_v2` + `certified_regions_sha256`), local `--raw-dir` / `--raw-h5-dir`, and online vs. `--offline` mode so a cached offline phantom or pre-7-DOF slab never masks updated experimental data. Synthetic fallback manifests omit experimental Zenodo DOIs and set `"is_experimental_xct": false`, while full benchmark scripts fail loudly by default unless `--allow-synthetic-fallback` is passed.

---

## 7-DOF Similarity Registration for Multi-Magnification Synchrotron Scans

Experimental multi-magnification synchrotron scans across scintillator turret objectives exhibit both 3D rigid stage offsets and **$\sim 1.3\%$ true optical magnification drift** relative to nominal turret ratios (`scale = 0.98640` on `169067 -> 169066` $2\times$, and `scale = 0.987968` chained across `169068 <- 169067 <- 169066 <- 169065` $8\times$, which would otherwise induce $\sim 8$–$16$ HR voxels of edge drift). Rather than running a competing translation-only fit, OpenLSR-X consumes canonical `SimilarityTransform3D` definitions from [`BASE-Laboratory/MultimodalBenchmark`](https://github.com/BASE-Laboratory/MultimodalBenchmark) (`src/openlsrx/data/zn13x_certified_regions.yaml` / `--certified-regions`):

1. **Unresampled Ground-Truth HR (`hr_resampled == False`):** Extracts an exact integer-cropped native HR slab directly from on-disk 32-bit Savu `4-TomopyRecon-tomo/data` HDF5 files or Zenodo `14245751` preview HDF5s (`hr_vol` is never interpolated).
2. **Cell-Centered 7-DOF LR Resampling (`resample_lr_to_hr_grid_7dof`):** Evaluates the 7-DOF similarity transform (or composed 3-rung chain `compose_similarity_transforms_3d` for $8\times$) at the physical centers `+0.5 * (s - 1) * sp_hi` of each $s \times s \times s$ HR voxel block, delegating directly to `MultimodalBenchmark`'s `resample_moving_to_fixed_zyx` when installed or OpenLSR-X's `float32` bit-exact (`max_abs_diff == 0.0`) `scipy.ndimage.affine_transform` evaluator.
3. **Integer HR-Grid Utility (`register_multires_pair_3d`):** Retained for exact nominal-ratio dual-scan micro-CT pairs (e.g., $18\ \mu\text{m} \to 6\ \mu\text{m}$ Bentheimer Sandstone) and synthetic translation recovery tests.

---

## Open Synchrotron Multi-Resolution Benchmark (`Zn-13X`, Diamond I13-2)

The `configs/i13-xct/*.yaml` profiles correspond directly to the open multi-resolution synchrotron XCT dataset published in *Scientific Data* ([Green et al., 2025, `10.1038/s41597-025-04605-9`](https://doi.org/10.1038/s41597-025-04605-9)):

| Magnification | Nominal Voxel ($\mu\text{m}$) | Calibrated Voxel ($\mu\text{m}$) | Field of View ($\text{mm}$) | Reconstructed `.h5` / `.nxs` Zenodo DOI |
| :--- | :--- | :--- | :--- | :--- |
| `2.5x` (`169068`) | `2.600` | `2.5945` | $6.7 \times 5.6$ | [`10.5281/zenodo.12206815`](https://doi.org/10.5281/zenodo.12206815) |
| `4x` (`169067`) | `1.625` | `1.6212` | $4.2 \times 3.5$ | [`10.5281/zenodo.13327651`](https://doi.org/10.5281/zenodo.13327651) |
| `8x` (`169066`) | `0.8125` | `0.8006` | $2.1 \times 1.8$ | [`10.5281/zenodo.13327682`](https://doi.org/10.5281/zenodo.13327682) |
| `20x` (`169065`) | `0.325` | `0.3211` | $0.83 \times 0.70$ | [`10.5281/zenodo.13327692`](https://doi.org/10.5281/zenodo.13327692) |
| *Preview (Provisional)* | *8-bit sub-volumes* | *500×1000×1000 ROI* | *Cropped ROI* | [`10.5281/zenodo.14245751`](https://doi.org/10.5281/zenodo.14245751) |

> **Provisional 8-Bit Preview Notice:** The `10.5281/zenodo.14245751` `uint8` preview sub-volumes apply independent per-scan 8-bit contrast windows (`provisional_8bit_preview: true` in `*_manifest.json`). Use them for lightweight 7-DOF registration and pipeline previews, and pass `--raw-h5-dir` pointing to the 32-bit float Savu reconstructions (`169068`–`169065`) for final quantitative attenuation and transport reporting.
