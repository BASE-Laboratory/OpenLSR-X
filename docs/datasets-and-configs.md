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
| `configs/i13-xct/certified_regions.yaml` | Diamond I13 / DIAD (`Zn13X` 7-DOF) | Canonical Stage 0B/0C 7-DOF similarity transforms, calibrated voxel spacings & cleanup anchors from `BASE-Laboratory/MultimodalBenchmark` |
| `configs/i13-xct/benchmark_splits_v1.json` | Diamond I13 / DIAD (`256^3` Splits) | Canonical $\ge 36\ \mu\text{m}$ guard-banded `train`/`val`/`test` $256^3$ HR block definitions shared with `MultimodalBenchmark` |
| `configs/i13-xct/i13_xct_config.yaml` | Diamond I13 (`Zn13X` legacy) | Legacy nominal-scale NeXus slicing template (`openlsrx download-zn13x --raw-h5-dir` preferred for 7-DOF alignment) |
| `configs/i13-xct/zn13x_1p6x_swinir_25d_physics.yaml` | Diamond I13 (`Zn13X` $1.6\times$) | `Task T1` Rung 1: $2.600 \to 1.625\ \mu\text{m}$ (`169068 -> 169067`) rational $8/5$ `2.5D SwinIR` + ruby $z$-shadow exclusion |
| `configs/i13-xct/zn13x_2x_1625_to_08125.yaml` | Diamond I13 (`Zn13X` $2\times$) | `Task T1` Rung 2: $1.625 \to 0.8125\ \mu\text{m}$ (`169067 -> 169066`) 7-DOF similarity-registered slab (`scale = 0.98640`) |
| `configs/i13-xct/zn13x_2x_swinir_25d_physics.yaml` | Diamond I13 (`Zn13X` $2\times$) | Multi-axis `2.5D SwinIR` + physics/topology losses + `confidence_fourier` fusion on 7-DOF registered slabs |
| `configs/i13-xct/zn13x_2p5x_swinir_25d_physics.yaml` | Diamond I13 (`Zn13X` $2.5\times$) | `Task T1` Rung 3: $0.8125 \to 0.325\ \mu\text{m}$ (`169066 -> 169065`) rational $5/2$ `2.5D SwinIR` + 7-term cupping detrend |
| `configs/i13-xct/zn13x_5x_swinir_25d_physics.yaml` | Diamond I13 (`Zn13X` $5\times$) | `Task T1` Rung 4: $1.625 \to 0.325\ \mu\text{m}$ (`169067 -> 169065`) chained 2-rung 7-DOF `2.5D SwinIR` |
| `configs/i13-xct/zn13x_8x_2600_to_0325.yaml` | Diamond I13 (`Zn13X` $8\times$) | `Task T1` Rung 5: $2.600 \to 0.325\ \mu\text{m}$ (`169068 -> 169065`) chained 3-rung 7-DOF slab (`scale = 0.987968`) |
| `configs/i13-xct/zn13x_t2_diad_3x_swinir_25d.yaml` | I13-2 $\to$ K11 DIAD (`Task T2` $3\times$) | Cross-beamline $1.625\ \mu\text{m} \to 0.54\ \mu\text{m}$ (`169067 -> k11-43334`), gated on Stage 0C octant TRE $\le 1.0\ \mu\text{m}$ |

---

## Preparing Open Rock & Diamond I13-2 / K11 DIAD `Zn-13X` Benchmarks

Run the built-in fetcher to prepare `bentheimer_sandstone`, `indiana_limestone`, physical $18\ \mu\text{m} \to 6\ \mu\text{m}$ ($3\times$) Bentheimer dual-scan pairs (`bentheimer_dualscan_core1`, `bentheimer_dualscan_core2`), and **7-DOF similarity-registered + conservatively cleaned** Diamond Light Source I13-2 `Task T1` (`1.6x`, `2x`, `2.5x`, `5x`, `8x`) and Stage 0C-gated `Task T2` (`t2_diad_3x`) 3D reference volumes with SHA-256 provenance manifests:

```bash
# Prepare laboratory rock benchmarks (Bentheimer from Zenodo 5542624 & calibrated carbonate phantom, default 128^3)
openlsrx download-benchmarks --dest data/benchmarks

# Prepare physical 18 um -> 6 um (3x, default 75^3 -> 225^3) dual-scan Bentheimer pairs from Zenodo 5542624
openlsrx download-benchmarks --datasets bentheimer_dualscan_core1 bentheimer_dualscan_core2 --dest data/benchmarks

# Prepare all 5 Diamond Light Source I13-2 Task T1 Zn-13X rungs (1.6x, 2x, 2.5x, 5x, 8x)
openlsrx download-zn13x --pair t1_all --dest data/benchmarks --lr-size 48 [--raw-h5-dir /path/to/savu_h5] [--block-id test_00]
```

For the 16-bit Zenodo `5542624` dual-scan `.raw` volumes, `prepare_bentheimer_dualscan_benchmark` verifies MD5 checksums, resolves any integer HR-grid axial offset (`Core1_Subvol1` has a $-1$ HR-voxel / $-6\ \mu\text{m}$ raw offset between the $333^3$ LR and $1000^3$ HR subvolumes) by exact integer HR-voxel slicing (`hr_shift_zyx == [0, 0, 0]`, zero interpolation), and scales raw `uint16` intensities linearly by `1 / 65535.0` without whole-volume percentile clipping so `PairedVolumeDataModule` fits both percentile bounds and `cross_scan_calibration` strictly on the `train` slab. Cache validation checks requested size, alignment schema (`multimodal_benchmark_7dof_v2` + `certified_regions_sha256`), `clean_artifacts`, `split_block_id`, local `--raw-dir` / `--raw-h5-dir`, and online vs. `--offline` mode.

---

## 7-DOF Similarity Registration & Conservative Synchrotron Artifact Cleanup

Experimental multi-magnification synchrotron scans across scintillator turret objectives exhibit 3D rigid stage offsets, **$\sim 1.3\%$ true optical magnification drift** relative to nominal turret ratios, concentric partial-arc ring artifacts, local-tomography radial cupping (`169065`), and pink-beam hardening shadows around the `500 µm` ruby calibration sphere (`169067`/`169068`). OpenLSR-X delegates registration and conservative cleanup directly to [`BASE-Laboratory/MultimodalBenchmark`](https://github.com/BASE-Laboratory/MultimodalBenchmark) (`src/openlsrx/data/zn13x_certified_regions.yaml`, `zn13x_benchmark_splits_v1.json`, and `artifact_cleanup.py`):

1. **Unresampled Ground-Truth HR (`hr_resampled == False`) & Conservative Cleanup (`clean_synchrotron_xct_slab`):** Extracts an exact integer-cropped native HR slab (either centered on the certified ROI or sliced at an exact guard-banded $256^3$ block from `benchmark_splits_v1.json`), excluding ruby-shadowed slices (`compute_ruby_beam_hardening_clean_z_mask`) and applying edge-protected polar median de-ringing (`remove_rings_polar_conservative`) plus 7-term quadratic + radial-quartic cupping detrend (`detrend_local_tomo_cupping_3d`) on truncated `0.325 µm` local tomograms (`--no-clean-artifacts` disables cleanup for raw ablations).
2. **Cell-Centered 7-DOF LR Resampling (`resample_lr_to_hr_grid_7dof`):** Evaluates the 7-DOF similarity transform (or composed chain `compose_similarity_transforms_3d` for $5\times$ and $8\times$) at the physical centers `+0.5 * (s - 1) * sp_hi` of each $s \times s \times s$ HR block across integer and rational scales (`1.6 = 8/5`, `2.5 = 5/2`).
3. **Stage 0C Cross-Beamline Gate (`verify_stage0c_cross_beamline_gate`):** `Task T2` (`169067` `1.625 µm` $\to$ `k11-43334` `0.54 µm`, `4-AstraReconGpu-tomo/data` with X-axis chirality flip) refuses execution unless `--stage0c-yaml` certifies octant drift $\le 1.0\ \mu\text{m}$ (or `--allow-uncertified-t2` is explicitly passed for exploratory diagnostics).

---

## Open Synchrotron Multi-Resolution Benchmark (`Zn-13X`, Diamond I13-2 & K11 DIAD)

The `configs/i13-xct/*.yaml` profiles correspond directly to the open multi-resolution synchrotron XCT dataset published in *Scientific Data* ([Green et al., 2025, `10.1038/s41597-025-04605-9`](https://doi.org/10.1038/s41597-025-04605-9)):

| Beamline / Scan | Nominal Voxel ($\mu\text{m}$) | Calibrated Voxel ($\mu\text{m}$) | Field of View ($\text{mm}$) | Reconstructed `.h5` / `.nxs` Zenodo DOI |
| :--- | :--- | :--- | :--- | :--- |
| I13-2 `2.5x` (`169068`) | `2.600` | `2.5945` | $6.7 \times 5.6$ | [`10.5281/zenodo.12206815`](https://doi.org/10.5281/zenodo.12206815) |
| I13-2 `4x` (`169067`) | `1.625` | `1.6212` | $4.2 \times 3.5$ | [`10.5281/zenodo.13327651`](https://doi.org/10.5281/zenodo.13327651) |
| I13-2 `8x` (`169066`) | `0.8125` | `0.8006` | $2.1 \times 1.8$ | [`10.5281/zenodo.13327682`](https://doi.org/10.5281/zenodo.13327682) |
| I13-2 `20x` (`169065`) | `0.325` | `0.3211` | $0.83 \times 0.70$ | [`10.5281/zenodo.13327692`](https://doi.org/10.5281/zenodo.13327692) |
| K11 DIAD (`43334`) | `0.540` | `0.5400` | $1.38 \times 1.17$ | [`10.5281/zenodo.13349402`](https://doi.org/10.5281/zenodo.13349402) |
| *Preview (Provisional)* | *8-bit sub-volumes* | *500×1000×1000 ROI* | *Cropped ROI* | [`10.5281/zenodo.14245751`](https://doi.org/10.5281/zenodo.14245751) |

> **Provisional 8-Bit Preview Notice:** The `10.5281/zenodo.14245751` `uint8` preview sub-volumes apply independent per-scan 8-bit contrast windows (`provisional_8bit_preview: true` in `*_manifest.json`). Use them for lightweight 7-DOF registration and pipeline previews, and pass `--raw-h5-dir` pointing to the 32-bit float Savu reconstructions for final quantitative attenuation and transport reporting.
