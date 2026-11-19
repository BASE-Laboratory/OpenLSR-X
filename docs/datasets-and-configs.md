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
| `configs/i13-xct/i13_xct_config.yaml` | Diamond I13 (`Zn13X`) | Raw Savu TomopyRecon HDF5/NeXus template with coaxial ROI bounds and joint `[vmin, vmax]` calibration |
| `configs/i13-xct/zn13x_2x_1625_to_08125.yaml` | Diamond I13 (`Zn13X` $2\times$) | Experimental $4\times$ ($1.625\ \mu\text{m}$) $\to 8\times$ ($0.8125\ \mu\text{m}$) aligned slab |
| `configs/i13-xct/zn13x_2x_swinir_25d_physics.yaml` | Diamond I13 (`Zn13X` $2\times$) | Multi-axis `2.5D SwinIR` + physics/topology losses + `confidence_fourier` fusion |
| `configs/i13-xct/zn13x_8x_2600_to_0325.yaml` | Diamond I13 (`Zn13X` $8\times$) | Experimental $2.5\times$ ($2.600\ \mu\text{m}$) $\to 20\times$ ($0.325\ \mu\text{m}$) aligned slab |

---

## Preparing Open Rock & Diamond I13-2 `Zn-13X` Benchmarks

Run the built-in fetcher to prepare `bentheimer_sandstone`, `indiana_limestone`, physical $18\ \mu\text{m} \to 6\ \mu\text{m}$ ($3\times$) Bentheimer dual-scan pairs (`bentheimer_dualscan_core1`, `bentheimer_dualscan_core2`), and coaxial sub-voxel-registered Diamond Light Source I13-2 `Zn-13X` ($2\times$ & $8\times$) 3D reference volumes with cryptographic checksum verification and SHA-256 provenance manifests:

```bash
# Prepare laboratory rock benchmarks (Bentheimer from Zenodo 5542624 & calibrated carbonate phantom, default 128^3)
openlsrx download-benchmarks --dest data/benchmarks

# Prepare physical 18 um -> 6 um (3x, default 75^3 -> 225^3) dual-scan Bentheimer pairs from Zenodo 5542624
openlsrx download-benchmarks --datasets bentheimer_dualscan_core1 bentheimer_dualscan_core2 --dest data/benchmarks

# Prepare Diamond Light Source I13-2 Zn-13X synchrotron pairs (default 48^3 LR from local .h5 or Zenodo 14245751)
openlsrx download-zn13x --dest data/benchmarks --lr-size 48
```

For the 16-bit Zenodo `5542624` dual-scan `.raw` volumes, `prepare_bentheimer_dualscan_benchmark` verifies MD5 checksums, resolves any integer HR-grid axial offset (`Core1_Subvol1` has a $-1$ HR-voxel / $-6\ \mu\text{m}$ raw offset between the $333^3$ LR and $1000^3$ HR subvolumes) by exact integer HR-voxel slicing (`hr_shift_zyx == [0, 0, 0]`, zero interpolation), and scales raw `uint16` intensities linearly by `1 / 65535.0` without whole-volume percentile clipping so `PairedVolumeDataModule` fits both percentile bounds and `cross_scan_calibration` strictly on the `train` slab. Cache validation checks requested size, alignment status, local `--raw-dir` / `--raw-h5-dir`, and online vs. `--offline` mode so a cached offline smoke phantom never masks real experimental data. Synthetic fallback manifests omit experimental Zenodo DOIs and set `"is_experimental_xct": false`, while full benchmark scripts fail loudly by default unless `--allow-synthetic-fallback` is passed.

---

## Sub-Voxel 3D Registration for Multi-Magnification Synchrotron Scans

Experimental multi-magnification XCT scans exhibit sub-millimetre stage drift and distinct scintillator point-spread functions across objectives. `openlsrx.data.register_multires_pair_3d` and `openlsrx.data.extract_aligned_slab_from_h5` perform:

1. On-disk HDF5 ROI hyperslab slicing directly from $>50\ \text{GB}$ Savu `4-TomopyRecon-tomo/data` files (or coaxial center-ROI cropping of Zenodo `14245751` `16906*_8bit_cropped.h5` sub-volumes) without loading full cubes into host RAM.
2. Joint physical attenuation normalization using shared `[vmin, vmax]` calibration bounds across LR and HR scans.
3. HR-grid sub-voxel 3D FFT phase cross-correlation (`upsample_factor=4`, accurate to $\le 0.25$ HR voxels) followed by sub-voxel cubic shift interpolation and normalized cross-correlation quality verification (`registration_quality_ok`).

---

## Open Synchrotron Multi-Resolution Benchmark (`Zn-13X`, Diamond I13-2)

The `configs/i13-xct/*.yaml` profiles correspond directly to the open multi-resolution synchrotron XCT dataset published in *Scientific Data* ([Green et al., 2025, `10.1038/s41597-025-04605-9`](https://doi.org/10.1038/s41597-025-04605-9)):

| Magnification | Voxel Size ($\mu\text{m}$) | Field of View ($\text{mm}$) | Reconstructed `.h5` / `.nxs` Zenodo DOI |
| :--- | :--- | :--- | :--- |
| `2.5x` | `2.600` | $6.7 \times 5.6$ | [`10.5281/zenodo.12206815`](https://doi.org/10.5281/zenodo.12206815) |
| `4x` | `1.625` | $4.2 \times 3.5$ | [`10.5281/zenodo.13327651`](https://doi.org/10.5281/zenodo.13327651) |
| `8x` | `0.8125` | $2.1 \times 1.8$ | [`10.5281/zenodo.13327682`](https://doi.org/10.5281/zenodo.13327682) |
| `20x` | `0.325` | $0.83 \times 0.70$ | [`10.5281/zenodo.13327692`](https://doi.org/10.5281/zenodo.13327692) |
| *Preview* | *8-bit sub-volumes* | *Cropped ROI* | [`10.5281/zenodo.14245751`](https://doi.org/10.5281/zenodo.14245751) |
