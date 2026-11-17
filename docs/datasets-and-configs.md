# Datasets & Configuration

OpenLSR-X provides two composable PyTorch Lightning `DataModule` classes selectable in YAML via `data.class_path`:

1. `openlsrx.data.SyntheticPoreSpyDataModule` — Deterministic 3D porous-media generation across noise and feature-scale regimes.
2. `openlsrx.data.PairedVolumeDataModule` — File-based loader supporting 3D multi-page `.tif`/`.tiff` stacks, directories of 2D slices, `.npy` arrays, and synchrotron `.h5`/`.hdf5`/`.nxs` files.

---

## Leakage-Free 3D Spatial Splitting

When evaluating 2D-trained models on orthogonal 3D XCT planes, naive random 2D slice splits leak structural information across axes because every `xy` or `xz` plane intersects all `yz` slices of the same cube.

Both DataModules partition the 3D volume along `split_axis` into **strictly non-overlapping 3D slabs** (`train`, `val`, `test`) separated by `guard_slices_lr` buffer slices before extracting 2D training or orthogonal evaluation slices.

---

## Dataset Profiles Included in `configs/`

| Config Path | Target Dataset | Description |
| :--- | :--- | :--- |
| `configs/base/config.yaml` | Synthetic PoreSpy | Quickstart $128^3$ dual-porosity attenuation benchmark |
| `configs/base/srresnet_ablation.yaml` | Synthetic PoreSpy | Non-adversarial L1/CAWR generator ablation |
| `configs/porespy/Low_Noise/*.yaml` | Factorial PoreSpy | Low acquisition noise × Low/High feature frequency |
| `configs/porespy/High_Noise/*.yaml` | Factorial PoreSpy | High acquisition noise × Low/High feature frequency |
| `configs/labxct/bs_config.yaml` | Bentheimer Sandstone | Well-sorted quartz sandstone (~22% intergranular porosity) |
| `configs/labxct/il_config.yaml` | Indiana Limestone | Bimodal vuggy + microporous carbonate (~18% total porosity) |
| `configs/i13-xct/i13_xct_config.yaml` | Diamond I13 (`Zn13X`) | Raw Savu TomopyRecon HDF5/NeXus template with ROI crops |
| `configs/i13-xct/zn13x_2x_1625_to_08125.yaml` | Diamond I13 (`Zn13X` $2\times$) | Experimental $4\times$ ($1.625\ \mu\text{m}$) $\to 8\times$ ($0.8125\ \mu\text{m}$) aligned slab |
| `configs/i13-xct/zn13x_8x_2600_to_0325.yaml` | Diamond I13 (`Zn13X` $8\times$) | Experimental $2.5\times$ ($2.600\ \mu\text{m}$) $\to 20\times$ ($0.325\ \mu\text{m}$) aligned slab |

---

## Preparing Open Rock & Diamond I13-2 `Zn-13X` Benchmarks

Run the built-in fetcher to prepare `bentheimer_sandstone`, `indiana_limestone`, and aligned Diamond Light Source I13-2 `Zn-13X` ($2\times$ & $8\times$) 3D reference volumes with SHA-256 provenance manifests:

```bash
# Prepare laboratory rock benchmarks (Bentheimer & Indiana Limestone)
openlsrx download-benchmarks --dest data/benchmarks --size 128

# Prepare Diamond Light Source I13-2 Zn-13X synchrotron pairs (from local .h5, Zenodo, or calibrated phantom)
openlsrx download-benchmarks --datasets zn13x_2x zn13x_8x --dest data/benchmarks --size 96
```

When running on air-gapped HPC compute nodes without outbound internet access (and when `--raw-h5-dir` is not supplied), `download-benchmarks` automatically synthesizes calibrated digital-rock and two-phase `Zn-13X` synchrotron phantoms and records `"source_kind"` in `<dataset>_manifest.json`.

---

## Vectorized 3D Registration for Multi-Magnification Synchrotron Scans

Experimental multi-magnification XCT scans suffer from sub-millimetre stage drift and distinct scintillator point-spread functions across objectives. `openlsrx.data.register_multires_pair_3d` and `openlsrx.data.extract_aligned_slab_from_h5` perform:

1. Chunked ROI slab extraction directly from $>50\ \text{GB}$ Savu `4-TomopyRecon-tomo/data` HDF5 files without loading full cubes into host RAM.
2. Calibrated attenuation normalization using verified I13-2 dynamic bounds.
3. Coarse-grid 3D FFT phase cross-correlation (`skimage.registration.phase_cross_correlation`) followed by local 3D SSIM refinement to produce strictly shape-matched `(LR, HR)` slabs.

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
