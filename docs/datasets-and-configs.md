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
| `configs/i13-xct/i13_xct_config.yaml` | Diamond I13 (`Zn13X`) | Synchrotron TomopyRecon HDF5/NeXus template with ROI crops |

---

## Preparing Open Rock Micro-CT Benchmarks

Run the built-in fetcher to prepare `bentheimer_sandstone` and `indiana_limestone` 3D reference volumes with SHA-256 provenance manifests:

```bash
openlsrx download-benchmarks --dest data/benchmarks --size 128
```

When running on air-gapped HPC compute nodes without outbound internet access, `download-benchmarks` automatically synthesizes calibrated sandstone and bimodal carbonate digital-rock phantoms and records `"source_kind": "offline_calibrated_phantom"` in `<dataset>_manifest.json`.

---

## Using Custom Paired Laboratory or Synchrotron Scans

For paired low-resolution and high-resolution reconstructions (such as registered micro-CT scans or Diamond Light Source I13 NeXus files), configure `PairedVolumeDataModule` with explicit paths, alignment boxes, and intensity bounds:

```yaml
data:
  class_path: openlsrx.data.PairedVolumeDataModule
  init_args:
    low_path: /path/to/low_res_recon.nxs
    high_path: /path/to/high_res_recon.nxs
    h5_dataset_key: 4-TomopyRecon-tomo/data
    transpose_axes: [1, 2, 0]
    scale_factor: 2
    norm_mode: fixed
    low_vmin: -5.23e-4
    low_vmax: 1.11e-3
    high_vmin: -2.50e-4
    high_vmax: 6.79e-4
    low_z_range: [845, 1485]
    low_y_range: [1010, 1650]
    low_x_range: [858, 1498]
    high_z_range: [701, 1981]
    high_y_range: [763, 2043]
    high_x_range: [466, 1746]
```
