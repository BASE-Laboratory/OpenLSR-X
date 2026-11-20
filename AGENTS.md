# Agent & Contributor Guidelines (`OpenLSR-X`)

Read [ECOSYSTEM.md](https://github.com/BASE-Laboratory/basecoat/blob/main/ECOSYSTEM.md) before adding a module, dependency, file reader, or dataset constant. The master copy of `ECOSYSTEM.md` lives in `BASE-Laboratory/basecoat`.

## Ecosystem Layer

- **Repository:** `BASE-Laboratory/OpenLSR-X`
- **Layer:** **Layer 2 — Image tools**
- **Allowed ecosystem dependencies:** `basecoat` (Layer 1) **only**.
- **Banned imports (enforced by `[tool.basecoat.layers]`):** `multimodal_benchmark`, `registration`, `openimpala`, `x_fuse`, `braggtrack`, `zarr`, `numcodecs`.

## Ownership Boundaries

- **Owns:** 2D/2.5D/3D super-resolution models (`SwinIR`, `RRDB-ESRGAN`, `SRGAN`, `SR3DResNet`), orthogonal 3-axis Fourier/confidence fusion and view-disagreement uncertainty (`src/openlsrx/inference/`), physics/topology/frequency training losses (`src/openlsrx/losses/`), and super-resolution evaluation metrics (`src/openlsrx/metrics/`).
- **Delegates to `basecoat` (`v0.2`):** Calibrated `Volume` / `VolumeSet` / `AxisSpec` containers, OME-Zarr v0.4 read/write (`read_ome_zarr`, `write_ome_zarr`), out-of-core Savu `(X, Z, Y)` HDF5 reading (`open_savu_h5`, `read_savu_subvolume`), generic TIFF/HDF5/NPY/raw reading and writing (`open_volume`, `read_volume`, `write_volume`), and 7-DOF `SimilarityTransform3D` chaining + cell-centred pullback resampling (`resample_volume`).
- **Deprecated for `v0.3` removal (retained in `v0.2.x` for manuscript reproducibility):** `export_for_openimpala` (superseded by `export_volume_ome_zarr` once `OpenImpala v5.1` accepts `.ome.zarr`) and bundled Zn-13X constants in `src/openlsrx/data/zn13x.py` (pass `--certified-regions` from `MultimodalBenchmark`).
