# OpenLSR-X Documentation

**OpenLSR-X** is a PyTorch Lightning package and reproducible benchmark suite for **2D/`2.5D`-to-3D Super-Resolution and Petrophysical Transport Validation of X-Ray Computed Tomography (XCT)** scans of porous media.

## Why OpenLSR-X?

X-ray micro-CT imaging faces a fundamental trade-off between **field of view (FOV)** and **spatial resolution**. Resolving sub-micron pore throats typically restricts sample diameters to a few millimeters—below the representative elementary volume (REV) of heterogeneous sandstones, carbonates, and catalyst pellets—while large-FOV scans blur critical pore throats and bias downstream flow simulations.

Directly training 3D convolutional or adversarial networks on gigavoxel XCT volumes requires substantial GPU memory and paired 3D ground-truth volumes that are costly to acquire and register. **OpenLSR-X** addresses both limitations:

1. **Memory-Efficient 2D & `2.5D` Architectures + Native 3D Baseline:** Supports shifted-window Transformers (`SwinIRGenerator`), normalization-free Residual-in-Residual Dense Blocks (`RRDBGenerator` + `RaGAN` with $2^a 3^b$ and arbitrary fractional upsampling), classical residual `SRGAN`/`SRResNet`, and a volumetric `SR3DResNet` baseline, coupled with attenuation-conservation, soft-skeleton `clDice`, 3D Sobel gradient, and Focal Frequency losses.
2. **Scale-Aware Cell-Centered 3D Orthogonal Fourier Fusion:** Applies trained 2D/`2.5D` generators across all three orthogonal planes (`yz`, `xz`, `xy`) using cell-centered coordinates (`grid_mode=True`) and fuses them in symmetric-padded 3D Fourier space (`fourier` or `confidence_fourier`), eliminating slice-stacking ("venetian blind") artifacts while reporting inter-view standard deviation $\sigma_{\text{ortho}}(\mathbf{x})$ ($n=3$).
3. **Split-Isolated Petrophysical & 3D Transport Validation:** Enforces spatially guarded 3D slab splitting prior to train-only intensity normalization and cross-scan radiometric calibration (`affine` / monotonic `quantile`), evaluating 3D Fourier Shell Correlation (`FSC = 0.5`), segmented petrophysics, and residual-verified steady-state 3D Fickian diffusive tortuosity ($D_{\text{eff}}/D_0$, $\tau$) with direct `OpenImpala` AMReX export.

## Next Steps

- [Getting Started](getting-started.md) — Environment installation, GPU/CPU setup, and CLI usage.
- [End-to-End Tutorial](tutorial.md) — Interactive Python API and CLI walkthrough.
- [Datasets & Configuration](datasets-and-configs.md) — Synthetic PoreSpy, physical $3\times$ Bentheimer dual-scan, and Diamond Light Source I13-2 `Zn-13X` configurations.
- [Benchmarks & Metrics](benchmarks-and-metrics.md) — Reproducing factorial studies, LaTeX tables, spectral FSC, and 3D diffusive transport diagnostics.
- [Analytical & Physical Verification](verification.md) — Closed-form mathematical checks and multi-core physical dual-scan benchmarks.
- [API Reference](api.md) — Complete Python module documentation.
