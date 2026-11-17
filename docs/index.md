# OpenLSR-X Documentation

**OpenLSR-X** is a PyTorch Lightning package and reproducible benchmark suite for **2D-to-3D Super-Resolution of X-Ray Computed Tomography (XCT)** scans of porous media.

## Why OpenLSR-X?

X-ray micro-CT imaging faces a fundamental trade-off between **field of view (FOV)** and **spatial resolution**. Resolving sub-micron pore throats typically restricts sample diameters to a few millimeters—below the representative elementary volume (REV) of heterogeneous sandstones, carbonates, and catalyst pellets—while large-FOV scans blur critical pore throats and bias downstream flow simulations.

Directly training 3D convolutional or adversarial networks on gigavoxel XCT volumes requires prohibitive GPU memory and paired 3D ground-truth volumes that are costly to acquire and register. **OpenLSR-X** addresses both limitations:

1. **Memory-Efficient 2D Slice Training:** Trains an XCT-tailored SRGAN (or non-adversarial SRResNet ablation) on 2D slices using reflection border padding, channel-wise `PixelNorm`, generator warmup epochs, and optional VGG-19 perceptual supervision.
2. **Isotropic 3D Orthogonal Fourier Fusion:** Applies the trained 2D generator across all three orthogonal planes (`yz`, `xz`, `xy`) of a 3D low-resolution scan and fuses them in 3D Fourier space by weighting each view's super-resolved in-plane spatial frequencies—eliminating slice-stacking ("venetian blind") artifacts.
3. **Domain-Specific Petrophysical Validation:** Quantifies accuracy on physical porous-media properties—porosity, pore-phase Dice/IoU, specific surface area, Euler connectivity, inscribed pore-radius distributions, and two-point correlation functions.

## Next Steps

- [Getting Started](getting-started.md) — Environment installation, GPU setup, and CLI usage.
- [Datasets & Configuration](datasets-and-configs.md) — Synthetic PoreSpy, Open Rock Micro-CT, LabXCT, and Diamond Light Source I13 HDF5 configurations.
- [Benchmarks & Metrics](benchmarks-and-metrics.md) — Reproducing factorial studies, LaTeX tables, and petrophysical diagnostics.
- [API Reference](api.md) — Python module documentation.
