# OpenLSR-X

![](https://img.shields.io/badge/day%20📅-22-blue)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://github.com/Lightning-AI/lightning)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

---

## What is OpenLSR-X?

OpenLSR-X implements a Learnable Super-Resolution Generative Adversarial Network (SRGAN) for X-Ray Computed Tomography (XCT) data using PyTorch and PyTorch Lightning. It is based on [Ledig et al. (CVPR 2017)](https://doi.org/10.1109/CVPR.2017.19) and adapted from [Aladdin Persson's implementation](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/SRGAN).

### Key Features & Benefits

- **Stable Training:** Uses a mixture of MSE loss and VGG loss for improved stability.
- **High-Quality Results:** Employs 'reflect' padding in convolutional layers to reduce border artifacts.
- **Modular Design:** Easily extendable for new XCT datasets and tasks.
- **Batch Job Support:** Includes scripts for HPC environments.
- **Configurable:** All hyperparameters and data paths are set via YAML config files.

---

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd OpenLSR-X
   ```
2. Create the environment with uv:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   uv pip install -e .
   ```
3. (Optional) Reproduce the exact environment:
   ```bash
   uv pip install -r requirements.lock
   ```

### Developer Setup

- Run the CLI from source:
  ```bash
  python3 -m openlsrx.cli --help
  ```
- Run tests:
  ```bash
  python3 -m unittest discover -s tests
  ```

### Usage Example

Train a model using the provided config:
```bash
python3 -m openlsrx.cli fit -c configs/base/config.yaml
```
Override parameters from the command line:
```bash
python3 -m openlsrx.cli fit -c configs/base/config.yaml --model.learning_rate 0.0001 --data.batch_size 16
```
For dataset-specific training or prediction, use the corresponding entry points, e.g.:
```bash
python3 -m openlsrx.entrypoints.bentsand fit -c configs/labxct/bs_config.yaml
python3 -m openlsrx.entrypoints.porespy_xct fit -c configs/porespy/High_Noise/ps_hn_hf_config.yaml
```

---## DataLoaders Summary

The project provides several PyTorch Lightning DataModules for handling different XCT datasets:

- **PoreSpy3D_Volume:** Loads 3D PoreSpy volumes, processes orthogonal slices, and creates train/validation/test splits.
- **I13XCTDataModule:** Handles Diamond Light Source I13 XCT datasets, including normalization, alignment, and manual axis selection.
- **IndLimeDataModule:** For Indiana Limestone data, loads and aligns 2D images, supports orthogonal predictions.
- **BentSandDataModule:** For Bentheimer Sandstone, processes and aligns images, supports orthogonal predictions.
- **PoreSpyDataModule:** Loads PoreSpy data for all three axes, supports parallel and perpendicular test splits.

Each DataModule supports configuration of batch size, patch size, and augmentation via YAML files. All modules use the core `DatasetObject` class for paired data.

---

## Project Structure

- `src/openlsrx/` - Package code
- `src/openlsrx/models/` - SRGAN, generator/discriminator, and blocks
- `src/openlsrx/data/` - Data processing and LightningDataModules
- `src/openlsrx/entrypoints/` - Dataset-specific CLI modules
- `src/openlsrx/losses/` - Loss functions
- `configs/` - YAML config files
- `scripts/` - Batch job scripts for cluster environments
- `tests/` - Unit tests

---## Support & Documentation

- For issues, use the repository's [GitHub Issues](../../issues) page.
- For advanced usage and API details, see code comments and docstrings.

---

## Maintainers & Contributing

Maintained by Calum Green.

- Contributions are welcome! Please see the repository's CONTRIBUTING guidelines (if available) or open an issue to discuss your ideas.

---

## License

BSD 3-Clause License  
See [LICENSE](LICENSE) for details.


---

This README was generated using AI.

