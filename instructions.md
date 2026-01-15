# OpenLSR-X Instructions

This document is a complete working manual for the OpenLSR-X repository. It covers
what the project does, how it is structured, how to install and run it, and how
to test and troubleshoot it. All examples assume you are in the repo root.

---

## 1) What this repo is

OpenLSR-X implements a Learnable Super-Resolution GAN (SRGAN) for XCT data using
PyTorch and PyTorch Lightning. It supports multiple datasets through Lightning
DataModules and dataset-specific CLI entrypoints.

Core capabilities:
- Train SRGAN on XCT data from multiple datasets.
- Predict super-resolved outputs and save TIFF stacks.
- Run in HPC environments via batch scripts.

---

## 2) Repository layout

```
OpenLSR-X/
  src/
    openlsrx/
      __init__.py
      __main__.py
      cli.py
      models/
        blocks.py
        submodels.py
        srgan.py
      data/
        dataset.py
        porespy_funcs.py
        porespy_funcs_3d.py
      losses/
        loss.py
      entrypoints/
        main.py
        i13_xct.py
        i13_predict.py
        bentsand.py
        indlime.py
        porespy_xct.py
        porespy_predict.py
  configs/
    base/
      config.yaml
      srgan-xct_min.yml
    i13-xct/
      i13_xct_config.yaml
      i13_predict_config.yaml
    labxct/
      bs_config.yaml
      il_config.yaml
      bs_config_predict.yaml
      il_config_predict.yaml
    porespy/
      High_Noise/*.yaml
      Low_Noise/*.yaml
      Predict/*.yaml
  scripts/
    hpc/
      example_batch_job.sh
      example_batch_script.sh
    i13-xct/
      i13_xct_job.sh
      i13_predict_job.sh
    labxct/
      Predict/*.sh
      Young/*.sh
    porespy/
      High_Noise/*.sh
      Low_Noise/*.sh
      Predict/*.sh
  tests/
    test_blocks.py
    test_dataset.py
    test_srgan.py
    test_submodels.py
  pyproject.toml
  requirements.txt
  README.md
  LICENSE
```

---

## 3) Installation (uv only)

This repo uses `uv` for environment and dependency management.

1) Create a venv and install dependencies:
```bash
uv venv
uv pip install -r requirements.txt
uv pip install -e .
```

2) Verify the CLI:
```bash
python -m openlsrx.cli --help
```

### Windows note on `python3`
On Windows, `python3` is usually not present. Use `python` or the venv path:
```bash
.\.venv\Scripts\python.exe -m openlsrx.cli --help
```

---

## 4) GPU setup (Windows + RTX A5000)

To use CUDA, you must install a CUDA-enabled PyTorch wheel. CUDA wheels do not
exist for Python 3.13 on Windows. Use Python 3.12 or 3.11 for GPU support.

Example (Python 3.12 venv):
```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:
```bash
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 5) Running training and prediction

### Generic CLI
The default CLI uses the PoreSpy 3D datamodule:
```bash
python -m openlsrx.cli fit -c configs/base/config.yaml
```

### Dataset-specific entrypoints
These entrypoints bind SRGAN to a specific DataModule.

Bentheimer Sandstone:
```bash
python -m openlsrx.entrypoints.bentsand fit -c configs/labxct/bs_config.yaml
python -m openlsrx.entrypoints.bentsand predict -c configs/labxct/bs_config_predict.yaml
```

Indiana Limestone:
```bash
python -m openlsrx.entrypoints.indlime fit -c configs/labxct/il_config.yaml
python -m openlsrx.entrypoints.indlime predict -c configs/labxct/il_config_predict.yaml
```

I13 XCT:
```bash
python -m openlsrx.entrypoints.i13_xct fit -c configs/i13-xct/i13_xct_config.yaml
python -m openlsrx.entrypoints.i13_predict predict -c configs/i13-xct/i13_predict_config.yaml
```

PoreSpy:
```bash
python -m openlsrx.entrypoints.porespy_xct fit -c configs/porespy/High_Noise/ps_hn_hf_config.yaml
python -m openlsrx.entrypoints.porespy_predict predict -c configs/porespy/Predict/ps_hn_hf_predict_config.yaml
```

---

## 6) Key configuration fields

All configs are YAML and use LightningCLI conventions. Common fields:

`trainer`:
- `max_epochs`
- `devices`
- `accelerator` ("gpu" or "cpu")
- `log_every_n_steps`

`model` (SRGAN):
- `learning_rate`
- `adaptive_size` (disc adaptive pool size)
- `accumulate_n_batches`
- `super_res_factor` (2 or 4, etc.)
- `use_vgg_loss` (True/False)
- `vgg_loss_weight`
- `vgg_layers`
- `lr_scheduler_type` ("constant" or "CAWR")

`data` (dataset-dependent):
- `low_file`, `high_file`
- `dataset_size`, `train_size`, `valid_size`, `test_size`
- `batch_size`, `num_workers`, `pin_memory`, `persistent_workers`

---

## 7) How data flows (high-level)

1) DataModule loads low/high-resolution paired images or volumes.
2) `DatasetObject` returns `(low_res, high_res)` per item.
3) `SRGAN.training_step` upsamples `low_res` with the generator.
4) Discriminator and generator losses are computed.
5) Results and predictions are saved to disk for validation/test/predict.

---

## 8) Testing

Run tests:
```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Notes:
- Tests will use GPU if available.
- SRGAN tests use a smaller network for speed.

---

## 9) Common issues and fixes

### Missing `porespy` dependencies
`porespy` pulls in compiled dependencies (e.g. `edt`). On Windows, you need
Microsoft C++ Build Tools installed.

### `jsonargparse` missing
If you see:
`Requirement 'jsonargparse[signatures]>=4.27.7' not met`
Run:
```bash
uv pip install "jsonargparse[signatures]>=4.27.7"
```

### CUDA not available
If `torch.cuda.is_available()` is False, you likely installed CPU-only torch or
used Python 3.13. Use Python 3.12/3.11 and install a CUDA wheel.

---

## 10) Example: end-to-end quickstart

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install -e .
python -m openlsrx.entrypoints.bentsand fit -c configs/labxct/bs_config.yaml
```

---

## 11) Development notes

- Source code is under `src/openlsrx/`.
- Use module execution (`python -m openlsrx...`) to ensure package imports.
- Tests are in `tests/` and use `unittest`.

