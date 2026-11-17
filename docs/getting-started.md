# Getting Started

## Installation

OpenLSR-X targets **Python 3.10–3.12** and uses [`uv`](https://docs.astral.sh/uv/) for deterministic dependency management.

### Standard GPU / Workstation Install

```bash
git clone https://github.com/BASE-Laboratory/OpenLSR-X.git
cd OpenLSR-X
uv venv --python 3.11
uv pip install -e ".[dev,docs]"
```

### Lightweight CPU-Only Install (Laptops & CI)

```bash
uv venv --python 3.11
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install -e ".[dev,docs]"
```

Verify your installation:

```bash
openlsrx --help
```

---

## Core CLI Workflows

All training, validation, testing, prediction, and evaluation tasks run through the single `openlsrx` command (or `python -m openlsrx`).

### 1. Train on Synthetic PoreSpy 3D XCT

```bash
openlsrx fit -c configs/base/config.yaml
```

Override hyperparameters directly from the command line:

```bash
openlsrx fit -c configs/base/config.yaml \
  --trainer.max_epochs 200 \
  --model.learning_rate 5e-5 \
  --data.init_args.noise_regime high_noise
```

### 2. Resume or Predict from a Checkpoint

```bash
openlsrx predict -c configs/porespy/Predict/ps_predict_config.yaml \
  --ckpt_path lightning_logs/version_0/checkpoints/last.ckpt \
  --data.init_args.predict_axis xz
```

Predictions are exported cleanly under your run's output directory as both individual 2D `.tiff` slices and assembled 3D multi-page BigTIFF stacks (`*_predicted_stack.tiff`, `*_low_res_stack.tiff`, `*_high_res_stack.tiff`).

### 3. Evaluate Petrophysical & Image Metrics

```bash
openlsrx evaluate \
  --pred outputs/predicted_xz/ps_predict_predicted_stack.tiff \
  --target outputs/predicted_xz/ps_predict_high_res_stack.tiff \
  --output results/eval_report.json
```

---

## HPC Cluster Execution

Ready-to-customize batch scripts are provided in `scripts/hpc/`:

- **SLURM clusters:** `sbatch scripts/hpc/slurm_train.sh configs/porespy/High_Noise/ps_hn_hf_config.yaml`
- **Sun Grid Engine (e.g., UCL MMM Hub `Young`):** `qsub -v CONFIG_PATH=configs/labxct/bs_config.yaml scripts/hpc/sge_young_train.sh`
