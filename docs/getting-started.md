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

*(Optional)* For direct SimpleITK runtime delegation of `Zn-13X` 7-DOF similarity resampling to `BASE-Laboratory/MultimodalBenchmark` (otherwise OpenLSR-X uses its built-in `float32` bit-exact `scipy.ndimage` evaluator of the same `certified_regions.yaml` transforms):

```bash
uv pip install "multimodal-benchmark[registration] @ git+https://github.com/BASE-Laboratory/MultimodalBenchmark.git"
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

### 2. Reconstruct 3D Volumes from a Checkpoint

Run 3-axis orthogonal fusion (`predict_mode: orthogonal_3axis` by default) via LightningCLI `predict` or the standalone `predict-volume` subcommand (which automatically restores train-fitted normalization and cross-scan calibration from the checkpoint):

```bash
openlsrx predict -c configs/porespy/Predict/ps_predict_config.yaml \
  --ckpt_path lightning_logs/version_0/checkpoints/last.ckpt
```

```bash
openlsrx predict-volume \
  --ckpt lightning_logs/version_0/checkpoints/last.ckpt \
  --input data/benchmarks/zn13x_2x_lr.tif \
  --output-dir outputs/zn13x_sr \
  --fusion confidence_fourier \
  --save-uncertainty
```

In `orthogonal_3axis` mode, predictions are exported under `<output_dir>/predicted_orthogonal/` as assembled 3D `(Z, Y, X)` multi-page BigTIFF stacks (`*_predicted_stack.tiff`, `*_uncertainty_std_stack.tiff`, `*_low_res_stack.tiff`, and `*_high_res_stack.tiff` when paired).

### 3. Evaluate Petrophysical, Spectral & Transport Metrics

```bash
openlsrx evaluate \
  --pred outputs/ps_predict/predicted_orthogonal/ps_predict_predicted_stack.tiff \
  --target outputs/ps_predict/predicted_orthogonal/ps_predict_high_res_stack.tiff \
  --include-spectral \
  --include-transport \
  --output results/eval_report.json
```

---

## HPC Cluster Execution

Ready-to-customize batch scripts are provided in `scripts/hpc/`:

- **Isambard-AI (`GH200` SLURM array):** `sbatch scripts/hpc/isambard_ai_array.slurm`
- **Generic SLURM clusters:** `sbatch scripts/hpc/slurm_train.sh configs/porespy/High_Noise/ps_hn_hf_config.yaml`
- **Sun Grid Engine (e.g., UCL MMM Hub `Young`):** `qsub -v CONFIG_PATH=configs/labxct/bs_config.yaml scripts/hpc/sge_young_train.sh`
