# Contributing to OpenLSR-X

Thank you for your interest in contributing to **OpenLSR-X**, maintained by **BASE Laboratory** and building on the PhD research of Calum Green.

## Reporting Bugs & Requesting Features

Please open an issue on the [GitHub Issues](https://github.com/BASE-Laboratory/OpenLSR-X/issues) tracker including:
- Your operating system, Python version, and PyTorch/CUDA version.
- Minimal command or YAML configuration reproducing the issue.
- Relevant stack traces or metric discrepancies.

## Development Setup

We use [`uv`](https://docs.astral.sh/uv/) for fast, reproducible environment management:

```bash
git clone https://github.com/BASE-Laboratory/OpenLSR-X.git
cd OpenLSR-X
uv venv --python 3.11
uv pip install -e ".[dev,docs]"
```

*(For CPU-only development machines, install the lightweight CPU PyTorch wheel first via `uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision`.)*

## Quality Gates

Before opening a pull request, ensure all checks pass locally:

1. **Lockfile, linting, formatting & static typing:**
   ```bash
   uv lock --check
   uv run --no-sync ruff check src tests scripts examples
   uv run --no-sync ruff format --check src tests scripts examples
   uv run --no-sync mypy src tests scripts examples
   ```
2. **Unit, analytical metric, and config schema tests:**
   ```bash
   uv run --no-sync pytest --cov=openlsrx
   ```
3. **Documentation build:**
   ```bash
   uv run --no-sync mkdocs build --strict
   ```

## Scientific Reproducibility Guidelines

- **No spatial train/test leakage:** Any changes to 3D volume splitting in `openlsrx.data` must maintain strict voxel disjointness across `train`, `val`, and `test` slabs.
- **Deterministic seeding:** Thread explicit RNG seeds (`numpy.random.default_rng(seed)` / `pl.seed_everything(seed)`) through synthetic data generation and benchmark scripts.
- **Config-driven preprocessing:** Keep scan-specific crop boxes, rotation angles, and attenuation normalization bounds in YAML files under `configs/`, never hardcoded inside library functions.
