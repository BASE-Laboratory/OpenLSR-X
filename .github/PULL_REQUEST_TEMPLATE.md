## Summary

<!-- Describe what this PR changes and why. -->

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature / scientific method (e.g., model block, loss, metric, data loader)
- [ ] Benchmark or configuration update
- [ ] Documentation / JOSS manuscript update
- [ ] CI/CD or packaging update

## Scientific & Engineering Checklist

- [ ] `uv run ruff check src tests scripts` and `uv run ruff format --check src tests scripts` pass
- [ ] `uv run mypy src tests scripts` passes with zero errors
- [ ] `uv run pytest --cov=openlsrx --cov-fail-under=85` passes (including YAML config schema checks)
- [ ] `uv run mkdocs build --strict` builds cleanly if docs or docstrings changed
- [ ] No 3D spatial train/val/test voxel leakage introduced in `openlsrx.data`
