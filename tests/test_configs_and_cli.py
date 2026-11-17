"""Schema validation for all repository YAML configs and CLI subcommands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = sorted((REPO_ROOT / "configs").rglob("*.yaml"))


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_all_yaml_configs_validate_with_lightning_cli(config_path: Path) -> None:
    subcmd = "predict" if "predict" in config_path.stem else "fit"
    proc = subprocess.run(
        [sys.executable, "-m", "openlsrx", subcmd, "--print_config", "-c", str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Config failed validation ({config_path}):\n{proc.stderr}"
    assert "model:" in proc.stdout


def test_cli_evaluate_and_download_benchmarks(tmp_path: Path) -> None:
    bench_dir = tmp_path / "benchmarks"
    dl_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "openlsrx",
            "download-benchmarks",
            "--datasets",
            "bentheimer_sandstone",
            "--dest",
            str(bench_dir),
            "--size",
            "24",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dl_proc.returncode == 0, dl_proc.stderr
    hr_tif = bench_dir / "bentheimer_sandstone_hr.tif"
    assert hr_tif.is_file()

    # Perturb slightly and run CLI evaluate
    vol = tifffile.imread(hr_tif)
    pred_tif = tmp_path / "pred.tif"
    out_json = tmp_path / "metrics.json"
    tifffile.imwrite(pred_tif, np.clip(vol + 0.02, 0.0, 1.0).astype(np.float32))

    ev_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "openlsrx",
            "evaluate",
            "--pred",
            str(pred_tif),
            "--target",
            str(hr_tif),
            "--output",
            str(out_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ev_proc.returncode == 0, ev_proc.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["image_metrics"]["psnr"] > 25.0
    assert "porosity_abs_error" in report["petrophysical_metrics"]
