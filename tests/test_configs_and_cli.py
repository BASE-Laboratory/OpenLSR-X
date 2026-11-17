"""Schema validation for all repository YAML configs and in-process CLI tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from openlsrx.cli import main as cli_main

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


def test_cli_inprocess_download_and_evaluate(tmp_path: Path) -> None:
    bench_dir = tmp_path / "benchmarks"
    with pytest.raises(SystemExit) as exc_dl:
        cli_main(
            [
                "download-benchmarks",
                "--datasets",
                "bentheimer_sandstone",
                "indiana_limestone",
                "--dest",
                str(bench_dir),
                "--size",
                "24",
            ]
        )
    assert exc_dl.value.code == 0
    hr_tif = bench_dir / "bentheimer_sandstone_hr.tif"
    il_tif = bench_dir / "indiana_limestone_hr.tif"
    assert hr_tif.is_file() and il_tif.is_file()

    # Second call hits cached manifest path cleanly
    with pytest.raises(SystemExit) as exc_cached:
        cli_main(
            [
                "download-benchmarks",
                "--datasets",
                "bentheimer_sandstone",
                "--dest",
                str(bench_dir),
                "--size",
                "24",
            ]
        )
    assert exc_cached.value.code == 0

    vol = tifffile.imread(hr_tif)
    pred_tif = tmp_path / "pred.tif"
    out_json = tmp_path / "metrics.json"
    tifffile.imwrite(pred_tif, np.clip(vol + 0.02, 0.0, 1.0).astype(np.float32))

    with pytest.raises(SystemExit) as exc_ev:
        cli_main(
            [
                "evaluate",
                "--pred",
                str(pred_tif),
                "--target",
                str(hr_tif),
                "--threshold",
                "yen",
                "--output",
                str(out_json),
            ]
        )
    assert exc_ev.value.code == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["image_metrics"]["psnr"] > 25.0
    assert "porosity_abs_error" in report["petrophysical_metrics"]


def test_cli_evaluate_shape_mismatch_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.npy"
    b = tmp_path / "b.npy"
    np.save(a, np.zeros((8, 8, 8), dtype=np.float32))
    np.save(b, np.zeros((10, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="Shape mismatch"):
        cli_main(["evaluate", "--pred", str(a), "--target", str(b)])
