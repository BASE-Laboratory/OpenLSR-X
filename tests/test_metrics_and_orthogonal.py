"""Unit tests for image metrics, petrophysical XCT metrics, baselines, and 3D orthogonal fusion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openlsrx.baselines import upsample_2d, upsample_3d
from openlsrx.inference import fuse_orthogonal_volumes, super_resolve_volume_orthogonal
from openlsrx.metrics import evaluate_image_metrics, evaluate_petrophysical_metrics
from openlsrx.models import Generator


def _make_analytic_sphere_phantom(
    size: int = 48, radius: float = 14.0
) -> tuple[np.ndarray, np.ndarray]:
    zz, yy, xx = np.ogrid[:size, :size, :size]
    c = (size - 1) / 2.0
    dist_sq = (zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2
    pore_mask = dist_sq <= (radius**2)
    # Pore = 0.15, solid = 0.85
    vol = np.where(pore_mask, 0.15, 0.85).astype(np.float32)
    return vol, pore_mask


def test_analytical_image_and_petrophysical_metrics() -> None:
    vol, mask = _make_analytic_sphere_phantom(size=48, radius=14.0)
    img_m = evaluate_image_metrics(vol, vol)
    assert math.isinf(img_m["psnr"])
    assert img_m["ssim"] == pytest.approx(1.0, abs=1e-5)
    assert img_m["ssim_3d_triplanar"] == pytest.approx(1.0, abs=1e-5)
    assert img_m["rmse"] == pytest.approx(0.0, abs=1e-7)

    petro_m = evaluate_petrophysical_metrics(vol, vol, pore_is_low=True)
    expected_phi = float(mask.mean())
    assert petro_m["porosity_true"] == pytest.approx(expected_phi, abs=1e-6)
    assert petro_m["porosity_abs_error"] == pytest.approx(0.0, abs=1e-7)
    assert petro_m["pore_dice"] == pytest.approx(1.0, abs=1e-6)
    assert petro_m["pore_radius_wasserstein"] == pytest.approx(0.0, abs=1e-6)
    assert petro_m["two_point_corr_rmse"] == pytest.approx(0.0, abs=1e-7)
    assert petro_m["through_plane_anisotropy_error"] == pytest.approx(0.0, abs=1e-6)


def test_interpolation_baselines_shapes_and_ordering() -> None:
    hr, _ = _make_analytic_sphere_phantom(size=32, radius=10.0)
    # 2x binned LR
    lr = hr.reshape(16, 2, 16, 2, 16, 2).mean(axis=(1, 3, 5))

    up2 = upsample_2d(lr, scale_factor=2, mode="lanczos4")
    assert up2.shape == (16, 32, 32)

    near3 = upsample_3d(lr, scale_factors=2, mode="nearest")
    tric3 = upsample_3d(lr, scale_factors=2, mode="tricubic")
    assert near3.shape == hr.shape
    assert tric3.shape == hr.shape
    # Smooth sphere phantom should have higher PSNR under tricubic than nearest-neighbor
    assert evaluate_image_metrics(tric3, hr)["psnr"] > evaluate_image_metrics(near3, hr)["psnr"]


@pytest.mark.parametrize("fusion", ["mean", "median", "fourier"])
def test_orthogonal_3d_super_resolution_and_fusion(fusion: str) -> None:
    gen = Generator(in_channels=1, out_channels=8, num_blocks=1, super_res_factor=2)
    lr = np.random.default_rng(3).random((12, 12, 12), dtype=np.float32)

    outputs = super_resolve_volume_orthogonal(
        generator=gen,
        low_volume=lr,
        scale_factor=2,
        axes=("yz", "xz", "xy"),
        fusion=fusion,  # type: ignore[arg-type]
        batch_size=6,
    )
    assert set(outputs.keys()) == {"yz", "xz", "xy", "fused"}
    for vol in outputs.values():
        assert vol.shape == (24, 24, 24)
        assert np.isfinite(vol).all()

    # Identity sanity check on fusion: fusing identical volumes returns that exact volume
    ref = outputs["yz"]
    fused_ident = fuse_orthogonal_volumes({"yz": ref, "xz": ref, "xy": ref}, mode=fusion)  # type: ignore[arg-type]
    assert np.allclose(fused_ident, ref, atol=1e-5)
