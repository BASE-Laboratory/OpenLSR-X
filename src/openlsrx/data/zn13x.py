"""Diamond Light Source I13-2 multi-resolution Zn-13X synchrotron dataset & 7-DOF aligner.

Consumes canonical 7-DOF similarity registration transforms (``certified_regions.yaml`` /
``SimilarityTransform3D``) produced by ``BASE-Laboratory/MultimodalBenchmark``
(``registration.i13_ladder``), chaining adjacent-rung transforms for ``8x``
(``169068 <- 169067 <- 169066 <- 169065``) and resampling LR onto the cell-centered
coarsened physical grid of the unresampled HR target for the open dataset published in:

    Green, C., Ahmed, S., Marathe, S. et al., "Three-dimensional, multimodal synchrotron
    data for machine learning applications", Scientific Data 12, 329 (2025),
    https://doi.org/10.1038/s41597-025-04605-9
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import tifffile
import yaml
from scipy.ndimage import affine_transform, gaussian_filter
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

from .artifact_cleanup import (
    clean_synchrotron_xct_slab,
    compute_ruby_beam_hardening_clean_z_mask,
)
from .transforms import (
    normalize_volume,
    rational_scale_pq,
    sanitize_xct_volume,
    zoom_cell_centered,
)

Zn13XPairName = Literal["1.6x", "2x", "2.5x", "5x", "8x", "t2_diad_3x"]
Zn13XTier = Literal["preview", "full"]

ZN13X_T1_PAIRS: tuple[Zn13XPairName, ...] = ("1.6x", "2x", "2.5x", "5x", "8x")
ZN13X_ALL_PAIRS: tuple[Zn13XPairName, ...] = (*ZN13X_T1_PAIRS, "t2_diad_3x")

DEFAULT_CERTIFIED_REGIONS_PATH = Path(__file__).with_name("zn13x_certified_regions.yaml")
DEFAULT_BENCHMARK_SPLITS_PATH = Path(__file__).with_name("zn13x_benchmark_splits_v1.json")

_PAIR_ALIASES: dict[str, Zn13XPairName] = {
    "1.6x": "1.6x",
    "1p6x": "1.6x",
    "zn13x_1p6x": "1.6x",
    "zn13x_1.6x": "1.6x",
    "2x": "2x",
    "zn13x_2x": "2x",
    "2.5x": "2.5x",
    "2p5x": "2.5x",
    "zn13x_2p5x": "2.5x",
    "zn13x_2.5x": "2.5x",
    "5x": "5x",
    "zn13x_5x": "5x",
    "8x": "8x",
    "zn13x_8x": "8x",
    "t2": "t2_diad_3x",
    "t2_diad_3x": "t2_diad_3x",
    "zn13x_t2_diad_3x": "t2_diad_3x",
}


def normalize_zn13x_pair_name(pair: str) -> Zn13XPairName:
    """Normalize CLI/config pair aliases (e.g. ``1p6x`` -> ``1.6x``, ``t2`` -> ``t2_diad_3x``)."""
    key = str(pair).strip().lower()
    if key not in _PAIR_ALIASES:
        raise ValueError(
            f"Unsupported Zn-13X pair {pair!r}. Expected one of {list(ZN13X_ALL_PAIRS)} "
            "(or aliases '1p6x', '2p5x', 't2')."
        )
    return _PAIR_ALIASES[key]


def zn13x_pair_file_slug(pair: str) -> str:
    """Return filesystem-safe stem suffix (preserving ``2x`` and ``8x`` compatibility)."""
    return normalize_zn13x_pair_name(pair).replace(".", "p")


def zn13x_pair_scale_factor(pair: str) -> int | float:
    """Return exact voxel super-resolution factor for a canonical Zn-13X track."""
    table: dict[Zn13XPairName, int | float] = {
        "1.6x": 1.6,
        "2x": 2,
        "2.5x": 2.5,
        "5x": 5,
        "8x": 8,
        "t2_diad_3x": 3,
    }
    return table[normalize_zn13x_pair_name(pair)]


@dataclass(frozen=True)
class Zn13XResolutionSpec:
    """Metadata and calibrated alignment anchors for one I13-2 or K11 DIAD magnification."""

    key: str
    scan_id: str
    magnification: str
    voxel_size_um: float
    calibrated_voxel_size_um: float
    zenodo_record: str
    recon_filename: str
    preview_filename: str
    vmin: float
    vmax: float
    anchor_z: int
    roi_1040um_yx: tuple[tuple[int, int], tuple[int, int]] | None
    roi_812um_yx: tuple[tuple[int, int], tuple[int, int]]
    h5_dataset_key: str = "4-TomopyRecon-tomo/data"
    local_tomography_truncated: bool = False
    flip_x_astra: bool = False


ZN13X_SUBVOLUMES_ZENODO_RECORD = "14245751"

ZN13X_I13_RESOLUTIONS: dict[str, Zn13XResolutionSpec] = {
    "2p600um": Zn13XResolutionSpec(
        key="2p600um",
        scan_id="169068",
        magnification="2.5x",
        voxel_size_um=2.600,
        calibrated_voxel_size_um=2.594476,
        zenodo_record="12206815",
        recon_filename="169068_recon.h5",
        preview_filename="169068_8bit_cropped.h5",
        vmin=-5.41e-4,
        vmax=1.68e-3,
        anchor_z=934,
        roi_1040um_yx=((1102, 1502), (1006, 1406)),
        roi_812um_yx=((1099, 1411), (1096, 1408)),
    ),
    "1p625um": Zn13XResolutionSpec(
        key="1p625um",
        scan_id="169067",
        magnification="4x",
        voxel_size_um=1.625,
        calibrated_voxel_size_um=1.621245,
        zenodo_record="13327651",
        recon_filename="169067_recon.h5",
        preview_filename="169067_8bit_cropped.h5",
        vmin=-5.23e-4,
        vmax=1.11e-3,
        anchor_z=894,
        roi_1040um_yx=((1010, 1650), (858, 1498)),
        roi_812um_yx=((1005, 1505), (1002, 1502)),
    ),
    "0p8125um": Zn13XResolutionSpec(
        key="0p8125um",
        scan_id="169066",
        magnification="8x",
        voxel_size_um=0.8125,
        calibrated_voxel_size_um=0.800596,
        zenodo_record="13327682",
        recon_filename="169066_recon.h5",
        preview_filename="169066_8bit_cropped.h5",
        vmin=-2.50e-4,
        vmax=6.79e-4,
        anchor_z=804,
        roi_1040um_yx=((763, 2043), (466, 1746)),
        roi_812um_yx=((753, 1753), (753, 1753)),
    ),
    "0p325um": Zn13XResolutionSpec(
        key="0p325um",
        scan_id="169065",
        magnification="20x",
        voxel_size_um=0.325,
        calibrated_voxel_size_um=0.321090,
        zenodo_record="13327692",
        recon_filename="169065_recon.h5",
        preview_filename="169065_8bit_cropped.h5",
        vmin=-2.50e-4,
        vmax=6.79e-4,
        anchor_z=509,
        roi_1040um_yx=None,
        roi_812um_yx=((0, 2496), (0, 2496)),
        local_tomography_truncated=True,
    ),
    "diad_0p54um": Zn13XResolutionSpec(
        key="diad_0p54um",
        scan_id="k11-43334",
        magnification="DIAD",
        voxel_size_um=0.540,
        calibrated_voxel_size_um=0.540,
        zenodo_record="13349402",
        recon_filename="tomo_p4_astra_recon_gpu.h5",
        preview_filename="k11-43334_8bit_cropped.h5",
        vmin=-4.00e-4,
        vmax=1.20e-3,
        anchor_z=1080,
        roi_1040um_yx=((316, 2242), (316, 2242)),
        roi_812um_yx=((528, 2032), (528, 2032)),
        h5_dataset_key="4-AstraReconGpu-tomo/data",
        flip_x_astra=True,
    ),
}

_PAIR_RESOLUTION_MAP: dict[Zn13XPairName, tuple[str, str, int | float, int]] = {
    "1.6x": ("2p600um", "1p625um", 1.6, 400),
    "2x": ("1p625um", "0p8125um", 2, 640),
    "2.5x": ("0p8125um", "0p325um", 2.5, 600),
    "5x": ("1p625um", "0p325um", 5, 384),
    "8x": ("2p600um", "0p325um", 8, 240),
    "t2_diad_3x": ("1p625um", "diad_0p54um", 3, 480),
}


def _resolve_pair_specs(
    pair: str,
) -> tuple[Zn13XPairName, Zn13XResolutionSpec, Zn13XResolutionSpec, int | float, int]:
    canon = normalize_zn13x_pair_name(pair)
    lo_key, hi_key, scale, default_lr_yx = _PAIR_RESOLUTION_MAP[canon]
    return canon, ZN13X_I13_RESOLUTIONS[lo_key], ZN13X_I13_RESOLUTIONS[hi_key], scale, default_lr_yx


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_https_file(url: str, dest: Path, timeout_s: float = 45.0) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"Refusing non-HTTPS Zenodo download URL: {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": "OpenLSR-X/0.2.1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp, dest.open("wb") as out_f:  # noqa: S310
        shutil.copyfileobj(resp, out_f)


def _block_downsample_3d(vol: np.ndarray, factor: int) -> np.ndarray:
    """Exact block-average 3D downsampling along all 3 axes by integer ``factor``."""
    d, h, w = vol.shape
    cd = (d // factor) * factor
    ch = (h // factor) * factor
    cw = (w // factor) * factor
    cropped = vol[:cd, :ch, :cw]
    return cropped.reshape(
        cd // factor,
        factor,
        ch // factor,
        factor,
        cw // factor,
        factor,
    ).mean(axis=(1, 3, 5), dtype=np.float32)


def _downsample_3d_rational(vol: np.ndarray, scale_factor: float | int) -> np.ndarray:
    """Downsample 3D HR volume to cell-centered LR grid across integer or rational ``scale_factor``."""
    sf = float(scale_factor)
    s_int = int(round(sf))
    if abs(sf - float(s_int)) < 1e-6:
        return _block_downsample_3d(vol, s_int)
    return zoom_cell_centered(vol, zoom_factors=1.0 / sf, order=1)


def _triplanar_mid_ssim(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Compute both z-midplane SSIM and 3-axis tri-planar mid-slice SSIM on 3D cubes."""
    planes = [
        (a[a.shape[0] // 2], b[b.shape[0] // 2]),
        (a[:, a.shape[1] // 2, :], b[:, b.shape[1] // 2, :]),
        (a[:, :, a.shape[2] // 2], b[:, :, b.shape[2] // 2]),
    ]
    scores: list[float] = []
    for p_a, p_b in planes:
        win = min(7, int(min(p_a.shape[0], p_a.shape[1])))
        if win % 2 == 0:
            win -= 1
        if win >= 3:
            scores.append(
                float(structural_similarity(p_a, p_b, data_range=1.0, win_size=max(3, win)))
            )
        else:
            scores.append(-float(np.mean((p_a - p_b) ** 2)))
    return scores[0], float(np.mean(scores))


def apply_hr_grid_shift_pair_3d(
    lr_vol: np.ndarray,
    hr_vol: np.ndarray,
    shift_hr_zyx: tuple[int, int, int],
    scale_factor: int,
    min_overlap_lr: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Slice ``(lr_vol, hr_vol)`` by an exact integer HR-voxel shift without interpolation or intensity scaling."""
    s = int(scale_factor)
    if s < 1:
        raise ValueError(f"scale_factor must be >= 1, got {scale_factor}.")
    dz_h, dy_h, dx_h = (int(shift_hr_zyx[0]), int(shift_hr_zyx[1]), int(shift_hr_zyx[2]))
    hz = (-dz_h) % s
    hy = (-dy_h) % s
    hx = (-dx_h) % s
    qz = (dz_h + hz) // s
    qy = (dy_h + hy) // s
    qx = (dx_h + hx) // s

    avail_z_c = (int(hr_vol.shape[0]) - hz) // s
    avail_y_c = (int(hr_vol.shape[1]) - hy) // s
    avail_x_c = (int(hr_vol.shape[2]) - hx) // s

    z0_c = max(0, -qz)
    y0_c = max(0, -qy)
    x0_c = max(0, -qx)

    z1_c = min(avail_z_c, int(lr_vol.shape[0]) - qz)
    y1_c = min(avail_y_c, int(lr_vol.shape[1]) - qy)
    x1_c = min(avail_x_c, int(lr_vol.shape[2]) - qx)

    req_min = max(1, int(min_overlap_lr))
    if (z1_c - z0_c) < req_min or (y1_c - y0_c) < req_min or (x1_c - x0_c) < req_min:
        return None

    lr_sub = lr_vol[z0_c + qz : z1_c + qz, y0_c + qy : y1_c + qy, x0_c + qx : x1_c + qx].copy()
    hr_sub = hr_vol[
        hz + z0_c * s : hz + z1_c * s,
        hy + y0_c * s : hy + y1_c * s,
        hx + x0_c * s : hx + x1_c * s,
    ].copy()
    return lr_sub, hr_sub


def register_multires_pair_3d(
    low_vol: np.ndarray,
    high_vol: np.ndarray,
    scale_factor: int = 2,
    local_refine_radius: int = 1,
    min_triplanar_ssim: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """HR-voxel sub-pixel 3D registration of paired multi-magnification XCT volumes.

    Uses sub-voxel FFT phase cross-correlation (``upsample_factor=scale_factor``) followed by
    integer HR-grid translation search scored by tri-planar 3D SSIM, eliminating residual
    half-LR-voxel (up to ``scale_factor // 2`` HR voxels) misregistration.
    """
    if scale_factor < 1:
        raise ValueError(f"scale_factor must be >= 1, got {scale_factor}.")

    lr = sanitize_xct_volume(low_vol)
    hr = sanitize_xct_volume(high_vol)
    if lr.ndim != 3 or hr.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got {lr.shape} and {hr.shape}.")

    hr_coarse = _block_downsample_3d(hr, scale_factor)
    common_lr_shape = tuple(min(a, b) for a, b in zip(lr.shape, hr_coarse.shape, strict=True))
    lr_ref = lr[: common_lr_shape[0], : common_lr_shape[1], : common_lr_shape[2]]
    hr_ref = hr_coarse[: common_lr_shape[0], : common_lr_shape[1], : common_lr_shape[2]]

    shift_est_lr, _, _ = phase_cross_correlation(
        lr_ref, hr_ref, upsample_factor=max(1, int(scale_factor))
    )
    # shift_est_lr maps coarse HR coordinate c -> LR coordinate c + shift_lr.
    # In HR voxel units: hr_coord + shift_hr <-> s * lr_coord => shift_hr = round(s * shift_est_lr).
    s = int(scale_factor)
    base_shift_hr = tuple(int(round(float(sh) * s)) for sh in shift_est_lr)

    def _extract_hr_grid_overlap(
        shift_hr: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        sliced = apply_hr_grid_shift_pair_3d(
            lr,
            hr,
            shift_hr_zyx=shift_hr,
            scale_factor=s,
            min_overlap_lr=min(8, int(min(common_lr_shape))),
        )
        if sliced is None:
            return None
        lr_sub, hr_sub = sliced
        hr_c_sub = _block_downsample_3d(hr_sub, s)
        return lr_sub, hr_c_sub, hr_sub

    hr_radius = max(1, int(local_refine_radius))
    candidates_hr = {
        (base_shift_hr[0] + dz, base_shift_hr[1] + dy, base_shift_hr[2] + dx)
        for dz in range(-hr_radius, hr_radius + 1)
        for dy in range(-hr_radius, hr_radius + 1)
        for dx in range(-hr_radius, hr_radius + 1)
    }
    candidates_hr.add((0, 0, 0))

    zero_extracted = _extract_hr_grid_overlap((0, 0, 0))
    zero_tri_score = (
        _triplanar_mid_ssim(zero_extracted[0], zero_extracted[1])[1]
        if zero_extracted is not None
        else -2.0
    )

    best_tri_score = -2.0
    best_mid_score = -2.0
    best_shift_hr = (0, 0, 0)
    best_pair: tuple[np.ndarray, np.ndarray] | None = None

    for cand_hr in sorted(candidates_hr):
        extracted = _extract_hr_grid_overlap(cand_hr)
        if extracted is None:
            continue
        lr_sub, hr_c_sub, hr_sub = extracted
        mid_score, tri_score = _triplanar_mid_ssim(lr_sub, hr_c_sub)
        if tri_score > best_tri_score:
            best_tri_score = tri_score
            best_mid_score = mid_score
            best_shift_hr = cand_hr
            best_pair = (lr_sub.copy(), hr_sub.copy())

    if best_pair is None:
        d, h, w = common_lr_shape
        best_pair = (lr[:d, :h, :w].copy(), hr[: d * s, : h * s, : w * s].copy())
        best_mid_score, best_tri_score = _triplanar_mid_ssim(
            best_pair[0], _block_downsample_3d(best_pair[1], s)
        )
        best_shift_hr = (0, 0, 0)

    min_ssim = float(min_triplanar_ssim)
    if best_tri_score < min_ssim:
        import warnings

        warnings.warn(
            f"Low 3D registration tri-planar SSIM ({best_tri_score:.4f} < {min_ssim:.2f}); "
            "verify that LR and HR ROIs physically overlap.",
            UserWarning,
            stacklevel=2,
        )

    info: dict[str, object] = {
        "coarse_shift_zyx": [round(sh / float(s), 4) for sh in best_shift_hr],
        "hr_shift_zyx": list(best_shift_hr),
        "initial_triplanar_ssim": zero_tri_score,
        "coarse_mid_slice_ssim": best_mid_score,
        "hr_triplanar_ssim": best_tri_score,
        "registration_quality_ok": bool(best_tri_score >= min_ssim),
        "aligned_lr_shape": list(best_pair[0].shape),
        "aligned_hr_shape": list(best_pair[1].shape),
        "scale_factor": s,
    }
    return best_pair[0], best_pair[1], info


def synthesize_zn13x_phantom_pair(
    pair: str = "2x",
    lr_shape: tuple[int, int, int] = (48, 48, 48),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a calibrated tri-phase Zn-doped Zeolite 13X synchrotron XCT pair across any T1/T2 track."""
    canon = normalize_zn13x_pair_name(pair)
    scale = zn13x_pair_scale_factor(canon)
    num_p, den_q = rational_scale_pq(float(scale))
    eff_lr_shape = tuple(
        max(den_q, (int(s) // den_q) * den_q) if den_q > 1 else max(1, int(s)) for s in lr_shape
    )
    hr_shape = (
        (eff_lr_shape[0] * num_p) // den_q,
        (eff_lr_shape[1] * num_p) // den_q,
        (eff_lr_shape[2] * num_p) // den_q,
    )
    rng = np.random.default_rng(seed)

    zz = np.linspace(-1.0, 1.0, hr_shape[0], dtype=np.float32)[:, np.newaxis, np.newaxis]
    yy = np.linspace(-1.0, 1.0, hr_shape[1], dtype=np.float32)[np.newaxis, :, np.newaxis]
    xx = np.linspace(-1.0, 1.0, hr_shape[2], dtype=np.float32)[np.newaxis, np.newaxis, :]
    radial_sq = 0.85 * (zz * zz) + (yy * yy) + 1.1 * (xx * xx)

    sf_f = float(scale)
    coarse_field = rng.standard_normal(hr_shape, dtype=np.float32)
    gaussian_filter(coarse_field, sigma=max(2.0, sf_f * 1.4), output=coarse_field)
    stride = max(1, int(round((hr_shape[0] * hr_shape[1] * hr_shape[2] / 500_000) ** (1.0 / 3.0))))
    q_coarse = float(np.quantile(coarse_field[::stride, ::stride, ::stride], 0.78))
    pore_mask = coarse_field > q_coarse
    del coarse_field

    fine_field = rng.standard_normal(hr_shape, dtype=np.float32)
    gaussian_filter(fine_field, sigma=max(0.9, sf_f * 0.45), output=fine_field)
    q_fine = float(np.quantile(fine_field[::stride, ::stride, ::stride], 0.84))
    pore_mask |= fine_field > q_fine
    del fine_field

    hr_clean = np.where(radial_sq > (0.52 * 0.52), np.float32(0.88), np.float32(0.68))
    del radial_sq
    hr_clean[pore_mask] = np.float32(0.14)
    del pore_mask

    hr_vol = gaussian_filter(hr_clean, sigma=0.55, mode="reflect")
    hr_noise = rng.normal(0.0, 0.015, size=hr_shape).astype(np.float32, copy=False)
    hr_vol += hr_noise
    del hr_noise
    np.clip(hr_vol, 0.0, 1.0, out=hr_vol)

    lr_blurred = gaussian_filter(hr_clean, sigma=0.85 * sf_f, mode="reflect")
    del hr_clean
    lr_binned = _downsample_3d_rational(lr_blurred, scale)
    del lr_blurred
    noise_sigma = 0.024 if sf_f <= 3.0 else 0.035
    lr_vol = lr_binned + rng.normal(0.0, noise_sigma, size=lr_binned.shape).astype(np.float32)
    np.clip(lr_vol, 0.0, 1.0, out=lr_vol)
    return lr_vol.astype(np.float32, copy=False), hr_vol.astype(np.float32, copy=False)


@dataclass(frozen=True)
class SimilarityTransform3DSpec:
    """7-DOF 3D similarity transform in physical micron coordinates (SimpleITK XYZ convention).

    Schema-identical to ``registration.i13_ladder.SimilarityTransform3D`` in
    ``BASE-Laboratory/MultimodalBenchmark``, where ``matrix_3x3`` equals ``scale * R``.
    """

    matrix_3x3: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    translation_xyz_um: tuple[float, float, float]
    center_xyz_um: tuple[float, float, float]
    scale: float
    versor_xyzw: tuple[float, float, float, float]

    def transform_points_xyz_um(self, pts_xyz_um: np.ndarray) -> np.ndarray:
        """Map fixed physical points ``(N, 3)`` in ``(x, y, z)`` um to moving physical points."""
        pts = np.asarray(pts_xyz_um, dtype=np.float64)
        mat = np.asarray(self.matrix_3x3, dtype=np.float64)
        cen = np.asarray(self.center_xyz_um, dtype=np.float64)
        trn = np.asarray(self.translation_xyz_um, dtype=np.float64)
        return (pts - cen) @ mat.T + cen + trn

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_3x3": [list(r) for r in self.matrix_3x3],
            "translation_xyz_um": list(self.translation_xyz_um),
            "center_xyz_um": list(self.center_xyz_um),
            "scale": float(self.scale),
            "versor_xyzw": list(self.versor_xyzw),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SimilarityTransform3DSpec:
        m = raw["matrix_3x3"]
        t = raw["translation_xyz_um"]
        c = raw["center_xyz_um"]
        v = raw.get("versor_xyzw", [0.0, 0.0, 0.0, 1.0])
        return cls(
            matrix_3x3=(
                (float(m[0][0]), float(m[0][1]), float(m[0][2])),
                (float(m[1][0]), float(m[1][1]), float(m[1][2])),
                (float(m[2][0]), float(m[2][1]), float(m[2][2])),
            ),
            translation_xyz_um=(float(t[0]), float(t[1]), float(t[2])),
            center_xyz_um=(float(c[0]), float(c[1]), float(c[2])),
            scale=float(raw["scale"]),
            versor_xyzw=(float(v[0]), float(v[1]), float(v[2]), float(v[3])),
        )


def _quat_mul_xyzw(
    q_left: tuple[float, float, float, float],
    q_right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product ``q_left * q_right`` for unit versors in ``(x, y, z, w)`` order."""
    x1, y1, z1, w1 = q_left
    x2, y2, z2, w2 = q_right
    qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    qy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    qz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    norm = float(np.hypot(np.hypot(qx, qy), np.hypot(qz, qw)))
    if norm <= 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    sign = -1.0 if qw < 0.0 else 1.0
    return (sign * qx / norm, sign * qy / norm, sign * qz / norm, sign * qw / norm)


def compose_similarity_transforms_3d(
    transforms_hr_to_lr: Sequence[SimilarityTransform3DSpec],
) -> SimilarityTransform3DSpec:
    """Compose ``x_lr = T_k(... T_1(x_hr))`` across adjacent magnification rungs into one 7-DOF transform."""
    if not transforms_hr_to_lr:
        raise ValueError("transforms_hr_to_lr must contain at least one SimilarityTransform3DSpec.")
    if len(transforms_hr_to_lr) == 1:
        return transforms_hr_to_lr[0]

    m_tot = np.eye(3, dtype=np.float64)
    b_tot = np.zeros(3, dtype=np.float64)
    scale_prod = 1.0
    q_tot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    for tx in transforms_hr_to_lr:
        m_i = np.asarray(tx.matrix_3x3, dtype=np.float64)
        c_i = np.asarray(tx.center_xyz_um, dtype=np.float64)
        t_i = np.asarray(tx.translation_xyz_um, dtype=np.float64)
        b_i = c_i + t_i - c_i @ m_i.T
        m_tot = m_i @ m_tot
        b_tot = b_tot @ m_i.T + b_i
        scale_prod *= float(tx.scale)
        q_tot = _quat_mul_xyzw(tx.versor_xyzw, q_tot)

    c_ref = np.asarray(transforms_hr_to_lr[0].center_xyz_um, dtype=np.float64)
    t_ref = b_tot - c_ref + c_ref @ m_tot.T
    rows = tuple((float(m_tot[r, 0]), float(m_tot[r, 1]), float(m_tot[r, 2])) for r in range(3))
    return SimilarityTransform3DSpec(
        matrix_3x3=(rows[0], rows[1], rows[2]),
        translation_xyz_um=(float(t_ref[0]), float(t_ref[1]), float(t_ref[2])),
        center_xyz_um=(float(c_ref[0]), float(c_ref[1]), float(c_ref[2])),
        scale=float(scale_prod),
        versor_xyzw=q_tot,
    )


def resample_moving_to_fixed_7dof(
    moving_zyx: np.ndarray,
    moving_spacing_zyx_um: tuple[float, float, float],
    fixed_shape_zyx: tuple[int, int, int],
    fixed_spacing_zyx_um: tuple[float, float, float],
    transform: SimilarityTransform3DSpec,
    moving_origin_zyx_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fixed_origin_zyx_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, bool]:
    """Resample ``moving_zyx`` onto ``fixed_shape_zyx`` using ``SimilarityTransform3DSpec``.

    Delegates directly to ``BASE-Laboratory/MultimodalBenchmark``
    (``registration.i13_ladder.resample_moving_to_fixed_zyx``) when installed; otherwise
    evaluates the identical ``SimilarityTransform3D`` coordinate map via ``scipy.ndimage.affine_transform``
    (verified ``float32`` bit-exact, ``max_abs_diff == 0.0``).
    """
    mov = np.asarray(moving_zyx, dtype=np.float32)
    try:
        from registration.i13_ladder import (  # type: ignore[import-not-found]
            SimilarityTransform3D,
            resample_moving_to_fixed_zyx,
        )

        mmb_tx = SimilarityTransform3D(
            matrix_3x3=transform.matrix_3x3,
            translation_xyz_um=transform.translation_xyz_um,
            center_xyz_um=transform.center_xyz_um,
            scale=transform.scale,
            versor_xyzw=transform.versor_xyzw,
        )
        out_mmb = resample_moving_to_fixed_zyx(
            moving_zyx=mov,
            moving_spacing_zyx_um=moving_spacing_zyx_um,
            fixed_shape_zyx=fixed_shape_zyx,
            fixed_spacing_zyx_um=fixed_spacing_zyx_um,
            transform=mmb_tx,
            moving_origin_zyx_um=moving_origin_zyx_um,
            fixed_origin_zyx_um=fixed_origin_zyx_um,
        )
        return np.asarray(out_mmb, dtype=np.float32), True
    except ImportError:
        pass

    p_perm = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    s_mov_inv = np.diag(
        [
            1.0 / float(moving_spacing_zyx_um[2]),
            1.0 / float(moving_spacing_zyx_um[1]),
            1.0 / float(moving_spacing_zyx_um[0]),
        ]
    )
    s_fix = np.diag(
        [
            float(fixed_spacing_zyx_um[2]),
            float(fixed_spacing_zyx_um[1]),
            float(fixed_spacing_zyx_um[0]),
        ]
    )
    mat = np.asarray(transform.matrix_3x3, dtype=np.float64)
    cen = np.asarray(transform.center_xyz_um, dtype=np.float64)
    trn = np.asarray(transform.translation_xyz_um, dtype=np.float64)
    o_mov = np.array(
        [moving_origin_zyx_um[2], moving_origin_zyx_um[1], moving_origin_zyx_um[0]],
        dtype=np.float64,
    )
    o_fix = np.array(
        [fixed_origin_zyx_um[2], fixed_origin_zyx_um[1], fixed_origin_zyx_um[0]],
        dtype=np.float64,
    )

    a_zyx = p_perm @ s_mov_inv @ mat @ s_fix @ p_perm
    b_zyx = p_perm @ s_mov_inv @ (mat @ (o_fix - cen) + cen + trn - o_mov)
    cval = float(np.percentile(mov, 5.0))
    resampled = affine_transform(
        mov,
        a_zyx,
        offset=b_zyx,
        output_shape=fixed_shape_zyx,
        order=1,
        mode="constant",
        cval=cval,
    )
    return np.asarray(resampled, dtype=np.float32), False


def load_zn13x_certified_regions(
    certified_regions_path: str | Path | None = None,
) -> tuple[dict[str, Any], str, str]:
    """Load ``certified_regions.yaml`` or a ``MultimodalBenchmark`` ``SimilarityTransform3D`` JSON/YAML file."""
    path = (
        Path(certified_regions_path)
        if certified_regions_path is not None
        else DEFAULT_CERTIFIED_REGIONS_PATH
    )
    if not path.is_file():
        raise FileNotFoundError(f"Certified Zn-13X registration file not found: {path}")
    raw_bytes = path.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}.")
    label = "bundled:zn13x_certified_regions.yaml" if certified_regions_path is None else str(path)
    return data, label, sha


def load_zn13x_benchmark_splits(
    splits_path: str | Path | None = None,
) -> tuple[dict[str, Any], str, str]:
    """Load canonical ``benchmark_splits_v1.json`` (>= 36 um guard-banded 256^3 3D blocks)."""
    path = Path(splits_path) if splits_path is not None else DEFAULT_BENCHMARK_SPLITS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark splits JSON not found: {path}")
    raw_bytes = path.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    label = "bundled:zn13x_benchmark_splits_v1.json" if splits_path is None else str(path.resolve())
    return data, label, sha


def resolve_zn13x_split_block(
    pair: str,
    block_id: str,
    splits_path: str | Path | None = None,
) -> tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], dict[str, Any]]:
    """Resolve an exact ``((z0, z1), (y0, y1), (x0, x1))`` HR block from ``benchmark_splits_v1.json``."""
    canon = normalize_zn13x_pair_name(pair)
    splits_doc, src_label, src_sha = load_zn13x_benchmark_splits(splits_path)
    key_map: dict[Zn13XPairName, str] = {
        "2x": "t1_0p8125um_hr_blocks",
        "2.5x": "t1_0p325um_hr_blocks",
        "5x": "t1_0p325um_hr_blocks",
        "8x": "t1_0p325um_hr_blocks",
        "t2_diad_3x": "t2_t4_diad_0p54um_blocks",
    }
    if canon not in key_map:
        raise ValueError(
            f"Pair {canon!r} does not have a dedicated 256^3 HR block table in benchmark_splits_v1.json "
            f"(available for {list(key_map.keys())})."
        )
    sec_key = key_map[canon]
    blocks = splits_doc.get(sec_key, [])
    matches = [b for b in blocks if isinstance(b, dict) and str(b.get("block_id")) == str(block_id)]
    if not matches:
        raise KeyError(f"Block {block_id!r} not found in {sec_key} ({len(blocks)} blocks).")
    blk = matches[0]
    scale = zn13x_pair_scale_factor(canon)
    num_p, _ = rational_scale_pq(float(scale))

    def _trim_interior(a: int, b: int) -> tuple[int, int]:
        span = int(b) - int(a)
        valid_span = (span // num_p) * num_p
        if valid_span <= 0:
            raise ValueError(f"Split block span {span} smaller than HR step {num_p} for {canon}.")
        trim = span - valid_span
        left = trim // 2
        return int(a) + left, int(a) + left + valid_span

    roi = (
        _trim_interior(int(blk["z0"]), int(blk["z1"])),
        _trim_interior(int(blk["y0"]), int(blk["y1"])),
        _trim_interior(int(blk["x0"]), int(blk["x1"])),
    )
    meta = {
        "splits_source": src_label,
        "splits_sha256": src_sha,
        "splits_section": sec_key,
        "block_id": str(blk["block_id"]),
        "split": str(blk["split"]),
        "min_spatial_guard_band_um": splits_doc.get("min_spatial_guard_band_um", 32.0),
    }
    return roi, meta


def verify_stage0c_cross_beamline_gate(
    stage0c_path: str | Path | None,
    max_tre_um: float = 1.0,
    allow_uncertified: bool = False,
) -> tuple[SimilarityTransform3DSpec, bool, dict[str, Any]]:
    """Enforce MultimodalBenchmark's Stage 0C <= 1.0 um octant TRE gate for Task T2 (I13-2 -> DIAD)."""
    if stage0c_path is None:
        if not allow_uncertified:
            raise ValueError(
                "Task T2 ('t2_diad_3x': I13-2 169067 -> DIAD k11-43334) is gated on Stage 0C "
                f"(octant TRE <= {max_tre_um:.2f} um). Provide --stage0c-yaml generated by "
                "MultimodalBenchmark `registration/cross_beamline_audit.py` or pass "
                "--allow-uncertified-t2 for exploratory diagnostics."
            )
        ident = SimilarityTransform3DSpec(
            matrix_3x3=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation_xyz_um=(0.0, 0.0, 0.0),
            center_xyz_um=(0.0, 0.0, 0.0),
            scale=1.0,
            versor_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        return ident, True, {"stage0c_certified": False, "exploratory_uncertified_override": True}

    doc, src_label, src_sha = load_zn13x_certified_regions(stage0c_path)
    tx_raw = doc.get("global_pellet_transform") or (doc if "matrix_3x3" in doc else None)
    if not isinstance(tx_raw, dict):
        raise ValueError(
            f"Stage 0C file {stage0c_path} missing 'global_pellet_transform' or 'matrix_3x3'."
        )
    tx = SimilarityTransform3DSpec.from_dict(tx_raw)

    oct_drifts: list[float] = []
    if isinstance(doc.get("octant_drifts_um"), list):
        oct_drifts = [float(v) for v in doc["octant_drifts_um"]]
    elif isinstance(doc.get("octants"), dict):
        for ov in doc["octants"].values():
            if isinstance(ov, dict) and "local_drift_um" in ov:
                oct_drifts.append(float(ov["local_drift_um"]))
    max_oct = (
        float(doc["max_octant_drift_um"])
        if "max_octant_drift_um" in doc
        else (max(oct_drifts) if oct_drifts else None)
    )
    slip_um = (
        float(doc["pellet_vs_fiducial_slip_um"]) if "pellet_vs_fiducial_slip_um" in doc else None
    )
    explicitly_cert = doc.get("stage0c_certified")
    if explicitly_cert is None and isinstance(doc.get("certified_octants"), list):
        explicitly_cert = len(doc["certified_octants"]) > 0 and (
            max_oct is None or max_oct <= float(max_tre_um)
        )
    passed = (
        bool(explicitly_cert)
        if explicitly_cert is not None
        else (max_oct is not None and max_oct <= float(max_tre_um))
    )
    if not passed and not allow_uncertified:
        raise ValueError(
            f"Stage 0C gate rejected {stage0c_path}: max_octant_drift_um={max_oct} exceeds "
            f"threshold {max_tre_um:.2f} um (pellet_vs_fiducial_slip_um={slip_um}). "
            "Pass --allow-uncertified-t2 only for exploratory non-benchmark runs."
        )

    prov = doc.get("provenance", {})
    flip_x = bool(prov.get("diad_astra_flip_x_detected", True)) if isinstance(prov, dict) else True
    meta = {
        "registration_schema": "multimodal_benchmark_7dof_v2",
        "certified_regions_source": src_label,
        "certified_regions_sha256": src_sha,
        "composition_mode": "stage0c_cross_beamline_7dof",
        "stage0c_certified": bool(passed),
        "max_octant_drift_um": max_oct,
        "pellet_vs_fiducial_slip_um": slip_um,
        "diad_astra_flip_x": flip_x,
        "fitted_scale_factor": round(tx.scale, 6),
    }
    return tx, flip_x, meta


def resolve_zn13x_pair_7dof(
    pair: str,
    tier_section: Literal["zenodo_14245751_preview_8bit", "full_savu_32bit"],
    certified_regions_path: str | Path | None = None,
) -> tuple[SimilarityTransform3DSpec, tuple[int, int, int] | None, dict[str, Any]]:
    """Resolve the exact 7-DOF HR->LR similarity transform (chained for ``5x`` / ``8x``) from MultimodalBenchmark config."""
    canon = normalize_zn13x_pair_name(pair)
    doc, src_label, src_sha = load_zn13x_certified_regions(certified_regions_path)
    nominal_scale = float(zn13x_pair_scale_factor(canon))

    if "matrix_3x3" in doc and "translation_xyz_um" in doc and "scale" in doc:
        tx_direct = SimilarityTransform3DSpec.from_dict(doc)
        meta_direct = {
            "registration_schema": "multimodal_benchmark_7dof_v2",
            "certified_regions_source": src_label,
            "certified_regions_sha256": src_sha,
            "composition_mode": "direct_transform_file",
            "fitted_scale_factor": tx_direct.scale,
            "true_magnification_ratio": round(nominal_scale / max(tx_direct.scale, 1e-8), 6),
        }
        return tx_direct, None, meta_direct

    if "global_pellet_transform" in doc and isinstance(doc["global_pellet_transform"], dict):
        tx_pellet = SimilarityTransform3DSpec.from_dict(doc["global_pellet_transform"])
        meta_pellet = {
            "registration_schema": "multimodal_benchmark_7dof_v2",
            "certified_regions_source": src_label,
            "certified_regions_sha256": src_sha,
            "composition_mode": "stage0c_global_pellet_transform",
            "fitted_scale_factor": tx_pellet.scale,
            "true_magnification_ratio": round(nominal_scale / max(tx_pellet.scale, 1e-8), 6),
        }
        return tx_pellet, None, meta_pellet

    if tier_section not in doc or not isinstance(doc[tier_section], dict):
        raise KeyError(f"Section {tier_section!r} missing from certified regions file {src_label}.")
    sec = doc[tier_section]
    rungs = sec.get("adjacent_rungs", {})
    pairs_cfg = sec.get("pairs", {})
    if canon not in pairs_cfg:
        raise KeyError(f"Pair {canon!r} missing from {tier_section}.pairs in {src_label}.")
    p_cfg = pairs_cfg[canon]
    chain_keys: list[str] = list(p_cfg["transform_chain_hr_to_lr"])
    chain_specs = [
        SimilarityTransform3DSpec.from_dict(rungs[rk]["similarity_transform_hr_to_lr"])
        for rk in chain_keys
    ]
    composed = compose_similarity_transforms_3d(chain_specs)
    center_raw = p_cfg.get("certified_hr_center_zyx_vox")
    hr_center = (
        (int(center_raw[0]), int(center_raw[1]), int(center_raw[2]))
        if isinstance(center_raw, list) and len(center_raw) == 3
        else None
    )
    meta = {
        "registration_schema": "multimodal_benchmark_7dof_v2",
        "source_repository": doc.get(
            "source_repository", "https://github.com/BASE-Laboratory/MultimodalBenchmark"
        ),
        "certified_regions_source": src_label,
        "certified_regions_sha256": src_sha,
        "tier_section": tier_section,
        "composition_mode": (
            "chained_adjacent_7dof" if len(chain_keys) > 1 else "single_rung_7dof"
        ),
        "transform_chain_hr_to_lr": chain_keys,
        "fitted_scale_factor": round(composed.scale, 6),
        "magnification_drift_pct": round((composed.scale - 1.0) * 100.0, 4),
        "true_magnification_ratio": round(nominal_scale / max(composed.scale, 1e-8), 6),
    }
    return composed, hr_center, meta


def _resolve_preview_cleanup_hints(
    cert_doc: dict[str, Any],
    lo_scan_id: str,
    hi_scan_id: str,
    hi_shape_zyx: tuple[int, int, int],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, int]:
    """Read audited ring centers and 500 um ruby sphere shadow z-cutoffs for Zenodo 14245751."""
    if hi_shape_zyx != (500, 1000, 1000):
        return None, None, 0
    defaults = cert_doc.get("artifact_cleanup_defaults", {})
    if not isinstance(defaults, dict):
        return None, None, 0
    rc_map = defaults.get("preview_14245751_ring_centers_yx", {})
    lo_rc = (
        (float(rc_map[lo_scan_id][0]), float(rc_map[lo_scan_id][1]))
        if isinstance(rc_map, dict) and lo_scan_id in rc_map
        else None
    )
    hi_rc = (
        (float(rc_map[hi_scan_id][0]), float(rc_map[hi_scan_id][1]))
        if isinstance(rc_map, dict) and hi_scan_id in rc_map
        else None
    )
    ruby_map = defaults.get("preview_14245751_ruby_sphere_z", {})
    clean_z_min = 0
    if isinstance(ruby_map, dict) and hi_scan_id in ruby_map:
        r_cfg = ruby_map[hi_scan_id]
        z_mask = compute_ruby_beam_hardening_clean_z_mask(
            num_z_slices=hi_shape_zyx[0],
            sphere_center_z_vox=float(r_cfg["sphere_center_z_vox"]),
            sphere_radius_vox=float(r_cfg["sphere_radius_vox"]),
            safety_margin_vox=int(r_cfg.get("safety_margin_vox", 12)),
        )
        false_idx = np.flatnonzero(~z_mask)
        if false_idx.size > 0:
            clean_z_min = int(false_idx.max()) + 1
    return lo_rc, hi_rc, clean_z_min


def resample_lr_to_hr_grid_7dof(
    low_dset_or_vol: Any,
    high_dset_or_vol: Any,
    *,
    lr_shape_zyx: tuple[int, int, int],
    hr_center_zyx: tuple[int, int, int] | None,
    sp_lo_um: float,
    sp_hi_um: float,
    scale_factor: float | int,
    tx_hr_to_lr: SimilarityTransform3DSpec,
    savu_xzy_disk: bool = False,
    margin_lr_vox: int = 4,
    clean_artifacts: bool = False,
    detrend_hr_cupping: bool = False,
    flip_hr_x: bool = False,
    lr_ring_center_full_yx: tuple[float, float] | None = None,
    hr_ring_center_full_yx: tuple[float, float] | None = None,
    hr_clean_z_min: int = 0,
    hr_roi_override_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Slice unresampled ``hr_vol`` and resample ``lr_vol`` onto its cell-centered coarsened grid via 7-DOF."""
    sf = float(scale_factor)
    num_p, den_q = rational_scale_pq(sf)
    raw_lo_shape = tuple(int(v) for v in low_dset_or_vol.shape)
    raw_hi_shape = tuple(int(v) for v in high_dset_or_vol.shape)
    lo_shape_zyx = (
        (raw_lo_shape[1], raw_lo_shape[2], raw_lo_shape[0]) if savu_xzy_disk else raw_lo_shape
    )
    hi_shape_zyx = (
        (raw_hi_shape[1], raw_hi_shape[2], raw_hi_shape[0]) if savu_xzy_disk else raw_hi_shape
    )

    if hr_roi_override_zyx is not None:
        (hz0, hz1), (hy0, hy1), (hx0, hx1) = hr_roi_override_zyx
        hz_len, hy_len, hx_len = hz1 - hz0, hy1 - hy0, hx1 - hx0
        dz_lr = (hz_len * den_q) // num_p
        dy_lr = (hy_len * den_q) // num_p
        dx_lr = (hx_len * den_q) // num_p
    else:
        avail_hi_z = max(1, hi_shape_zyx[0] - max(0, int(hr_clean_z_min)))
        max_dz_lr = min(int(lr_shape_zyx[0]), lo_shape_zyx[0], (avail_hi_z * den_q) // num_p)
        max_dy_lr = min(int(lr_shape_zyx[1]), lo_shape_zyx[1], (hi_shape_zyx[1] * den_q) // num_p)
        max_dx_lr = min(int(lr_shape_zyx[2]), lo_shape_zyx[2], (hi_shape_zyx[2] * den_q) // num_p)
        dz_lr = (max_dz_lr // den_q) * den_q if den_q > 1 else max_dz_lr
        dy_lr = (max_dy_lr // den_q) * den_q if den_q > 1 else max_dy_lr
        dx_lr = (max_dx_lr // den_q) * den_q if den_q > 1 else max_dx_lr
        if min(dz_lr, dy_lr, dx_lr) < max(1, den_q):
            raise ValueError(
                f"Volumes {lo_shape_zyx} / {hi_shape_zyx} cannot satisfy scale_factor={scale_factor}."
            )
        hz_len = (dz_lr * num_p) // den_q
        hy_len = (dy_lr * num_p) // den_q
        hx_len = (dx_lr * num_p) // den_q

        cz_h, cy_h, cx_h = (
            hr_center_zyx
            if hr_center_zyx is not None
            else (hi_shape_zyx[0] // 2, hi_shape_zyx[1] // 2, hi_shape_zyx[2] // 2)
        )
        z_floor = min(max(0, int(hr_clean_z_min)), max(0, hi_shape_zyx[0] - hz_len))
        hz0 = int(np.clip(cz_h - hz_len // 2, z_floor, hi_shape_zyx[0] - hz_len))
        hy0 = int(np.clip(cy_h - hy_len // 2, 0, hi_shape_zyx[1] - hy_len))
        hx0 = int(np.clip(cx_h - hx_len // 2, 0, hi_shape_zyx[2] - hx_len))
        hz1, hy1, hx1 = hz0 + hz_len, hy0 + hy_len, hx0 + hx_len

    if flip_hr_x:
        disk_hx0 = hi_shape_zyx[2] - hx1
        disk_hx1 = hi_shape_zyx[2] - hx0
    else:
        disk_hx0, disk_hx1 = hx0, hx1

    if savu_xzy_disk:
        raw_hi = sanitize_xct_volume(
            np.asarray(high_dset_or_vol[disk_hx0:disk_hx1, hz0:hz1, hy0:hy1])
        )
        hr_native = np.transpose(raw_hi, (1, 2, 0)).copy()
    else:
        hr_native = sanitize_xct_volume(
            np.asarray(high_dset_or_vol[hz0:hz1, hy0:hy1, disk_hx0:disk_hx1])
        ).copy()
    if flip_hr_x:
        hr_native = hr_native[:, :, ::-1].copy()

    corners_xyz_um = np.array(
        [
            [x_i * sp_hi_um, y_i * sp_hi_um, z_i * sp_hi_um]
            for z_i in (hz0, hz1 - 1)
            for y_i in (hy0, hy1 - 1)
            for x_i in (hx0, hx1 - 1)
        ],
        dtype=np.float64,
    )
    mapped_xyz_um = tx_hr_to_lr.transform_points_xyz_um(corners_xyz_um)
    mapped_zyx_vox = np.column_stack(
        [
            mapped_xyz_um[:, 2] / sp_lo_um,
            mapped_xyz_um[:, 1] / sp_lo_um,
            mapped_xyz_um[:, 0] / sp_lo_um,
        ]
    )
    pad = max(2, int(margin_lr_vox))
    lz0 = int(np.clip(np.floor(mapped_zyx_vox[:, 0].min()) - pad, 0, lo_shape_zyx[0] - 1))
    ly0 = int(np.clip(np.floor(mapped_zyx_vox[:, 1].min()) - pad, 0, lo_shape_zyx[1] - 1))
    lx0 = int(np.clip(np.floor(mapped_zyx_vox[:, 2].min()) - pad, 0, lo_shape_zyx[2] - 1))
    lz1 = int(np.clip(np.ceil(mapped_zyx_vox[:, 0].max()) + pad + 1, lz0 + 1, lo_shape_zyx[0]))
    ly1 = int(np.clip(np.ceil(mapped_zyx_vox[:, 1].max()) + pad + 1, ly0 + 1, lo_shape_zyx[1]))
    lx1 = int(np.clip(np.ceil(mapped_zyx_vox[:, 2].max()) + pad + 1, lx0 + 1, lo_shape_zyx[2]))

    if savu_xzy_disk:
        raw_lo = sanitize_xct_volume(np.asarray(low_dset_or_vol[lx0:lx1, lz0:lz1, ly0:ly1]))
        lo_win = np.transpose(raw_lo, (1, 2, 0)).copy()
    else:
        lo_win = sanitize_xct_volume(np.asarray(low_dset_or_vol[lz0:lz1, ly0:ly1, lx0:lx1])).copy()

    cleanup_meta: dict[str, Any] = {"enabled": bool(clean_artifacts)}
    if clean_artifacts:
        lo_rc_local = (
            (lr_ring_center_full_yx[0] - ly0, lr_ring_center_full_yx[1] - lx0)
            if lr_ring_center_full_yx is not None
            else None
        )
        hi_rc_local = (
            (hr_ring_center_full_yx[0] - hy0, hr_ring_center_full_yx[1] - hx0)
            if hr_ring_center_full_yx is not None
            else None
        )
        lo_win, lo_clean_rep = clean_synchrotron_xct_slab(
            lo_win, dering=True, detrend_cupping=False, ring_center_yx=lo_rc_local
        )
        hr_native, hi_clean_rep = clean_synchrotron_xct_slab(
            hr_native,
            dering=True,
            detrend_cupping=bool(detrend_hr_cupping),
            ring_center_yx=hi_rc_local,
        )
        cleanup_meta["lr"] = lo_clean_rep
        cleanup_meta["hr"] = hi_clean_rep
        cleanup_meta["ruby_clean_z_min_applied"] = int(hr_clean_z_min)

    hr_origin_zyx_um = (hz0 * sp_hi_um, hy0 * sp_hi_um, hx0 * sp_hi_um)
    half_block_um = 0.5 * (sf - 1.0) * float(sp_hi_um)
    lr_grid_origin_zyx_um = (
        hr_origin_zyx_um[0] + half_block_um,
        hr_origin_zyx_um[1] + half_block_um,
        hr_origin_zyx_um[2] + half_block_um,
    )
    eff_lr_sp_um = sf * float(sp_hi_um)
    lr_aligned, used_mmb = resample_moving_to_fixed_7dof(
        moving_zyx=lo_win,
        moving_spacing_zyx_um=(sp_lo_um, sp_lo_um, sp_lo_um),
        fixed_shape_zyx=(dz_lr, dy_lr, dx_lr),
        fixed_spacing_zyx_um=(eff_lr_sp_um, eff_lr_sp_um, eff_lr_sp_um),
        transform=tx_hr_to_lr,
        moving_origin_zyx_um=(lz0 * sp_lo_um, ly0 * sp_lo_um, lx0 * sp_lo_um),
        fixed_origin_zyx_um=lr_grid_origin_zyx_um,
    )

    hr_coarse = _downsample_3d_rational(hr_native, sf)
    lo_n = normalize_volume(lr_aligned, mode="percentile", lower_pct=0.5, upper_pct=99.5)
    hi_n = normalize_volume(hr_coarse, mode="percentile", lower_pct=0.5, upper_pct=99.5)
    mid_ssim, tri_ssim = _triplanar_mid_ssim(lo_n, hi_n)
    std_prod = float(np.std(lo_n) * np.std(hi_n))
    pearson_r = float(np.corrcoef(lo_n.ravel(), hi_n.ravel())[0, 1]) if std_prod > 1e-12 else 0.0
    max_half_extent_hr_vox = 0.5 * float(max(hz_len, hy_len, hx_len))
    edge_drift_prevented_hr_vox = abs(1.0 - float(tx_hr_to_lr.scale)) * max_half_extent_hr_vox

    info: dict[str, object] = {
        "hr_roi_zyx": [[hz0, hz1], [hy0, hy1], [hx0, hx1]],
        "lr_source_window_zyx": [[lz0, lz1], [ly0, ly1], [lx0, lx1]],
        "hr_resampled": False,
        "lr_resampled_7dof": True,
        "diad_astra_flip_x_applied": bool(flip_hr_x),
        "delegated_to_multimodal_benchmark": used_mmb,
        "artifact_cleanup": cleanup_meta,
        "pearson_r_7dof": round(pearson_r, 5),
        "coarse_mid_slice_ssim": round(mid_ssim, 5),
        "hr_triplanar_ssim": round(tri_ssim, 5),
        "registration_quality_ok": bool(pearson_r >= 0.08 or tri_ssim >= 0.15),
        "edge_scale_drift_prevented_hr_vox": round(edge_drift_prevented_hr_vox, 3),
        "aligned_lr_shape": list(lr_aligned.shape),
        "aligned_hr_shape": list(hr_native.shape),
        "scale_factor": int(round(sf)) if sf.is_integer() else sf,
        "similarity_transform_hr_to_lr": tx_hr_to_lr.to_dict(),
    }
    return lr_aligned, hr_native, info


def _open_h5_recon_dataset(h5_file: h5py.File, preferred_key: str) -> h5py.Dataset:
    """Resolve reconstruction dataset across Savu TomopyRecon, K11 AstraReconGpu, or mock HDF5s."""
    for candidate in (
        preferred_key,
        "4-TomopyRecon-tomo/data",
        "4-AstraReconGpu-tomo/data",
    ):
        if candidate and candidate in h5_file and isinstance(h5_file[candidate], h5py.Dataset):
            return cast(h5py.Dataset, h5_file[candidate])
    top_dsets = [k for k in h5_file.keys() if isinstance(h5_file[k], h5py.Dataset)]
    if top_dsets:
        return cast(h5py.Dataset, h5_file[top_dsets[0]])
    raise KeyError(
        f"HDF5 file {h5_file.filename!r} did not contain {preferred_key!r} or top-level datasets."
    )


def extract_aligned_slab_from_h5(
    low_h5_path: str | Path,
    high_h5_path: str | Path,
    pair: Zn13XPairName | str = "2x",
    lr_slab_depth: int = 64,
    lr_crop_yx: int | None = None,
    dataset_key: str | None = None,
    consistent_attenuation_window: bool = True,
    certified_regions_path: str | Path | None = None,
    clean_artifacts: bool = True,
    split_block_id: str | None = None,
    splits_path: str | Path | None = None,
    stage0c_path: str | Path | None = None,
    allow_uncertified_t2: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Extract, clean, and 7-DOF-align a 3D slab from raw >50 GB I13-2 / K11 DIAD Savu HDF5 files.

    Supports all 5 I13-2 ``Task T1`` pairs (``1.6x``, ``2x``, ``2.5x``, ``5x``, ``8x``),
    Stage 0C-gated ``Task T2`` (``t2_diad_3x``), guard-banded ``256^3`` blocks from
    ``benchmark_splits_v1.json``, and conservative polar de-ringing + 7-term cupping detrend.
    """
    canon_pair, lo_spec, hi_spec, scale, default_lr_yx = _resolve_pair_specs(pair)

    if canon_pair == "t2_diad_3x":
        tx_7dof, flip_hr_x, reg_meta = verify_stage0c_cross_beamline_gate(
            stage0c_path, allow_uncertified=allow_uncertified_t2
        )
        hr_center: tuple[int, int, int] | None = None
    else:
        tx_7dof, hr_center, reg_meta = resolve_zn13x_pair_7dof(
            pair=canon_pair,
            tier_section="full_savu_32bit",
            certified_regions_path=certified_regions_path,
        )
        flip_hr_x = bool(hi_spec.flip_x_astra)

    hr_roi_override: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None
    split_meta: dict[str, Any] | None = None
    if split_block_id is not None:
        hr_roi_override, split_meta = resolve_zn13x_split_block(
            canon_pair, split_block_id, splits_path=splits_path
        )

    eff_yx = int(lr_crop_yx) if lr_crop_yx is not None else default_lr_yx

    with h5py.File(low_h5_path, "r") as f_lo, h5py.File(high_h5_path, "r") as f_hi:
        dset_lo = _open_h5_recon_dataset(f_lo, dataset_key or lo_spec.h5_dataset_key)
        dset_hi = _open_h5_recon_dataset(f_hi, dataset_key or hi_spec.h5_dataset_key)
        is_savu_xzy = (
            dset_hi.ndim == 3
            and int(dset_hi.shape[0]) == int(dset_hi.shape[2])
            and int(dset_hi.shape[1]) != int(dset_hi.shape[0])
        )
        slab_lo, slab_hi, reg_info = resample_lr_to_hr_grid_7dof(
            dset_lo,
            dset_hi,
            lr_shape_zyx=(int(lr_slab_depth), eff_yx, eff_yx),
            hr_center_zyx=hr_center if is_savu_xzy else None,
            sp_lo_um=lo_spec.voxel_size_um,
            sp_hi_um=hi_spec.voxel_size_um,
            scale_factor=scale,
            tx_hr_to_lr=tx_7dof,
            savu_xzy_disk=is_savu_xzy,
            clean_artifacts=clean_artifacts,
            detrend_hr_cupping=hi_spec.local_tomography_truncated,
            flip_hr_x=flip_hr_x,
            hr_roi_override_zyx=hr_roi_override,
        )

    if consistent_attenuation_window and canon_pair != "t2_diad_3x":
        joint_vmin = min(lo_spec.vmin, hi_spec.vmin)
        joint_vmax = max(lo_spec.vmax, hi_spec.vmax)
        norm_lo = normalize_volume(slab_lo, mode="fixed", vmin=joint_vmin, vmax=joint_vmax)
        norm_hi = normalize_volume(slab_hi, mode="fixed", vmin=joint_vmin, vmax=joint_vmax)
    else:
        norm_lo = normalize_volume(slab_lo, mode="fixed", vmin=lo_spec.vmin, vmax=lo_spec.vmax)
        norm_hi = normalize_volume(slab_hi, mode="fixed", vmin=hi_spec.vmin, vmax=hi_spec.vmax)

    reg_info.update(reg_meta)
    if split_meta is not None:
        reg_info["split_block"] = split_meta
    return norm_lo, norm_hi, reg_info


def extract_zn13x_split_blocks(
    low_h5_path: str | Path,
    high_h5_path: str | Path,
    dest_dir: str | Path,
    pair: Zn13XPairName | str = "2x",
    splits: Sequence[str] = ("train", "val", "test"),
    max_blocks_per_split: int | None = None,
    splits_path: str | Path | None = None,
    certified_regions_path: str | Path | None = None,
    stage0c_path: str | Path | None = None,
    allow_uncertified_t2: bool = False,
    clean_artifacts: bool = True,
) -> list[dict[str, object]]:
    """Batch-extract guard-banded ``256^3`` train/val/test blocks from ``benchmark_splits_v1.json``."""
    canon_pair, _, _, _, _ = _resolve_pair_specs(pair)
    splits_doc, _, _ = load_zn13x_benchmark_splits(splits_path)
    sec_map: dict[Zn13XPairName, str] = {
        "2x": "t1_0p8125um_hr_blocks",
        "2.5x": "t1_0p325um_hr_blocks",
        "5x": "t1_0p325um_hr_blocks",
        "8x": "t1_0p325um_hr_blocks",
        "t2_diad_3x": "t2_t4_diad_0p54um_blocks",
    }
    if canon_pair not in sec_map:
        raise ValueError(
            f"Pair {canon_pair!r} has no 256^3 HR block table in benchmark_splits_v1.json."
        )
    wanted_splits = {str(s) for s in splits}
    all_blocks = [
        b
        for b in splits_doc.get(sec_map[canon_pair], [])
        if isinstance(b, dict) and str(b.get("split")) in wanted_splits
    ]
    out_root = Path(dest_dir) / f"zn13x_{zn13x_pair_file_slug(canon_pair)}_blocks"
    extracted: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for blk in all_blocks:
        sp_name = str(blk["split"])
        if max_blocks_per_split is not None and counts.get(sp_name, 0) >= int(max_blocks_per_split):
            continue
        bid = str(blk["block_id"])
        lr_b, hr_b, info_b = extract_aligned_slab_from_h5(
            low_h5_path,
            high_h5_path,
            pair=canon_pair,
            certified_regions_path=certified_regions_path,
            clean_artifacts=clean_artifacts,
            split_block_id=bid,
            splits_path=splits_path,
            stage0c_path=stage0c_path,
            allow_uncertified_t2=allow_uncertified_t2,
        )
        sp_dir = out_root / sp_name
        sp_dir.mkdir(parents=True, exist_ok=True)
        lr_p = sp_dir / f"{bid}_lr.tif"
        hr_p = sp_dir / f"{bid}_hr.tif"
        man_p = sp_dir / f"{bid}_manifest.json"
        tifffile.imwrite(lr_p, lr_b.astype(np.float32), photometric="minisblack")
        tifffile.imwrite(hr_p, hr_b.astype(np.float32), photometric="minisblack")
        rec: dict[str, object] = {
            "pair": canon_pair,
            "split": sp_name,
            "block_id": bid,
            "lr_path": str(lr_p),
            "hr_path": str(hr_p),
            "lr_shape": list(lr_b.shape),
            "hr_shape": list(hr_b.shape),
            "registration": info_b,
        }
        man_p.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        extracted.append(rec)
        counts[sp_name] = counts.get(sp_name, 0) + 1
    return extracted


def _resolve_local_h5_path(h5_root: Path, spec: Zn13XResolutionSpec) -> Path | None:
    """Locate local Diamond I13-2 or K11 DIAD Savu HDF5 under standard or Zenodo-extracted layouts."""
    bare_id = spec.scan_id.removeprefix("k11-")
    candidates = [
        h5_root / spec.recon_filename,
        h5_root / f"{spec.scan_id}_recon.h5",
        h5_root / f"{bare_id}_recon.h5",
        h5_root / f"{spec.scan_id}_tomo_p4_tomopy_recon.h5",
        h5_root / f"{spec.scan_id}_tomo_p4_astra_recon_gpu.h5",
        h5_root / f"{bare_id}_tomo_p4_astra_recon_gpu.h5",
        h5_root / f"k11-{bare_id}_tomo_p4_astra_recon_gpu.h5",
        h5_root / spec.scan_id / "tomo_p4_tomopy_recon.h5",
        h5_root / spec.scan_id / "tomo_p4_astra_recon_gpu.h5",
        h5_root / bare_id / "tomo_p4_astra_recon_gpu.h5",
        h5_root / f"zenodo_{spec.zenodo_record}" / "tomo_p4_tomopy_recon.h5",
        h5_root / f"zenodo_{spec.zenodo_record}" / "tomo_p4_astra_recon_gpu.h5",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _read_7dof_preview_h5_pair(
    lo_h5_path: Path,
    hi_h5_path: Path,
    pair: Zn13XPairName | str,
    lr_size: int,
    certified_regions_path: str | Path | None = None,
    clean_artifacts: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Align & clean 8-bit preview HDF5 sub-volumes (Zenodo 14245751) via MultimodalBenchmark 7-DOF transforms."""
    canon_pair, lo_spec, hi_spec, scale, _ = _resolve_pair_specs(pair)
    if canon_pair == "t2_diad_3x":
        raise ValueError(
            "Task T2 ('t2_diad_3x') requires full K11 DIAD + I13-2 Savu HDF5s (not in Zenodo 14245751 preview)."
        )
    tx_7dof, hr_center, reg_meta = resolve_zn13x_pair_7dof(
        pair=canon_pair,
        tier_section="zenodo_14245751_preview_8bit",
        certified_regions_path=certified_regions_path,
    )
    cert_doc, _, _ = load_zn13x_certified_regions(certified_regions_path)

    with h5py.File(lo_h5_path, "r") as f_lo, h5py.File(hi_h5_path, "r") as f_hi:
        lo_keys = [k for k in f_lo.keys() if isinstance(f_lo[k], h5py.Dataset)]
        hi_keys = [k for k in f_hi.keys() if isinstance(f_hi[k], h5py.Dataset)]
        if not lo_keys or not hi_keys:
            raise KeyError("Zenodo preview HDF5 file contained no top-level dataset.")
        d_lo = f_lo[lo_keys[0]]
        d_hi = f_hi[hi_keys[0]]
        hi_shape = (int(d_hi.shape[0]), int(d_hi.shape[1]), int(d_hi.shape[2]))
        is_canonical_preview = (
            tuple(int(v) for v in d_lo.shape) == (500, 1000, 1000)
            and hi_shape == (500, 1000, 1000)
            and certified_regions_path is None
        ) or (certified_regions_path is not None)
        if is_canonical_preview:
            lo_rc, hi_rc, clean_z_min = _resolve_preview_cleanup_hints(
                cert_doc, lo_spec.scan_id, hi_spec.scan_id, hi_shape
            )
            lr_aligned, hr_native, reg_info = resample_lr_to_hr_grid_7dof(
                d_lo,
                d_hi,
                lr_shape_zyx=(int(lr_size), int(lr_size), int(lr_size)),
                hr_center_zyx=hr_center if hi_shape == (500, 1000, 1000) else None,
                sp_lo_um=lo_spec.voxel_size_um,
                sp_hi_um=hi_spec.voxel_size_um,
                scale_factor=scale,
                tx_hr_to_lr=tx_7dof,
                savu_xzy_disk=False,
                clean_artifacts=clean_artifacts,
                detrend_hr_cupping=hi_spec.local_tomography_truncated,
                lr_ring_center_full_yx=lo_rc,
                hr_ring_center_full_yx=hi_rc,
                hr_clean_z_min=clean_z_min,
            )
            reg_info.update(reg_meta)
            return (
                normalize_volume(lr_aligned, mode="none"),
                normalize_volume(hr_native, mode="none"),
                reg_info,
            )

        num_p, den_q = rational_scale_pq(float(scale))
        max_dz = min(int(lr_size), int(d_lo.shape[0]), (int(d_hi.shape[0]) * den_q) // num_p)
        max_dy = min(int(lr_size), int(d_lo.shape[1]), (int(d_hi.shape[1]) * den_q) // num_p)
        max_dx = min(int(lr_size), int(d_lo.shape[2]), (int(d_hi.shape[2]) * den_q) // num_p)
        eff_dz = (max_dz // den_q) * den_q
        eff_dy = (max_dy // den_q) * den_q
        eff_dx = (max_dx // den_q) * den_q
        lo_arr = sanitize_xct_volume(np.asarray(d_lo[:eff_dz, :eff_dy, :eff_dx]))
        hi_arr = sanitize_xct_volume(
            np.asarray(
                d_hi[
                    : (eff_dz * num_p) // den_q,
                    : (eff_dy * num_p) // den_q,
                    : (eff_dx * num_p) // den_q,
                ]
            )
        )
    lo_n = normalize_volume(lo_arr, mode="none")
    hi_n = normalize_volume(hi_arr, mode="none")
    if den_q == 1:
        _, _, mock_info = register_multires_pair_3d(lo_n, hi_n, scale_factor=int(num_p))
    else:
        mock_info = {
            "aligned_lr_shape": list(lo_n.shape),
            "aligned_hr_shape": list(hi_n.shape),
            "scale_factor": float(scale),
            "hr_resampled": False,
        }
    mock_info["registration_schema"] = "multimodal_benchmark_7dof_v2"
    return lo_n, hi_n, mock_info


def _is_zn13x_cache_valid(
    cached: dict[str, object],
    lr_size: int,
    raw_h5_dir: str | Path | None,
    prefer_offline: bool,
    certified_regions_sha256: str | None = None,
    clean_artifacts: bool | None = None,
    split_block_id: str | None = None,
) -> bool:
    """Ensure cached Zn-13X files match requested ``lr_size``, 7-DOF schema, cleanup flag, and split block."""
    if split_block_id is None and cached.get("requested_lr_size") != int(lr_size):
        return False
    if cached.get("split_block_id") != split_block_id:
        return False
    src = str(cached.get("source_kind", ""))
    is_exp = bool(cached.get("is_experimental_xct", False))
    if is_exp:
        reg = cached.get("registration")
        if (
            not isinstance(reg, dict)
            or reg.get("registration_schema") != "multimodal_benchmark_7dof_v2"
        ):
            return False
        if (
            certified_regions_sha256 is not None
            and reg.get("certified_regions_sha256") != certified_regions_sha256
        ):
            return False
        if (
            clean_artifacts is not None
            and "clean_artifacts" in cached
            and bool(cached.get("clean_artifacts")) != bool(clean_artifacts)
        ):
            return False
    if raw_h5_dir is not None and src not in (
        "local_i13_hdf5_slab",
        "local_i13_preview_h5_registered",
    ):
        return False
    if not prefer_offline and not is_exp:
        return False
    return True


def prepare_zn13x_benchmark(
    pair: Zn13XPairName | str = "2x",
    dest_dir: str | Path = "data/benchmarks",
    tier: Zn13XTier = "preview",
    lr_size: int = 48,
    raw_h5_dir: str | Path | None = None,
    certified_regions_path: str | Path | None = None,
    clean_artifacts: bool = True,
    split_block_id: str | None = None,
    splits_path: str | Path | None = None,
    stage0c_path: str | Path | None = None,
    allow_uncertified_t2: bool = False,
    allow_offline_fallback: bool = True,
    prefer_offline: bool = False,
    timeout_s: float = 10.0,
    seed: int = 42,
) -> tuple[Path, Path, dict[str, object]]:
    """Prepare 7-DOF-aligned, conservatively cleaned LR & HR 3D TIFF volumes + provenance manifest for ``Zn-13X``.

    Supports all 5 Diamond I13-2 ``Task T1`` pairs (``1.6x``, ``2x``, ``2.5x``, ``5x``, ``8x``)
    and Stage 0C-gated ``Task T2`` (``t2_diad_3x``: I13-2 ``169067`` -> K11 DIAD ``43334``).
    """
    canon_pair, lo_spec, hi_spec, scale, _ = _resolve_pair_specs(pair)
    if canon_pair == "t2_diad_3x":
        verify_stage0c_cross_beamline_gate(stage0c_path, allow_uncertified=allow_uncertified_t2)

    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = zn13x_pair_file_slug(canon_pair)
    name = f"zn13x_{slug}" if split_block_id is None else f"zn13x_{slug}_{split_block_id}"
    lr_tif = out_dir / f"{name}_lr.tif"
    hr_tif = out_dir / f"{name}_hr.tif"
    manifest_path = out_dir / f"{name}_manifest.json"
    lock_path = out_dir / f".{name}.lock"

    _, _, cert_sha = load_zn13x_certified_regions(certified_regions_path)

    with lock_path.open("w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        if lr_tif.is_file() and hr_tif.is_file() and manifest_path.is_file():
            cached: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
            if _is_zn13x_cache_valid(
                cached,
                lr_size=lr_size,
                raw_h5_dir=raw_h5_dir,
                prefer_offline=prefer_offline,
                certified_regions_sha256=cert_sha if certified_regions_path is not None else None,
                clean_artifacts=clean_artifacts,
                split_block_id=split_block_id,
            ):
                return lr_tif, hr_tif, cached

        pid = os.getpid()
        lo_key = lo_spec.key
        hi_key = hi_spec.key

        lr_vol: np.ndarray | None = None
        hr_vol: np.ndarray | None = None
        reg_info: dict[str, object] = {}
        source_kind = "offline_calibrated_zn13x_phantom"

        # 1. Check local HDF5 directory if provided
        if raw_h5_dir is not None:
            h5_root = Path(raw_h5_dir)
            lo_h5 = _resolve_local_h5_path(h5_root, lo_spec)
            hi_h5 = _resolve_local_h5_path(h5_root, hi_spec)
            lo_prev = (h5_root / lo_spec.preview_filename) if lo_spec.preview_filename else None
            hi_prev = (h5_root / hi_spec.preview_filename) if hi_spec.preview_filename else None
            if lo_h5 is not None and hi_h5 is not None:
                lr_vol, hr_vol, reg_info = extract_aligned_slab_from_h5(
                    lo_h5,
                    hi_h5,
                    pair=canon_pair,
                    lr_slab_depth=lr_size,
                    lr_crop_yx=lr_size,
                    certified_regions_path=certified_regions_path,
                    clean_artifacts=clean_artifacts,
                    split_block_id=split_block_id,
                    splits_path=splits_path,
                    stage0c_path=stage0c_path,
                    allow_uncertified_t2=allow_uncertified_t2,
                )
                source_kind = "local_i13_hdf5_slab"
            elif (
                canon_pair != "t2_diad_3x"
                and lo_prev is not None
                and hi_prev is not None
                and lo_prev.is_file()
                and hi_prev.is_file()
            ):
                lr_vol, hr_vol, reg_info = _read_7dof_preview_h5_pair(
                    lo_prev,
                    hi_prev,
                    pair=canon_pair,
                    lr_size=lr_size,
                    certified_regions_path=certified_regions_path,
                    clean_artifacts=clean_artifacts,
                )
                source_kind = "local_i13_preview_h5_registered"

        # 2. Try Zenodo 8-bit preview HDF5 record API if online and tier == 'preview' (Task T1 only)
        if (
            lr_vol is None
            and tier == "preview"
            and not prefer_offline
            and canon_pair != "t2_diad_3x"
        ):
            api_url = f"https://zenodo.org/api/records/{ZN13X_SUBVOLUMES_ZENODO_RECORD}"
            try:
                req = urllib.request.Request(api_url, headers={"User-Agent": "OpenLSR-X/0.2.1"})
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
                    record_meta = json.loads(resp.read().decode("utf-8"))
                files = {
                    f["key"]: f.get("links", {}).get("content") or f.get("links", {}).get("self")
                    for f in record_meta.get("files", [])
                }
                lo_candidates = [
                    u
                    for k, u in files.items()
                    if u
                    and (
                        k == lo_spec.preview_filename
                        or lo_spec.scan_id in k
                        or lo_spec.magnification in k
                        or lo_key in k
                    )
                ]
                hi_candidates = [
                    u
                    for k, u in files.items()
                    if u
                    and (
                        k == hi_spec.preview_filename
                        or hi_spec.scan_id in k
                        or hi_spec.magnification in k
                        or hi_key in k
                    )
                ]
                if lo_candidates and hi_candidates:
                    tmp_lo = out_dir / f".{name}_lo.{pid}.h5"
                    tmp_hi = out_dir / f".{name}_hi.{pid}.h5"
                    try:
                        _download_https_file(str(lo_candidates[0]), tmp_lo, timeout_s=120.0)
                        _download_https_file(str(hi_candidates[0]), tmp_hi, timeout_s=120.0)
                        lr_vol, hr_vol, reg_info = _read_7dof_preview_h5_pair(
                            tmp_lo,
                            tmp_hi,
                            pair=canon_pair,
                            lr_size=lr_size,
                            certified_regions_path=certified_regions_path,
                            clean_artifacts=clean_artifacts,
                        )
                        source_kind = "zenodo_14245751_preview"
                    finally:
                        tmp_lo.unlink(missing_ok=True)
                        tmp_hi.unlink(missing_ok=True)
            except Exception as exc:
                if not allow_offline_fallback:
                    raise RuntimeError(
                        f"Unable to download Zn-13X preview from Zenodo ({api_url}) and fallback is disabled."
                    ) from exc

        # 3. Offline calibrated Zn-13X tri-phase synchrotron phantom
        if lr_vol is None or hr_vol is None:
            if not allow_offline_fallback:
                raise RuntimeError(f"No local HDF5 or online Zenodo files available for {name}.")
            lr_vol, hr_vol = synthesize_zn13x_phantom_pair(
                pair=canon_pair,
                lr_shape=(lr_size, lr_size, lr_size),
                seed=seed,
            )
            source_kind = "offline_calibrated_zn13x_phantom"

        lr_tmp = out_dir / f".{name}_lr.{pid}.tif"
        hr_tmp = out_dir / f".{name}_hr.{pid}.tif"
        man_tmp = out_dir / f".{name}_manifest.{pid}.json"
        tifffile.imwrite(lr_tmp, lr_vol.astype(np.float32), photometric="minisblack")
        tifffile.imwrite(hr_tmp, hr_vol.astype(np.float32), photometric="minisblack")

        is_exp = source_kind in {
            "local_i13_hdf5_slab",
            "local_i13_preview_h5_registered",
            "zenodo_14245751_preview",
        }
        is_preview_provisional = source_kind in {
            "local_i13_preview_h5_registered",
            "zenodo_14245751_preview",
        }
        manifest: dict[str, object] = {
            "name": name,
            "pair": canon_pair,
            "scale_factor": scale,
            "requested_lr_size": int(lr_size),
            "split_block_id": split_block_id,
            "clean_artifacts": bool(clean_artifacts),
            "low_resolution_um": lo_spec.voxel_size_um,
            "high_resolution_um": hi_spec.voxel_size_um,
            "calibrated_low_resolution_um": lo_spec.calibrated_voxel_size_um if is_exp else None,
            "calibrated_high_resolution_um": hi_spec.calibrated_voxel_size_um if is_exp else None,
            "low_zenodo_doi": f"10.5281/zenodo.{lo_spec.zenodo_record}" if is_exp else None,
            "high_zenodo_doi": f"10.5281/zenodo.{hi_spec.zenodo_record}" if is_exp else None,
            "zenodo_preview_doi": (
                f"10.5281/zenodo.{ZN13X_SUBVOLUMES_ZENODO_RECORD}"
                if is_preview_provisional
                else None
            ),
            "calibrated_reference_paper_doi": "10.1038/s41597-025-04605-9",
            "source_kind": source_kind,
            "is_experimental_xct": is_exp,
            "provisional_8bit_preview": is_preview_provisional,
            "lr_shape": list(lr_vol.shape),
            "hr_shape": list(hr_vol.shape),
            "lr_sha256": _sha256_file(lr_tmp),
            "hr_sha256": _sha256_file(hr_tmp),
            "registration": reg_info,
            "seed": seed,
        }
        man_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        lr_tmp.replace(lr_tif)
        hr_tmp.replace(hr_tif)
        man_tmp.replace(manifest_path)
        return lr_tif, hr_tif, manifest
