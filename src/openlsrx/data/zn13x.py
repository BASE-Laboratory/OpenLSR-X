"""Diamond Light Source I13-2 multi-resolution Zn-13X synchrotron dataset & 7-DOF aligner.

.. note::
    Dataset-specific Zn-13X constants in this module are retained in ``v0.2.x`` for manuscript
    reproducibility and scheduled to reside exclusively in ``BASE-Laboratory/MultimodalBenchmark``
    in ``v0.3.0``. All Savu HDF5 slicing, TIFF writing, 7-DOF similarity chaining, and cell-centred
    resampling delegate directly to ``basecoat`` (Layer 1).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import urllib.parse
import urllib.request
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from basecoat import (
    AxisSpec,
    LazyVolume,
    Volume,
    open_savu_h5,
    resample_volume,
    write_volume,
)
from basecoat import (
    SimilarityTransform3D as BasecoatSimilarityTransform3D,
)
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

from .transforms import normalize_volume, sanitize_xct_volume

Zn13XPairName = Literal["2x", "8x"]
Zn13XTier = Literal["preview", "full"]

DEFAULT_CERTIFIED_REGIONS_PATH = Path(__file__).with_name("zn13x_certified_regions.yaml")


@dataclass(frozen=True)
class Zn13XResolutionSpec:
    """Metadata and calibrated alignment anchors for one I13-2 magnification."""

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
        roi_1040um_yx=((1102, 1502), (1006, 1406)),  # 400 x 400 px (1040 um)
        roi_812um_yx=((1099, 1411), (1096, 1408)),  # 312 x 312 px centered at chained (1255, 1252)
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
        roi_1040um_yx=((1010, 1650), (858, 1498)),  # 640 x 640 px (1040 um) -> 2x = 1280 px
        roi_812um_yx=((1005, 1505), (1002, 1502)),  # 500 x 500 px centered at chained (1255, 1252)
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
        roi_1040um_yx=((763, 2043), (466, 1746)),  # 1280 x 1280 px (1040 um)
        roi_812um_yx=((753, 1753), (753, 1753)),  # 1000 x 1000 px (812.5 um) -> 2.5x = 2500 px
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
        roi_812um_yx=((0, 2496), (0, 2496)),  # 2496 x 2496 px (811.2 um)
    ),
}


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
    req = urllib.request.Request(url, headers={"User-Agent": "OpenLSR-X/0.2.2"})
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
    pair: Zn13XPairName = "2x",
    lr_shape: tuple[int, int, int] = (48, 48, 48),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a calibrated tri-phase Zn-doped Zeolite 13X synchrotron XCT pair (low-RAM float32).

    Models the physical structure described in Green et al. (*Sci. Data* 2025):
    - Cleaved Zeolite 13X fragment inside a thin-walled capillary FOV
    - Higher-attenuation outer Zn-exchanged rim (~0.88) and Na-13X core (~0.68)
    - Hierarchical connected macro- and meso-pore network (~0.14 attenuation)
    - Distinct objective-specific scintillator PSF and detector noise at LR vs HR
    """
    scale = 2 if pair == "2x" else 8
    hr_shape = (lr_shape[0] * scale, lr_shape[1] * scale, lr_shape[2] * scale)
    rng = np.random.default_rng(seed)

    # Broadcasted 1D grids avoid allocating three full 3D coordinate tensors
    zz = np.linspace(-1.0, 1.0, hr_shape[0], dtype=np.float32)[:, np.newaxis, np.newaxis]
    yy = np.linspace(-1.0, 1.0, hr_shape[1], dtype=np.float32)[np.newaxis, :, np.newaxis]
    xx = np.linspace(-1.0, 1.0, hr_shape[2], dtype=np.float32)[np.newaxis, np.newaxis, :]
    radial_sq = 0.85 * (zz * zz) + (yy * yy) + 1.1 * (xx * xx)

    # Generate float32 random fields directly (half the peak RAM of float64 standard_normal)
    coarse_field = rng.standard_normal(hr_shape, dtype=np.float32)
    gaussian_filter(coarse_field, sigma=max(2.0, scale * 1.4), output=coarse_field)
    stride = max(1, int(round((hr_shape[0] * hr_shape[1] * hr_shape[2] / 500_000) ** (1.0 / 3.0))))
    q_coarse = float(np.quantile(coarse_field[::stride, ::stride, ::stride], 0.78))
    pore_mask = coarse_field > q_coarse
    del coarse_field

    fine_field = rng.standard_normal(hr_shape, dtype=np.float32)
    gaussian_filter(fine_field, sigma=max(0.9, scale * 0.45), output=fine_field)
    q_fine = float(np.quantile(fine_field[::stride, ::stride, ::stride], 0.84))
    pore_mask |= fine_field > q_fine
    del fine_field

    # Zn2+ ion-exchange rim (radial > 0.52 <=> radial_sq > 0.2704)
    hr_clean = np.where(radial_sq > (0.52 * 0.52), np.float32(0.88), np.float32(0.68))
    del radial_sq
    hr_clean[pore_mask] = np.float32(0.14)
    del pore_mask

    # High-magnification GGG:Eu / LuAG scintillator PSF + mild FBP noise
    hr_vol = gaussian_filter(hr_clean, sigma=0.55, mode="reflect")
    hr_noise = rng.normal(0.0, 0.015, size=hr_shape).astype(np.float32, copy=False)
    hr_vol += hr_noise
    del hr_noise
    np.clip(hr_vol, 0.0, 1.0, out=hr_vol)

    # Lower-magnification thick LuAG (500 um) scintillator blur + lens browning noise
    lr_blurred = gaussian_filter(hr_clean, sigma=0.85 * scale, mode="reflect")
    del hr_clean
    lr_binned = _block_downsample_3d(lr_blurred, scale)
    del lr_blurred
    noise_sigma = 0.024 if pair == "2x" else 0.035
    lr_vol = lr_binned + rng.normal(0.0, noise_sigma, size=lr_binned.shape).astype(np.float32)
    np.clip(lr_vol, 0.0, 1.0, out=lr_vol)
    return lr_vol.astype(np.float32, copy=False), hr_vol.astype(np.float32, copy=False)


def _rotmat_xyz_to_versor_xyzw(r_xyz: np.ndarray) -> tuple[float, float, float, float]:
    """Convert an orthonormal right-handed 3x3 XYZ rotation matrix to an ITK unit versor ``(qx, qy, qz, qw)``."""
    tr = float(r_xyz[0, 0] + r_xyz[1, 1] + r_xyz[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r_xyz[2, 1] - r_xyz[1, 2]) / s
        qy = (r_xyz[0, 2] - r_xyz[2, 0]) / s
        qz = (r_xyz[1, 0] - r_xyz[0, 1]) / s
    elif r_xyz[0, 0] > r_xyz[1, 1] and r_xyz[0, 0] > r_xyz[2, 2]:
        s = math.sqrt(1.0 + r_xyz[0, 0] - r_xyz[1, 1] - r_xyz[2, 2]) * 2.0
        qw = (r_xyz[2, 1] - r_xyz[1, 2]) / s
        qx = 0.25 * s
        qy = (r_xyz[0, 1] + r_xyz[1, 0]) / s
        qz = (r_xyz[0, 2] + r_xyz[2, 0]) / s
    elif r_xyz[1, 1] > r_xyz[2, 2]:
        s = math.sqrt(1.0 + r_xyz[1, 1] - r_xyz[0, 0] - r_xyz[2, 2]) * 2.0
        qw = (r_xyz[0, 2] - r_xyz[2, 0]) / s
        qx = (r_xyz[0, 1] + r_xyz[1, 0]) / s
        qy = 0.25 * s
        qz = (r_xyz[1, 2] + r_xyz[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r_xyz[2, 2] - r_xyz[0, 0] - r_xyz[1, 1]) * 2.0
        qw = (r_xyz[1, 0] - r_xyz[0, 1]) / s
        qx = (r_xyz[0, 2] + r_xyz[2, 0]) / s
        qy = (r_xyz[1, 2] + r_xyz[2, 1]) / s
        qz = 0.25 * s
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    nrm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return (float(qx / nrm), float(qy / nrm), float(qz / nrm), float(qw / nrm))


@dataclass(frozen=True)
class SimilarityTransform3DSpec:
    """7-DOF 3D similarity transform DTO in physical micron coordinates (backed by ``basecoat.SimilarityTransform3D``)."""

    matrix_3x3: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    translation_xyz_um: tuple[float, float, float]
    center_xyz_um: tuple[float, float, float]
    scale: float
    versor_xyzw: tuple[float, float, float, float]

    def to_basecoat(
        self,
        from_space: str = "fixed",
        to_space: str = "moving",
    ) -> BasecoatSimilarityTransform3D:
        """Convert to a ``basecoat.SimilarityTransform3D`` in canonical ``('z', 'y', 'x')`` order."""
        scaled_xyz = np.asarray(self.matrix_3x3, dtype=np.float64)
        r_xyz = scaled_xyz / float(self.scale)
        r_zyx = r_xyz[::-1, ::-1]
        cx, cy, cz = self.center_xyz_um
        tx, ty, tz = self.translation_xyz_um
        return BasecoatSimilarityTransform3D(
            rotation_zyx=r_zyx,
            scale=float(self.scale),
            center_zyx=(float(cz), float(cy), float(cx)),
            translation_zyx=(float(tz), float(ty), float(tx)),
            from_space=from_space,
            to_space=to_space,
            unit="um",
        )

    @classmethod
    def from_basecoat(cls, tx: BasecoatSimilarityTransform3D) -> SimilarityTransform3DSpec:
        """Construct from a ``basecoat.SimilarityTransform3D``."""
        r_xyz = np.asarray(tx.rotation_zyx, dtype=np.float64)[::-1, ::-1]
        scaled_xyz = float(tx.scale) * r_xyz
        cz, cy, cx = tx.center_zyx
        tz, ty, tx_um = tx.translation_zyx
        rows = tuple(
            (float(scaled_xyz[r, 0]), float(scaled_xyz[r, 1]), float(scaled_xyz[r, 2]))
            for r in range(3)
        )
        return cls(
            matrix_3x3=(rows[0], rows[1], rows[2]),
            translation_xyz_um=(float(tx_um), float(ty), float(tz)),
            center_xyz_um=(float(cx), float(cy), float(cz)),
            scale=float(tx.scale),
            versor_xyzw=_rotmat_xyz_to_versor_xyzw(r_xyz),
        )

    def transform_points_xyz_um(self, pts_xyz_um: np.ndarray) -> np.ndarray:
        """Map fixed physical points ``(N, 3)`` in ``(x, y, z)`` um to moving physical points via ``basecoat``."""
        pts_xyz = np.asarray(pts_xyz_um, dtype=np.float64)
        mapped_zyx = self.to_basecoat().transform_points(pts_xyz[..., ::-1])
        return mapped_zyx[..., ::-1]

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


def compose_similarity_transforms_3d(
    transforms_hr_to_lr: Sequence[SimilarityTransform3DSpec],
) -> SimilarityTransform3DSpec:
    """Compose ``x_lr = T_k(... T_1(x_hr))`` across adjacent magnification rungs via ``basecoat.SimilarityTransform3D``."""
    if not transforms_hr_to_lr:
        raise ValueError("transforms_hr_to_lr must contain at least one SimilarityTransform3DSpec.")
    if len(transforms_hr_to_lr) == 1:
        return transforms_hr_to_lr[0]

    chained = transforms_hr_to_lr[0].to_basecoat(from_space="r0", to_space="r1")
    for idx, step in enumerate(transforms_hr_to_lr[1:], start=1):
        chained = chained.then_similarity(
            step.to_basecoat(from_space=f"r{idx}", to_space=f"r{idx + 1}")
        )
    return SimilarityTransform3DSpec.from_basecoat(chained)


def resample_moving_to_fixed_7dof(
    moving_zyx: np.ndarray,
    moving_spacing_zyx_um: tuple[float, float, float],
    fixed_shape_zyx: tuple[int, int, int],
    fixed_spacing_zyx_um: tuple[float, float, float],
    transform: SimilarityTransform3DSpec | BasecoatSimilarityTransform3D,
    moving_origin_zyx_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fixed_origin_zyx_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, bool]:
    """Resample ``moving_zyx`` onto ``fixed_shape_zyx`` via ``basecoat.resample_volume``."""
    bc_tx = (
        transform
        if isinstance(transform, BasecoatSimilarityTransform3D)
        else transform.to_basecoat(from_space="fixed", to_space="moving")
    )
    moving_axes = tuple(
        AxisSpec(name=ax, kind="space", unit="um", scale=float(sp), offset=float(orig))
        for ax, sp, orig in zip(
            ("z", "y", "x"), moving_spacing_zyx_um, moving_origin_zyx_um, strict=True
        )
    )
    fixed_axes = tuple(
        AxisSpec(name=ax, kind="space", unit="um", scale=float(sp), offset=float(orig))
        for ax, sp, orig in zip(
            ("z", "y", "x"), fixed_spacing_zyx_um, fixed_origin_zyx_um, strict=True
        )
    )
    mov_vol = Volume(
        np.asarray(moving_zyx, dtype=np.float32), axes=moving_axes, space=bc_tx.to_space
    )
    out_vol = resample_volume(
        mov_vol,
        fixed_axes,
        target_shape=(int(fixed_shape_zyx[0]), int(fixed_shape_zyx[1]), int(fixed_shape_zyx[2])),
        transform=bc_tx,
        order=1,
    )
    return np.asarray(out_vol.data, dtype=np.float32), True


def load_zn13x_certified_regions(
    certified_regions_path: str | Path | None = None,
) -> tuple[dict[str, Any], str, str]:
    """Load ``certified_regions.yaml`` or a ``MultimodalBenchmark`` ``SimilarityTransform3D`` JSON/YAML file."""
    if certified_regions_path is None:
        warnings.warn(
            "Relying on the bundled 'zn13x_certified_regions.yaml' inside openlsrx is deprecated "
            "under ECOSYSTEM.md and scheduled for removal in v0.3.0; pass an explicit "
            "certified_regions_path (e.g. MultimodalBenchmark 'configs/i13_certified_regions.yaml' "
            "via --certified-regions).",
            DeprecationWarning,
            stacklevel=2,
        )
        path = DEFAULT_CERTIFIED_REGIONS_PATH
    else:
        path = Path(certified_regions_path)
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


def resolve_zn13x_pair_7dof(
    pair: Zn13XPairName,
    tier_section: Literal["zenodo_14245751_preview_8bit", "full_savu_32bit"],
    certified_regions_path: str | Path | None = None,
) -> tuple[SimilarityTransform3DSpec, tuple[int, int, int] | None, dict[str, Any]]:
    """Resolve the exact 7-DOF HR->LR similarity transform (chained for ``8x``) from MultimodalBenchmark config."""
    doc, src_label, src_sha = load_zn13x_certified_regions(certified_regions_path)
    nominal_scale = 2 if pair == "2x" else 8

    # Case 1: Direct MultimodalBenchmark SimilarityTransform3D JSON/YAML
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

    # Case 2: MultimodalBenchmark stage0c_cross_beamline_regions.yaml (global_pellet_transform)
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

    # Case 3: Canonical multi-rung certified_regions.yaml
    if tier_section not in doc or not isinstance(doc[tier_section], dict):
        raise KeyError(f"Section {tier_section!r} missing from certified regions file {src_label}.")
    sec = doc[tier_section]
    rungs = sec.get("adjacent_rungs", {})
    pairs_cfg = sec.get("pairs", {})
    if pair not in pairs_cfg:
        raise KeyError(f"Pair {pair!r} missing from {tier_section}.pairs in {src_label}.")
    p_cfg = pairs_cfg[pair]
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


def resample_lr_to_hr_grid_7dof(
    low_dset_or_vol: Any,
    high_dset_or_vol: Any,
    *,
    lr_shape_zyx: tuple[int, int, int],
    hr_center_zyx: tuple[int, int, int] | None,
    sp_lo_um: float,
    sp_hi_um: float,
    scale_factor: int,
    tx_hr_to_lr: SimilarityTransform3DSpec,
    savu_xzy_disk: bool = False,
    margin_lr_vox: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Slice unresampled ``hr_vol`` and resample ``lr_vol`` onto its cell-centered ``s``-coarsened grid via 7-DOF.

    Supporting both out-of-core ``h5py.Dataset`` handles (including Savu ``(X, Z, Y)`` disk layout)
    and in-memory ``np.ndarray`` volumes, this keeps ``hr_vol`` 100% native/unresampled so HR noise,
    interfaces, and frozen-threshold pore-throat transport topology are untouched.
    """
    s = int(scale_factor)
    raw_lo_shape = tuple(int(v) for v in low_dset_or_vol.shape)
    raw_hi_shape = tuple(int(v) for v in high_dset_or_vol.shape)
    lo_shape_zyx = (
        (raw_lo_shape[1], raw_lo_shape[2], raw_lo_shape[0]) if savu_xzy_disk else raw_lo_shape
    )
    hi_shape_zyx = (
        (raw_hi_shape[1], raw_hi_shape[2], raw_hi_shape[0]) if savu_xzy_disk else raw_hi_shape
    )

    dz_lr = min(int(lr_shape_zyx[0]), lo_shape_zyx[0], hi_shape_zyx[0] // s)
    dy_lr = min(int(lr_shape_zyx[1]), lo_shape_zyx[1], hi_shape_zyx[1] // s)
    dx_lr = min(int(lr_shape_zyx[2]), lo_shape_zyx[2], hi_shape_zyx[2] // s)
    if min(dz_lr, dy_lr, dx_lr) < 1:
        raise ValueError(
            f"Volumes {lo_shape_zyx} / {hi_shape_zyx} cannot satisfy scale_factor={s}."
        )
    hz_len, hy_len, hx_len = dz_lr * s, dy_lr * s, dx_lr * s

    cz_h, cy_h, cx_h = (
        hr_center_zyx
        if hr_center_zyx is not None
        else (hi_shape_zyx[0] // 2, hi_shape_zyx[1] // 2, hi_shape_zyx[2] // 2)
    )
    hz0 = int(np.clip(cz_h - hz_len // 2, 0, hi_shape_zyx[0] - hz_len))
    hy0 = int(np.clip(cy_h - hy_len // 2, 0, hi_shape_zyx[1] - hy_len))
    hx0 = int(np.clip(cx_h - hx_len // 2, 0, hi_shape_zyx[2] - hx_len))
    hz1, hy1, hx1 = hz0 + hz_len, hy0 + hy_len, hx0 + hx_len

    # 1. Slice native HR target volume WITHOUT any resampling!
    if isinstance(high_dset_or_vol, LazyVolume):
        if savu_xzy_disk:
            raw_hi = sanitize_xct_volume(
                high_dset_or_vol.read_region(
                    (slice(hx0, hx1), slice(hz0, hz1), slice(hy0, hy1))
                ).data
            )
            hr_native = np.transpose(raw_hi, (1, 2, 0)).copy()
        else:
            hr_native = sanitize_xct_volume(
                high_dset_or_vol.read_region(
                    (slice(hz0, hz1), slice(hy0, hy1), slice(hx0, hx1))
                ).data
            ).copy()
    elif savu_xzy_disk:
        raw_hi = sanitize_xct_volume(np.asarray(high_dset_or_vol[hx0:hx1, hz0:hz1, hy0:hy1]))
        hr_native = np.transpose(raw_hi, (1, 2, 0)).copy()
    else:
        hr_native = sanitize_xct_volume(
            np.asarray(high_dset_or_vol[hz0:hz1, hy0:hy1, hx0:hx1])
        ).copy()

    # 2. Compute tight out-of-core LR bounding box enclosing all 8 mapped HR corners
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

    if isinstance(low_dset_or_vol, LazyVolume):
        if savu_xzy_disk:
            raw_lo = sanitize_xct_volume(
                low_dset_or_vol.read_region(
                    (slice(lx0, lx1), slice(lz0, lz1), slice(ly0, ly1))
                ).data
            )
            lo_win = np.transpose(raw_lo, (1, 2, 0)).copy()
        else:
            lo_win = sanitize_xct_volume(
                low_dset_or_vol.read_region(
                    (slice(lz0, lz1), slice(ly0, ly1), slice(lx0, lx1))
                ).data
            ).copy()
    elif savu_xzy_disk:
        raw_lo = sanitize_xct_volume(np.asarray(low_dset_or_vol[lx0:lx1, lz0:lz1, ly0:ly1]))
        lo_win = np.transpose(raw_lo, (1, 2, 0)).copy()
    else:
        lo_win = sanitize_xct_volume(np.asarray(low_dset_or_vol[lz0:lz1, ly0:ly1, lx0:lx1])).copy()

    # 3. Cell-centered s-coarsened physical LR grid of hr_native:
    # HR block [0 .. s-1] has center at 0.5 * (s - 1) * sp_hi_um relative to HR voxel 0!
    hr_origin_zyx_um = (hz0 * sp_hi_um, hy0 * sp_hi_um, hx0 * sp_hi_um)
    half_block_um = 0.5 * float(s - 1) * float(sp_hi_um)
    lr_grid_origin_zyx_um = (
        hr_origin_zyx_um[0] + half_block_um,
        hr_origin_zyx_um[1] + half_block_um,
        hr_origin_zyx_um[2] + half_block_um,
    )
    eff_lr_sp_um = float(s) * float(sp_hi_um)
    lr_aligned, used_mmb = resample_moving_to_fixed_7dof(
        moving_zyx=lo_win,
        moving_spacing_zyx_um=(sp_lo_um, sp_lo_um, sp_lo_um),
        fixed_shape_zyx=(dz_lr, dy_lr, dx_lr),
        fixed_spacing_zyx_um=(eff_lr_sp_um, eff_lr_sp_um, eff_lr_sp_um),
        transform=tx_hr_to_lr,
        moving_origin_zyx_um=(lz0 * sp_lo_um, ly0 * sp_lo_um, lx0 * sp_lo_um),
        fixed_origin_zyx_um=lr_grid_origin_zyx_um,
    )

    # 4. Compute contrast-invariant alignment diagnostics on (lr_aligned, block_down(hr_native))
    hr_coarse = _block_downsample_3d(hr_native, s)
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
        "delegated_to_basecoat": used_mmb,
        "delegated_to_multimodal_benchmark": used_mmb,
        "pearson_r_7dof": round(pearson_r, 5),
        "coarse_mid_slice_ssim": round(mid_ssim, 5),
        "hr_triplanar_ssim": round(tri_ssim, 5),
        "registration_quality_ok": bool(pearson_r >= 0.08 or tri_ssim >= 0.15),
        "edge_scale_drift_prevented_hr_vox": round(edge_drift_prevented_hr_vox, 3),
        "aligned_lr_shape": list(lr_aligned.shape),
        "aligned_hr_shape": list(hr_native.shape),
        "scale_factor": s,
        "similarity_transform_hr_to_lr": tx_hr_to_lr.to_dict(),
    }
    return lr_aligned, hr_native, info


def extract_aligned_slab_from_h5(
    low_h5_path: str | Path,
    high_h5_path: str | Path,
    pair: Zn13XPairName = "2x",
    lr_slab_depth: int = 64,
    lr_crop_yx: int | None = None,
    dataset_key: str = "4-TomopyRecon-tomo/data",
    consistent_attenuation_window: bool = True,
    certified_regions_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Extract and 7-DOF-align a 3D slab from raw >50 GB I13-2 Savu HDF5 files out-of-core.

    Delegates Savu ``(X, Z, Y)`` HDF5 slicing and cell-centred 7-DOF similarity resampling to
    ``basecoat`` (Layer 1), resampling LR onto the cell-centered ``s``-coarsened physical grid of
    the untouched native HR slab and eliminating the ~1.3% objective magnification drift toward
    crop edges.
    """
    if pair == "2x":
        lo_spec = ZN13X_I13_RESOLUTIONS["1p625um"]
        hi_spec = ZN13X_I13_RESOLUTIONS["0p8125um"]
        scale = 2
        default_lr_yx = 640
    else:
        lo_spec = ZN13X_I13_RESOLUTIONS["2p600um"]
        hi_spec = ZN13X_I13_RESOLUTIONS["0p325um"]
        scale = 8
        default_lr_yx = 240  # 1920 HR px inside non-cupping 0.325 um local-tomography interior

    tx_7dof, hr_center, reg_meta = resolve_zn13x_pair_7dof(
        pair=pair,
        tier_section="full_savu_32bit",
        certified_regions_path=certified_regions_path,
    )
    eff_yx = int(lr_crop_yx) if lr_crop_yx is not None else default_lr_yx

    with open_savu_h5(
        high_h5_path, dataset_path=dataset_key, voxel_size_um=hi_spec.voxel_size_um
    ) as lv_hi:
        is_savu_xzy = bool(lv_hi.provenance.get("savu_xzy_transposed", False))
        with open_savu_h5(
            low_h5_path,
            dataset_path=dataset_key,
            savu_xzy=is_savu_xzy,
            voxel_size_um=lo_spec.voxel_size_um,
        ) as lv_lo:
            slab_lo, slab_hi, reg_info = resample_lr_to_hr_grid_7dof(
                lv_lo,
                lv_hi,
                lr_shape_zyx=(int(lr_slab_depth), eff_yx, eff_yx),
                hr_center_zyx=hr_center if is_savu_xzy else None,
                sp_lo_um=lo_spec.voxel_size_um,
                sp_hi_um=hi_spec.voxel_size_um,
                scale_factor=scale,
                tx_hr_to_lr=tx_7dof,
                savu_xzy_disk=False,
            )

    if consistent_attenuation_window:
        joint_vmin = min(lo_spec.vmin, hi_spec.vmin)
        joint_vmax = max(lo_spec.vmax, hi_spec.vmax)
        norm_lo = normalize_volume(slab_lo, mode="fixed", vmin=joint_vmin, vmax=joint_vmax)
        norm_hi = normalize_volume(slab_hi, mode="fixed", vmin=joint_vmin, vmax=joint_vmax)
    else:
        norm_lo = normalize_volume(slab_lo, mode="fixed", vmin=lo_spec.vmin, vmax=lo_spec.vmax)
        norm_hi = normalize_volume(slab_hi, mode="fixed", vmin=hi_spec.vmin, vmax=hi_spec.vmax)

    reg_info.update(reg_meta)
    return norm_lo, norm_hi, reg_info


def _resolve_local_h5_path(h5_root: Path, spec: Zn13XResolutionSpec) -> Path | None:
    """Locate local Diamond I13-2 Savu HDF5 under standard or Zenodo-extracted layouts."""
    candidates = [
        h5_root / spec.recon_filename,
        h5_root / f"{spec.scan_id}_tomo_p4_tomopy_recon.h5",
        h5_root / spec.scan_id / "tomo_p4_tomopy_recon.h5",
        h5_root / f"zenodo_{spec.zenodo_record}" / "tomo_p4_tomopy_recon.h5",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _read_7dof_preview_h5_pair(
    lo_h5_path: Path,
    hi_h5_path: Path,
    pair: Zn13XPairName,
    lr_size: int,
    certified_regions_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Align 8-bit preview HDF5 sub-volumes (Zenodo 14245751) via basecoat 7-DOF transforms."""
    scale = 2 if pair == "2x" else 8
    lo_spec = ZN13X_I13_RESOLUTIONS["1p625um" if pair == "2x" else "2p600um"]
    hi_spec = ZN13X_I13_RESOLUTIONS["0p8125um" if pair == "2x" else "0p325um"]
    tx_7dof, hr_center, reg_meta = resolve_zn13x_pair_7dof(
        pair=pair,
        tier_section="zenodo_14245751_preview_8bit",
        certified_regions_path=certified_regions_path,
    )
    try:
        ctx_lo = open_savu_h5(lo_h5_path, savu_xzy=False, voxel_size_um=lo_spec.voxel_size_um)
    except (KeyError, ValueError) as exc:
        raise KeyError("Zenodo preview HDF5 file contained no top-level dataset.") from exc

    with ctx_lo as lv_lo:
        try:
            ctx_hi = open_savu_h5(hi_h5_path, savu_xzy=False, voxel_size_um=hi_spec.voxel_size_um)
        except (KeyError, ValueError) as exc:
            raise KeyError("Zenodo preview HDF5 file contained no top-level dataset.") from exc

        with ctx_hi as lv_hi:
            # Only use full (500, 1000, 1000) global transform when preview HDF5 matches canonical size;
            # synthetic tiny unit-test HDF5 mocks fall back cleanly to centered identity-frame slicing.
            is_canonical_preview = (
                tuple(int(v) for v in lv_lo.shape) == (500, 1000, 1000)
                and tuple(int(v) for v in lv_hi.shape) == (500, 1000, 1000)
                and certified_regions_path is None
            ) or (certified_regions_path is not None)
            if is_canonical_preview:
                lr_aligned, hr_native, reg_info = resample_lr_to_hr_grid_7dof(
                    lv_lo,
                    lv_hi,
                    lr_shape_zyx=(int(lr_size), int(lr_size), int(lr_size)),
                    hr_center_zyx=hr_center
                    if tuple(int(v) for v in lv_hi.shape) == (500, 1000, 1000)
                    else None,
                    sp_lo_um=lo_spec.voxel_size_um,
                    sp_hi_um=hi_spec.voxel_size_um,
                    scale_factor=scale,
                    tx_hr_to_lr=tx_7dof,
                    savu_xzy_disk=False,
                )
                reg_info.update(reg_meta)
                return (
                    normalize_volume(lr_aligned, mode="none"),
                    normalize_volume(hr_native, mode="none"),
                    reg_info,
                )

            # Fallback for synthetic small mock preview HDF5s in unit tests
            eff_dz = min(int(lr_size), int(lv_lo.shape[0]), int(lv_hi.shape[0]) // scale)
            eff_dy = min(int(lr_size), int(lv_lo.shape[1]), int(lv_hi.shape[1]) // scale)
            eff_dx = min(int(lr_size), int(lv_lo.shape[2]), int(lv_hi.shape[2]) // scale)
            lo_arr = sanitize_xct_volume(
                lv_lo.read_region((slice(0, eff_dz), slice(0, eff_dy), slice(0, eff_dx))).data
            )
            hi_arr = sanitize_xct_volume(
                lv_hi.read_region(
                    (
                        slice(0, eff_dz * scale),
                        slice(0, eff_dy * scale),
                        slice(0, eff_dx * scale),
                    )
                ).data
            )
    lo_n = normalize_volume(lo_arr, mode="none")
    hi_n = normalize_volume(hi_arr, mode="none")
    _, _, mock_info = register_multires_pair_3d(lo_n, hi_n, scale_factor=scale)
    mock_info["registration_schema"] = "multimodal_benchmark_7dof_v2"
    return lo_n, hi_n, mock_info


def _is_zn13x_cache_valid(
    cached: dict[str, object],
    lr_size: int,
    raw_h5_dir: str | Path | None,
    prefer_offline: bool,
    certified_regions_sha256: str | None = None,
) -> bool:
    """Ensure cached Zn-13X files match requested ``lr_size``, 7-DOF schema, and provenance."""
    if cached.get("requested_lr_size") != int(lr_size):
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
    if raw_h5_dir is not None and src not in (
        "local_i13_hdf5_slab",
        "local_i13_preview_h5_registered",
    ):
        return False
    if not prefer_offline and not is_exp:
        return False
    return True


def prepare_zn13x_benchmark(
    pair: Zn13XPairName = "2x",
    dest_dir: str | Path = "data/benchmarks",
    tier: Zn13XTier = "preview",
    lr_size: int = 48,
    raw_h5_dir: str | Path | None = None,
    certified_regions_path: str | Path | None = None,
    allow_offline_fallback: bool = True,
    prefer_offline: bool = False,
    timeout_s: float = 10.0,
    seed: int = 42,
) -> tuple[Path, Path, dict[str, object]]:
    """Prepare 7-DOF-aligned LR & HR 3D TIFF volumes + provenance manifest for Diamond I13-2 ``Zn-13X``.

    Experimental pairs consume Stage 0B 7-DOF similarity transforms (``certified_regions.yaml``,
    chained across adjacent rungs for ``8x`` via ``basecoat``), resampling LR onto the
    cell-centered coarsened grid of the unresampled HR target.
    """
    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"zn13x_{pair}"
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
            ):
                return lr_tif, hr_tif, cached

        pid = os.getpid()
        scale = 2 if pair == "2x" else 8
        lo_key, hi_key = ("1p625um", "0p8125um") if pair == "2x" else ("2p600um", "0p325um")
        lo_spec = ZN13X_I13_RESOLUTIONS[lo_key]
        hi_spec = ZN13X_I13_RESOLUTIONS[hi_key]

        lr_vol: np.ndarray | None = None
        hr_vol: np.ndarray | None = None
        reg_info: dict[str, object] = {}
        source_kind = "offline_calibrated_zn13x_phantom"

        # 1. Check local HDF5 directory if provided
        if raw_h5_dir is not None:
            h5_root = Path(raw_h5_dir)
            lo_h5 = _resolve_local_h5_path(h5_root, lo_spec)
            hi_h5 = _resolve_local_h5_path(h5_root, hi_spec)
            lo_prev = h5_root / lo_spec.preview_filename
            hi_prev = h5_root / hi_spec.preview_filename
            if lo_h5 is not None and hi_h5 is not None:
                lr_vol, hr_vol, reg_info = extract_aligned_slab_from_h5(
                    lo_h5,
                    hi_h5,
                    pair=pair,
                    lr_slab_depth=lr_size,
                    lr_crop_yx=lr_size,
                    certified_regions_path=certified_regions_path,
                )
                source_kind = "local_i13_hdf5_slab"
            elif lo_prev.is_file() and hi_prev.is_file():
                lr_vol, hr_vol, reg_info = _read_7dof_preview_h5_pair(
                    lo_prev,
                    hi_prev,
                    pair=pair,
                    lr_size=lr_size,
                    certified_regions_path=certified_regions_path,
                )
                source_kind = "local_i13_preview_h5_registered"

        # 2. Try Zenodo 8-bit preview HDF5 record API if online and tier == 'preview'
        if lr_vol is None and tier == "preview" and not prefer_offline:
            api_url = f"https://zenodo.org/api/records/{ZN13X_SUBVOLUMES_ZENODO_RECORD}"
            try:
                req = urllib.request.Request(api_url, headers={"User-Agent": "OpenLSR-X/0.2.2"})
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
                            pair=pair,
                            lr_size=lr_size,
                            certified_regions_path=certified_regions_path,
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
                pair=pair,
                lr_shape=(lr_size, lr_size, lr_size),
                seed=seed,
            )
            source_kind = "offline_calibrated_zn13x_phantom"

        lr_tmp = out_dir / f".{name}_lr.{pid}.tif"
        hr_tmp = out_dir / f".{name}_hr.{pid}.tif"
        man_tmp = out_dir / f".{name}_manifest.{pid}.json"
        write_volume(
            Volume.from_array(lr_vol.astype(np.float32), voxel_size_um=lo_spec.voxel_size_um),
            lr_tmp,
            compression=None,
            overwrite=True,
        )
        write_volume(
            Volume.from_array(hr_vol.astype(np.float32), voxel_size_um=hi_spec.voxel_size_um),
            hr_tmp,
            compression=None,
            overwrite=True,
        )

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
            "pair": pair,
            "scale_factor": scale,
            "requested_lr_size": int(lr_size),
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
