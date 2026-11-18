"""Diamond Light Source I13-2 multi-resolution Zn-13X synchrotron dataset & sub-voxel 3D aligner.

Implements automated preparation, Zenodo fetching, offline calibrated phantom synthesis,
and vectorized HR-grid 3D phase-cross-correlation + tri-planar SSIM registration for the open
multi-resolution zinc-doped Zeolite 13X dataset published in:

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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

from .transforms import normalize_volume, sanitize_xct_volume

Zn13XPairName = Literal["2x", "8x"]
Zn13XTier = Literal["preview", "full"]


@dataclass(frozen=True)
class Zn13XResolutionSpec:
    """Metadata and calibrated alignment anchors for one I13-2 magnification."""

    key: str
    scan_id: str
    magnification: str
    voxel_size_um: float
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
        zenodo_record="12206815",
        recon_filename="169068_recon.h5",
        preview_filename="169068_8bit_cropped.h5",
        vmin=-5.41e-4,
        vmax=1.68e-3,
        anchor_z=934,
        roi_1040um_yx=((1102, 1502), (1006, 1406)),  # 400 x 400 px (1040 um)
        roi_812um_yx=((1146, 1458), (1050, 1362)),  # 312 x 312 px (811.2 um) -> 8x = 2496 px
    ),
    "1p625um": Zn13XResolutionSpec(
        key="1p625um",
        scan_id="169067",
        magnification="4x",
        voxel_size_um=1.625,
        zenodo_record="13327651",
        recon_filename="169067_recon.h5",
        preview_filename="169067_8bit_cropped.h5",
        vmin=-5.23e-4,
        vmax=1.11e-3,
        anchor_z=894,
        roi_1040um_yx=((1010, 1650), (858, 1498)),  # 640 x 640 px (1040 um) -> 2x = 1280 px
        roi_812um_yx=((1080, 1580), (928, 1428)),  # 500 x 500 px (812.5 um) -> 5x = 2500 px
    ),
    "0p8125um": Zn13XResolutionSpec(
        key="0p8125um",
        scan_id="169066",
        magnification="8x",
        voxel_size_um=0.8125,
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
    req = urllib.request.Request(url, headers={"User-Agent": "OpenLSR-X/0.2.0"})
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
        dz_h, dy_h, dx_h = shift_hr
        # Represent shift_hr = s * q_lr - r_hr with r_hr in [0, s - 1] so HR start is r_hr + s * z0_c
        qz, rz = divmod(dz_h, s)
        qy, ry = divmod(dy_h, s)
        qx, rx = divmod(dx_h, s)
        # Convert: lr_idx = c + q + (1 if r > 0? Wait: hr_slice starts at h_off in [0, s-1]:
        # Specifically, if we offset HR by h_off in [0, s-1], block c of hr[h_off:] covers HR [h_off + c*s, h_off + (c+1)*s),
        # which corresponds to LR index c + q_lr where s * q_lr - h_off == shift_hr => h_off = (-shift_hr) % s, q_lr = (shift_hr + h_off) // s!
        hz = (-dz_h) % s
        hy = (-dy_h) % s
        hx = (-dx_h) % s
        qz = (dz_h + hz) // s
        qy = (dy_h + hy) // s
        qx = (dx_h + hx) // s

        avail_z_c = (hr.shape[0] - hz) // s
        avail_y_c = (hr.shape[1] - hy) // s
        avail_x_c = (hr.shape[2] - hx) // s

        z0_c = max(0, -qz)
        y0_c = max(0, -qy)
        x0_c = max(0, -qx)

        z1_c = min(avail_z_c, lr.shape[0] - qz)
        y1_c = min(avail_y_c, lr.shape[1] - qy)
        x1_c = min(avail_x_c, lr.shape[2] - qx)

        min_overlap = min(8, int(min(common_lr_shape)))
        if (
            (z1_c - z0_c) < min_overlap
            or (y1_c - y0_c) < min_overlap
            or (x1_c - x0_c) < min_overlap
        ):
            return None

        lr_sub = lr[z0_c + qz : z1_c + qz, y0_c + qy : y1_c + qy, x0_c + qx : x1_c + qx]
        hr_sub = hr[
            hz + z0_c * s : hz + z1_c * s,
            hy + y0_c * s : hy + y1_c * s,
            hx + x0_c * s : hx + x1_c * s,
        ]
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


def extract_aligned_slab_from_h5(
    low_h5_path: str | Path,
    high_h5_path: str | Path,
    pair: Zn13XPairName = "2x",
    lr_slab_depth: int = 64,
    lr_crop_yx: int | None = None,
    dataset_key: str = "4-TomopyRecon-tomo/data",
    consistent_attenuation_window: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Extract and normalize an aligned 3D slab from raw >50 GB I13-2 HDF5 files without loading full cubes.

    When ``consistent_attenuation_window=True`` (default), both magnifications are normalized
    using the joint physical linear-attenuation range ``[min(vmin), max(vmax)]`` so identical
    physical ``mu`` values map to identical ``[0, 1]`` intensities across LR and HR (essential
    for linear-attenuation downsampling consistency in ``AttenuationConservationLoss``).
    """
    if pair == "2x":
        lo_spec = ZN13X_I13_RESOLUTIONS["1p625um"]
        hi_spec = ZN13X_I13_RESOLUTIONS["0p8125um"]
        scale = 2
        lo_yx = lo_spec.roi_1040um_yx
        hi_yx = hi_spec.roi_1040um_yx
    else:
        lo_spec = ZN13X_I13_RESOLUTIONS["2p600um"]
        hi_spec = ZN13X_I13_RESOLUTIONS["0p325um"]
        scale = 8
        lo_yx = lo_spec.roi_812um_yx
        hi_yx = hi_spec.roi_812um_yx

    assert lo_yx is not None and hi_yx is not None
    lo_y0, lo_y1 = lo_yx[0]
    lo_x0, lo_x1 = lo_yx[1]
    hi_y0, hi_y1 = hi_yx[0]
    hi_x0, hi_x1 = hi_yx[1]

    if lr_crop_yx is not None:
        lo_y1 = min(lo_y1, lo_y0 + lr_crop_yx)
        lo_x1 = min(lo_x1, lo_x0 + lr_crop_yx)
        hi_y1 = min(hi_y1, hi_y0 + lr_crop_yx * scale)
        hi_x1 = min(hi_x1, hi_x0 + lr_crop_yx * scale)

    # Raw Savu TomopyRecon HDF5 stores axes (x_orig, z_orig, y_orig) -> transposed (1, 2, 0) -> (Z, Y, X)
    with h5py.File(low_h5_path, "r") as f_lo, h5py.File(high_h5_path, "r") as f_hi:
        dset_lo = f_lo[dataset_key]
        dset_hi = f_hi[dataset_key]
        max_lo_z = int(dset_lo.shape[1])
        max_hi_z = int(dset_hi.shape[1])

        eff_depth = min(int(lr_slab_depth), max_lo_z, max_hi_z // scale)
        if eff_depth < 1:
            raise ValueError(
                f"HDF5 z-dimensions ({max_lo_z}, {max_hi_z}) cannot satisfy scale={scale}."
            )
        half_d = eff_depth // 2

        lo_z0 = min(max(0, lo_spec.anchor_z - half_d), max_lo_z - eff_depth)
        lo_z1 = lo_z0 + eff_depth
        hi_z0 = min(max(0, hi_spec.anchor_z - half_d * scale), max_hi_z - eff_depth * scale)
        hi_z1 = hi_z0 + eff_depth * scale

        raw_lo = sanitize_xct_volume(np.asarray(dset_lo[lo_x0:lo_x1, lo_z0:lo_z1, lo_y0:lo_y1]))
        slab_lo = np.transpose(raw_lo, (1, 2, 0))

        raw_hi = sanitize_xct_volume(np.asarray(dset_hi[hi_x0:hi_x1, hi_z0:hi_z1, hi_y0:hi_y1]))
        slab_hi = np.transpose(raw_hi, (1, 2, 0))

    if consistent_attenuation_window:
        joint_vmin = min(lo_spec.vmin, hi_spec.vmin)
        joint_vmax = max(lo_spec.vmax, hi_spec.vmax)
        norm_lo = normalize_volume(slab_lo, mode="fixed", vmin=joint_vmin, vmax=joint_vmax)
        norm_hi = normalize_volume(slab_hi, mode="fixed", vmin=joint_vmin, vmax=joint_vmax)
    else:
        norm_lo = normalize_volume(slab_lo, mode="fixed", vmin=lo_spec.vmin, vmax=lo_spec.vmax)
        norm_hi = normalize_volume(slab_hi, mode="fixed", vmin=hi_spec.vmin, vmax=hi_spec.vmax)

    return register_multires_pair_3d(norm_lo, norm_hi, scale_factor=scale)


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


def _read_centered_preview_h5_pair(
    lo_h5_path: Path,
    hi_h5_path: Path,
    lr_size: int,
    scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read coaxial center-cropped 8-bit subvolumes from Zenodo 14245751 HDF5 files."""
    with h5py.File(lo_h5_path, "r") as f_lo, h5py.File(hi_h5_path, "r") as f_hi:
        lo_keys = [k for k in f_lo.keys() if isinstance(f_lo[k], h5py.Dataset)]
        hi_keys = [k for k in f_hi.keys() if isinstance(f_hi[k], h5py.Dataset)]
        if not lo_keys or not hi_keys:
            raise KeyError("Zenodo preview HDF5 file contained no top-level dataset.")
        d_lo = f_lo[lo_keys[0]]
        d_hi = f_hi[hi_keys[0]]

        eff_dz = min(int(lr_size), int(d_lo.shape[0]), int(d_hi.shape[0]) // scale)
        eff_dy = min(int(lr_size), int(d_lo.shape[1]), int(d_hi.shape[1]) // scale)
        eff_dx = min(int(lr_size), int(d_lo.shape[2]), int(d_hi.shape[2]) // scale)

        lo_z0 = max(0, (int(d_lo.shape[0]) - eff_dz) // 2)
        lo_y0 = max(0, (int(d_lo.shape[1]) - eff_dy) // 2)
        lo_x0 = max(0, (int(d_lo.shape[2]) - eff_dx) // 2)

        hi_dz, hi_dy, hi_dx = eff_dz * scale, eff_dy * scale, eff_dx * scale
        hi_z0 = max(0, (int(d_hi.shape[0]) - hi_dz) // 2)
        hi_y0 = max(0, (int(d_hi.shape[1]) - hi_dy) // 2)
        hi_x0 = max(0, (int(d_hi.shape[2]) - hi_dx) // 2)

        lo_arr = sanitize_xct_volume(
            np.asarray(d_lo[lo_z0 : lo_z0 + eff_dz, lo_y0 : lo_y0 + eff_dy, lo_x0 : lo_x0 + eff_dx])
        )
        hi_arr = sanitize_xct_volume(
            np.asarray(d_hi[hi_z0 : hi_z0 + hi_dz, hi_y0 : hi_y0 + hi_dy, hi_x0 : hi_x0 + hi_dx])
        )
    return normalize_volume(lo_arr, mode="none"), normalize_volume(hi_arr, mode="none")


def _is_zn13x_cache_valid(
    cached: dict[str, object],
    lr_size: int,
    raw_h5_dir: str | Path | None,
    prefer_offline: bool,
    allow_offline_fallback: bool,
) -> bool:
    """Ensure cached Zn-13X files match requested ``lr_size`` and never mask real data with stale phantoms."""
    if cached.get("requested_lr_size") != int(lr_size):
        return False
    src = str(cached.get("source_kind", ""))
    is_exp = bool(cached.get("is_experimental_xct", False))
    if raw_h5_dir is not None and src not in (
        "local_i13_hdf5_slab",
        "local_i13_preview_h5_registered",
    ):
        return False
    if not prefer_offline and not is_exp:
        # Cached artifact is a synthetic phantom, whereas caller requested real experimental data!
        if not allow_offline_fallback or raw_h5_dir is not None:
            return False
        return False
    return True


def prepare_zn13x_benchmark(
    pair: Zn13XPairName = "2x",
    dest_dir: str | Path = "data/benchmarks",
    tier: Zn13XTier = "preview",
    lr_size: int = 48,
    raw_h5_dir: str | Path | None = None,
    allow_offline_fallback: bool = True,
    prefer_offline: bool = False,
    timeout_s: float = 10.0,
    seed: int = 42,
) -> tuple[Path, Path, dict[str, object]]:
    """Prepare aligned LR & HR 3D TIFF volumes + provenance manifest for Diamond I13-2 ``Zn-13X``.

    Resolution priority:
    1. Cached ``zn13x_<pair>_lr.tif`` + ``zn13x_<pair>_hr.tif`` + manifest if size and source provenance match.
    2. Local synchrotron HDF5 reconstructions in ``raw_h5_dir`` (supports ``<scan_id>_recon.h5``,
       Zenodo ``<scan_id>/tomo_p4_tomopy_recon.h5``, and ``16906*_8bit_cropped.h5`` layouts).
    3. Public Zenodo 8-bit preview HDF5 sub-volumes (``10.5281/zenodo.14245751``,
       ``16906*_8bit_cropped.h5``) when online and ``prefer_offline=False``.
    4. Calibrated tri-phase ``Zn-13X`` synchrotron phantom when offline or ``prefer_offline=True``
       (raises ``RuntimeError`` when ``allow_offline_fallback=False``).
    """
    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"zn13x_{pair}"
    lr_tif = out_dir / f"{name}_lr.tif"
    hr_tif = out_dir / f"{name}_hr.tif"
    manifest_path = out_dir / f"{name}_manifest.json"
    lock_path = out_dir / f".{name}.lock"

    with lock_path.open("w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        if lr_tif.is_file() and hr_tif.is_file() and manifest_path.is_file():
            cached: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
            if _is_zn13x_cache_valid(
                cached,
                lr_size=lr_size,
                raw_h5_dir=raw_h5_dir,
                prefer_offline=prefer_offline,
                allow_offline_fallback=allow_offline_fallback,
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
                )
                source_kind = "local_i13_hdf5_slab"
            elif lo_prev.is_file() and hi_prev.is_file():
                raw_lo, raw_hi = _read_centered_preview_h5_pair(
                    lo_prev, hi_prev, lr_size=lr_size, scale=scale
                )
                lr_vol, hr_vol, reg_info = register_multires_pair_3d(
                    raw_lo, raw_hi, scale_factor=scale
                )
                source_kind = "local_i13_preview_h5_registered"

        # 2. Try Zenodo 8-bit preview HDF5 record API if online and tier == 'preview'
        if lr_vol is None and tier == "preview" and not prefer_offline:
            api_url = f"https://zenodo.org/api/records/{ZN13X_SUBVOLUMES_ZENODO_RECORD}"
            try:
                req = urllib.request.Request(api_url, headers={"User-Agent": "OpenLSR-X/0.2.0"})
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
                        _download_https_file(str(lo_candidates[0]), tmp_lo, timeout_s=60.0)
                        _download_https_file(str(hi_candidates[0]), tmp_hi, timeout_s=60.0)
                        raw_lo_arr, raw_hi_arr = _read_centered_preview_h5_pair(
                            tmp_lo, tmp_hi, lr_size=lr_size, scale=scale
                        )
                        lr_vol, hr_vol, reg_info = register_multires_pair_3d(
                            raw_lo_arr,
                            raw_hi_arr,
                            scale_factor=scale,
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
        tifffile.imwrite(lr_tmp, lr_vol.astype(np.float32), photometric="minisblack")
        tifffile.imwrite(hr_tmp, hr_vol.astype(np.float32), photometric="minisblack")

        is_exp = source_kind in {
            "local_i13_hdf5_slab",
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
            "low_zenodo_doi": f"10.5281/zenodo.{lo_spec.zenodo_record}" if is_exp else None,
            "high_zenodo_doi": f"10.5281/zenodo.{hi_spec.zenodo_record}" if is_exp else None,
            "zenodo_preview_doi": (
                f"10.5281/zenodo.{ZN13X_SUBVOLUMES_ZENODO_RECORD}"
                if source_kind in {"zenodo_14245751_preview", "local_i13_preview_h5_registered"}
                else None
            ),
            "calibrated_reference_paper_doi": "10.1038/s41597-025-04605-9",
            "source_kind": source_kind,
            "is_experimental_xct": is_exp,
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
