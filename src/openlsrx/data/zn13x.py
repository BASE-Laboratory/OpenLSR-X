"""Diamond Light Source I13-2 multi-resolution Zn-13X synchrotron dataset & 3D aligner.

Implements automated preparation, Zenodo fetching, offline calibrated phantom synthesis,
and vectorized 3D phase-cross-correlation + SSIM registration for the open multi-resolution
zinc-doped Zeolite 13X dataset published in:

    Green, C., Ahmed, S., Marathe, S. et al., "Three-dimensional, multimodal synchrotron
    data for machine learning applications", Scientific Data 12, 329 (2025),
    https://doi.org/10.1038/s41597-025-04605-9
"""

from __future__ import annotations

import hashlib
import json
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

from .transforms import normalize_volume

Zn13XPairName = Literal["2x", "8x"]
Zn13XTier = Literal["preview", "full"]


@dataclass(frozen=True)
class Zn13XResolutionSpec:
    """Metadata and calibrated alignment anchors for one I13-2 magnification."""

    key: str
    magnification: str
    voxel_size_um: float
    zenodo_record: str
    recon_filename: str
    vmin: float
    vmax: float
    anchor_z: int
    roi_1040um_yx: tuple[tuple[int, int], tuple[int, int]] | None
    roi_812um_yx: tuple[tuple[int, int], tuple[int, int]]


ZN13X_SUBVOLUMES_ZENODO_RECORD = "14245751"

ZN13X_I13_RESOLUTIONS: dict[str, Zn13XResolutionSpec] = {
    "2p600um": Zn13XResolutionSpec(
        key="2p600um",
        magnification="2.5x",
        voxel_size_um=2.600,
        zenodo_record="12206815",
        recon_filename="169068_recon.h5",
        vmin=-5.41e-4,
        vmax=1.68e-3,
        anchor_z=934,
        roi_1040um_yx=((1102, 1502), (1006, 1406)),  # 400 x 400 px (1040 um)
        roi_812um_yx=((1146, 1458), (1050, 1362)),  # 312 x 312 px (811.2 um) -> 8x = 2496 px
    ),
    "1p625um": Zn13XResolutionSpec(
        key="1p625um",
        magnification="4x",
        voxel_size_um=1.625,
        zenodo_record="13327651",
        recon_filename="169067_recon.h5",
        vmin=-5.23e-4,
        vmax=1.11e-3,
        anchor_z=894,
        roi_1040um_yx=((1010, 1650), (858, 1498)),  # 640 x 640 px (1040 um) -> 2x = 1280 px
        roi_812um_yx=((1080, 1580), (928, 1428)),  # 500 x 500 px (812.5 um) -> 5x = 2500 px
    ),
    "0p8125um": Zn13XResolutionSpec(
        key="0p8125um",
        magnification="8x",
        voxel_size_um=0.8125,
        zenodo_record="13327682",
        recon_filename="169066_recon.h5",
        vmin=-2.50e-4,
        vmax=6.79e-4,
        anchor_z=804,
        roi_1040um_yx=((763, 2043), (466, 1746)),  # 1280 x 1280 px (1040 um)
        roi_812um_yx=((753, 1753), (753, 1753)),  # 1000 x 1000 px (812.5 um) -> 2.5x = 2500 px
    ),
    "0p325um": Zn13XResolutionSpec(
        key="0p325um",
        magnification="20x",
        voxel_size_um=0.325,
        zenodo_record="13327692",
        recon_filename="169065_recon.h5",
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


def register_multires_pair_3d(
    low_vol: np.ndarray,
    high_vol: np.ndarray,
    scale_factor: int = 2,
    local_refine_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Vectorized 3D registration of paired multi-magnification XCT volumes.

    Replaces brute-force 3D slice scanning with FFT phase cross-correlation at coarse
    resolution followed by local 3D SSIM refinement, returning strictly shape-matched
    ``(aligned_lr, aligned_hr, metadata)`` with ``aligned_hr.shape == s * aligned_lr.shape``.
    """
    if scale_factor < 1:
        raise ValueError(f"scale_factor must be >= 1, got {scale_factor}.")

    lr = np.asarray(low_vol, dtype=np.float32)
    hr = np.asarray(high_vol, dtype=np.float32)
    if lr.ndim != 3 or hr.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got {lr.shape} and {hr.shape}.")

    hr_coarse = _block_downsample_3d(hr, scale_factor)
    common_lr_shape = tuple(min(a, b) for a, b in zip(lr.shape, hr_coarse.shape, strict=True))
    lr_ref = lr[: common_lr_shape[0], : common_lr_shape[1], : common_lr_shape[2]]
    hr_ref = hr_coarse[: common_lr_shape[0], : common_lr_shape[1], : common_lr_shape[2]]

    shift_est, _, _ = phase_cross_correlation(lr_ref, hr_ref, upsample_factor=1)
    base_shift = tuple(int(round(float(s))) for s in shift_est)

    def _extract_overlap(
        shift_lr: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        dz, dy, dx = shift_lr
        # Map coarse HR coordinate c -> LR coordinate c + shift_lr
        z0_hr = max(0, -dz)
        y0_hr = max(0, -dy)
        x0_hr = max(0, -dx)

        z1_hr = min(hr_coarse.shape[0], lr.shape[0] - dz)
        y1_hr = min(hr_coarse.shape[1], lr.shape[1] - dy)
        x1_hr = min(hr_coarse.shape[2], lr.shape[2] - dx)

        if (z1_hr - z0_hr) < 8 or (y1_hr - y0_hr) < 8 or (x1_hr - x0_hr) < 8:
            return None

        lr_sub = lr[z0_hr + dz : z1_hr + dz, y0_hr + dy : y1_hr + dy, x0_hr + dx : x1_hr + dx]
        hr_c_sub = hr_coarse[z0_hr:z1_hr, y0_hr:y1_hr, x0_hr:x1_hr]
        s = scale_factor
        hr_sub = hr[z0_hr * s : z1_hr * s, y0_hr * s : y1_hr * s, x0_hr * s : x1_hr * s]
        return lr_sub, hr_c_sub, hr_sub

    best_score = -2.0
    best_shift = (0, 0, 0)
    best_pair: tuple[np.ndarray, np.ndarray] | None = None

    candidates = [
        (base_shift[0] + dz, base_shift[1] + dy, base_shift[2] + dx)
        for dz in range(-local_refine_radius, local_refine_radius + 1)
        for dy in range(-local_refine_radius, local_refine_radius + 1)
        for dx in range(-local_refine_radius, local_refine_radius + 1)
    ]
    if (0, 0, 0) not in candidates:
        candidates.append((0, 0, 0))

    for cand in candidates:
        extracted = _extract_overlap(cand)
        if extracted is None:
            continue
        lr_sub, hr_c_sub, hr_sub = extracted
        mid = lr_sub.shape[0] // 2
        win = min(7, min(lr_sub.shape[1], lr_sub.shape[2]))
        if win % 2 == 0:
            win -= 1
        score = (
            float(
                structural_similarity(
                    lr_sub[mid],
                    hr_c_sub[mid],
                    data_range=1.0,
                    win_size=max(3, win),
                )
            )
            if win >= 3
            else -float(np.mean((lr_sub - hr_c_sub) ** 2))
        )
        if score > best_score:
            best_score = score
            best_shift = cand
            best_pair = (lr_sub.copy(), hr_sub.copy())

    if best_pair is None:
        d, h, w = common_lr_shape
        s = scale_factor
        best_pair = (lr[:d, :h, :w].copy(), hr[: d * s, : h * s, : w * s].copy())
        best_shift = (0, 0, 0)

    info: dict[str, object] = {
        "coarse_shift_zyx": list(best_shift),
        "coarse_mid_slice_ssim": best_score,
        "aligned_lr_shape": list(best_pair[0].shape),
        "aligned_hr_shape": list(best_pair[1].shape),
        "scale_factor": scale_factor,
    }
    return best_pair[0], best_pair[1], info


def synthesize_zn13x_phantom_pair(
    pair: Zn13XPairName = "2x",
    lr_shape: tuple[int, int, int] = (48, 48, 48),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a calibrated two-phase Zn-doped Zeolite 13X synchrotron XCT pair.

    Models the physical structure described in Green et al. (*Sci. Data* 2025):
    - Cleaved Zeolite 13X fragment inside a thin-walled capillary FOV
    - Higher-attenuation outer Zn-exchanged rim (~0.88) and Na-13X core (~0.68)
    - Hierarchical connected macro- and meso-pore network (~0.14 attenuation)
    - Distinct objective-specific scintillator PSF and detector noise at LR vs HR
    """
    scale = 2 if pair == "2x" else 8
    hr_shape = (lr_shape[0] * scale, lr_shape[1] * scale, lr_shape[2] * scale)
    rng = np.random.default_rng(seed)

    # Coordinate grid in [-1, 1]^3 for cleaved pellet fragment + Zn diffusion rim
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, hr_shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, hr_shape[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, hr_shape[2], dtype=np.float32),
        indexing="ij",
    )
    radial = np.sqrt(0.85 * zz**2 + yy**2 + 1.1 * xx**2)

    # Hierarchical macropores + mesopores generated via multi-scale Gaussian random fields
    coarse_field = gaussian_filter(
        rng.standard_normal(hr_shape).astype(np.float32), sigma=max(2.0, scale * 1.4)
    )
    fine_field = gaussian_filter(
        rng.standard_normal(hr_shape).astype(np.float32), sigma=max(0.9, scale * 0.45)
    )
    macro_pores = coarse_field > np.quantile(coarse_field, 0.78)
    meso_pores = fine_field > np.quantile(fine_field, 0.84)
    pore_mask = np.logical_or(macro_pores, meso_pores)

    # Zn2+ ion-exchange rim (outer shell of pellet has higher X-ray attenuation at 27 keV)
    zn_rim = radial > 0.52
    hr_clean = np.where(zn_rim, 0.88, 0.68).astype(np.float32)
    hr_clean[pore_mask] = 0.14

    # High-magnification GGG:Eu / LuAG scintillator PSF + mild FBP noise
    hr_vol = gaussian_filter(hr_clean, sigma=0.55, mode="reflect")
    hr_vol = hr_vol + rng.normal(0.0, 0.015, size=hr_shape).astype(np.float32)
    hr_vol = np.clip(hr_vol, 0.0, 1.0).astype(np.float32, copy=False)

    # Lower-magnification thick LuAG (500 um) scintillator blur + lens browning noise
    lr_blurred = gaussian_filter(hr_clean, sigma=0.85 * scale, mode="reflect")
    lr_binned = _block_downsample_3d(lr_blurred, scale)
    noise_sigma = 0.024 if pair == "2x" else 0.035  # 2.5x lens browning increases photon noise
    lr_vol = lr_binned + rng.normal(0.0, noise_sigma, size=lr_binned.shape).astype(np.float32)
    lr_vol = np.clip(lr_vol, 0.0, 1.0).astype(np.float32, copy=False)
    return lr_vol, hr_vol


def extract_aligned_slab_from_h5(
    low_h5_path: str | Path,
    high_h5_path: str | Path,
    pair: Zn13XPairName = "2x",
    lr_slab_depth: int = 64,
    lr_crop_yx: int | None = None,
    dataset_key: str = "4-TomopyRecon-tomo/data",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Extract and normalize an aligned 3D slab from raw >50 GB I13-2 HDF5 files without loading full cubes."""
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
    half_d = lr_slab_depth // 2
    lo_z0 = max(0, lo_spec.anchor_z - half_d)
    lo_z1 = lo_z0 + lr_slab_depth
    hi_z0 = max(0, hi_spec.anchor_z - half_d * scale)
    hi_z1 = hi_z0 + lr_slab_depth * scale

    lo_y0, lo_y1 = lo_yx[0]
    lo_x0, lo_x1 = lo_yx[1]
    hi_y0, hi_y1 = hi_yx[0]
    hi_x0, hi_x1 = hi_yx[1]

    if lr_crop_yx is not None:
        lo_y1 = min(lo_y1, lo_y0 + lr_crop_yx)
        lo_x1 = min(lo_x1, lo_x0 + lr_crop_yx)
        hi_y1 = min(hi_y1, hi_y0 + lr_crop_yx * scale)
        hi_x1 = min(hi_x1, hi_x0 + lr_crop_yx * scale)

    # Note: Raw Savu TomopyRecon HDF5 stores axes (y_orig, z_orig, x_orig) -> transposed (1, 2, 0) in Calum's notebook
    with h5py.File(low_h5_path, "r") as f_lo:
        dset_lo = f_lo[dataset_key]
        raw_lo = np.asarray(dset_lo[lo_x0:lo_x1, lo_z0:lo_z1, lo_y0:lo_y1], dtype=np.float32)
        slab_lo = np.transpose(raw_lo, (1, 2, 0))

    with h5py.File(high_h5_path, "r") as f_hi:
        dset_hi = f_hi[dataset_key]
        raw_hi = np.asarray(dset_hi[hi_x0:hi_x1, hi_z0:hi_z1, hi_y0:hi_y1], dtype=np.float32)
        slab_hi = np.transpose(raw_hi, (1, 2, 0))

    norm_lo = normalize_volume(slab_lo, mode="fixed", vmin=lo_spec.vmin, vmax=lo_spec.vmax)
    norm_hi = normalize_volume(slab_hi, mode="fixed", vmin=hi_spec.vmin, vmax=hi_spec.vmax)
    return register_multires_pair_3d(norm_lo, norm_hi, scale_factor=scale)


def prepare_zn13x_benchmark(
    pair: Zn13XPairName = "2x",
    dest_dir: str | Path = "data/benchmarks",
    tier: Zn13XTier = "preview",
    lr_size: int = 48,
    raw_h5_dir: str | Path | None = None,
    allow_offline_fallback: bool = True,
    seed: int = 42,
) -> tuple[Path, Path, dict[str, object]]:
    """Prepare aligned LR & HR 3D TIFF volumes + provenance manifest for Diamond I13-2 ``Zn-13X``.

    Resolution priority:
    1. Cached ``zn13x_<pair>_lr.tif`` + ``zn13x_<pair>_hr.tif`` + manifest if already present.
    2. Local synchrotron HDF5 reconstructions in ``raw_h5_dir`` (e.g., on Diamond/Isambard Lustre).
    3. Public Zenodo 8-bit preview sub-volumes (``10.5281/zenodo.14245751``) when online.
    4. Calibrated two-phase ``Zn-13X`` synchrotron phantom when offline or air-gapped.
    """
    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"zn13x_{pair}"
    lr_tif = out_dir / f"{name}_lr.tif"
    hr_tif = out_dir / f"{name}_hr.tif"
    manifest_path = out_dir / f"{name}_manifest.json"

    if lr_tif.is_file() and hr_tif.is_file() and manifest_path.is_file():
        cached: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return lr_tif, hr_tif, cached

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
        lo_h5 = h5_root / lo_spec.recon_filename
        hi_h5 = h5_root / hi_spec.recon_filename
        if lo_h5.is_file() and hi_h5.is_file():
            lr_vol, hr_vol, reg_info = extract_aligned_slab_from_h5(
                lo_h5,
                hi_h5,
                pair=pair,
                lr_slab_depth=lr_size,
                lr_crop_yx=lr_size,
            )
            source_kind = "local_i13_hdf5_slab"

    # 2. Try Zenodo 8-bit preview record API if online and tier == 'preview'
    if lr_vol is None and tier == "preview":
        api_url = f"https://zenodo.org/api/records/{ZN13X_SUBVOLUMES_ZENODO_RECORD}"
        try:
            with urllib.request.urlopen(api_url, timeout=8) as resp:  # noqa: S310
                record_meta = json.loads(resp.read().decode("utf-8"))
            files = {f["key"]: f["links"]["self"] for f in record_meta.get("files", [])}
            # Match TIF/H5 files mentioning pixel sizes if present in record
            lo_candidates = [
                u for k, u in files.items() if lo_spec.magnification in k or lo_key in k
            ]
            hi_candidates = [
                u for k, u in files.items() if hi_spec.magnification in k or hi_key in k
            ]
            if lo_candidates and hi_candidates:
                tmp_lo = out_dir / f".{name}_lo.tmp"
                tmp_hi = out_dir / f".{name}_hi.tmp"
                try:
                    urllib.request.urlretrieve(lo_candidates[0], tmp_lo)  # noqa: S310
                    urllib.request.urlretrieve(hi_candidates[0], tmp_hi)  # noqa: S310
                    raw_lo_arr = normalize_volume(tifffile.imread(tmp_lo), mode="minmax")
                    raw_hi_arr = normalize_volume(tifffile.imread(tmp_hi), mode="minmax")
                    lr_vol, hr_vol, reg_info = register_multires_pair_3d(
                        raw_lo_arr[:lr_size, :lr_size, :lr_size],
                        raw_hi_arr[: lr_size * scale, : lr_size * scale, : lr_size * scale],
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

    # 3. Offline calibrated Zn-13X two-phase synchrotron phantom
    if lr_vol is None or hr_vol is None:
        if not allow_offline_fallback:
            raise RuntimeError(f"No local HDF5 or online Zenodo files available for {name}.")
        lr_vol, hr_vol = synthesize_zn13x_phantom_pair(
            pair=pair,
            lr_shape=(lr_size, lr_size, lr_size),
            seed=seed,
        )
        source_kind = "offline_calibrated_zn13x_phantom"

    tifffile.imwrite(lr_tif, lr_vol.astype(np.float32), photometric="minisblack")
    tifffile.imwrite(hr_tif, hr_vol.astype(np.float32), photometric="minisblack")

    manifest: dict[str, object] = {
        "name": name,
        "pair": pair,
        "scale_factor": scale,
        "low_resolution_um": lo_spec.voxel_size_um,
        "high_resolution_um": hi_spec.voxel_size_um,
        "low_zenodo_doi": f"10.5281/zenodo.{lo_spec.zenodo_record}",
        "high_zenodo_doi": f"10.5281/zenodo.{hi_spec.zenodo_record}",
        "paper_doi": "10.1038/s41597-025-04605-9",
        "source_kind": source_kind,
        "lr_shape": list(lr_vol.shape),
        "hr_shape": list(hr_vol.shape),
        "lr_sha256": _sha256_file(lr_tif),
        "hr_sha256": _sha256_file(hr_tif),
        "registration": reg_info,
        "seed": seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return lr_tif, hr_tif, manifest
