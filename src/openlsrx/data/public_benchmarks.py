"""Open-access micro-CT rock benchmark fetcher with cryptographic checksum verification and HPC-safe locking."""

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

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .synthetic import _porespy_blobs
from .transforms import degrade_xct_volume, normalize_volume
from .zn13x import apply_hr_grid_shift_pair_3d, register_multires_pair_3d


@dataclass(frozen=True)
class PublicRockSpec:
    """Metadata for an open-access 3D micro-CT porous-rock benchmark."""

    name: str
    description: str
    lithology: str
    reference: str
    license_name: str
    url: str | None
    sha256: str | None
    md5: str | None
    file_format: str
    raw_dtype: str
    raw_shape: tuple[int, int, int]
    target_porosity: float
    macro_blobiness: float
    micro_blobiness: float | None


@dataclass(frozen=True)
class PublicDualScanSpec:
    """Metadata for an open-access physical multi-resolution (LR + HR) micro-CT rock scan pair."""

    name: str
    description: str
    lithology: str
    reference: str
    zenodo_doi: str
    license_name: str
    scale_factor: int
    low_voxel_size_um: float
    high_voxel_size_um: float
    low_url: str
    low_filename: str
    low_md5: str
    low_raw_dtype: str
    low_raw_shape: tuple[int, int, int]
    high_url: str
    high_filename: str
    high_md5: str
    high_raw_dtype: str
    high_raw_shape: tuple[int, int, int]
    target_porosity: float
    macro_blobiness: float


AVAILABLE_PUBLIC_DATASETS: dict[str, PublicRockSpec] = {
    "bentheimer_sandstone": PublicRockSpec(
        name="bentheimer_sandstone",
        description="Bentheimer Sandstone experimental micro-CT subvolume (Core 1 Subvol 1, 6 um voxel size, ~22% porosity).",
        lithology="Sandstone",
        reference="Jackson, S. J., Niu, Y., Manoorkar, S., Mostaghimi, P. & Armstrong, R. T., Multi-resolution Bentheimer Sandstone micro-CT dataset, Zenodo (2021), https://doi.org/10.5281/zenodo.5542624",
        license_name="CC-BY-4.0",
        url="https://zenodo.org/records/5542624/files/Core1_Subvol1_LR.tif?download=1",
        sha256=None,
        md5="e6a30d40ef987571fab6b8393cef272a",
        file_format="tif",
        raw_dtype="uint8",
        raw_shape=(225, 225, 225),
        target_porosity=0.22,
        macro_blobiness=1.1,
        micro_blobiness=None,
    ),
    "bentheimer_laminated": PublicRockSpec(
        name="bentheimer_laminated",
        description="Laminated Bentheimer Sandstone experimental micro-CT subvolume (Core 2 Subvol 2, 6 um voxel size).",
        lithology="Sandstone",
        reference="Jackson, S. J., Niu, Y., Manoorkar, S., Mostaghimi, P. & Armstrong, R. T., Multi-resolution Bentheimer Sandstone micro-CT dataset, Zenodo (2021), https://doi.org/10.5281/zenodo.5542624",
        license_name="CC-BY-4.0",
        url="https://zenodo.org/records/5542624/files/Core2_Subvol2_LR.tif?download=1",
        sha256=None,
        md5="762404de42a61484134bb27c5d910f33",
        file_format="tif",
        raw_dtype="uint8",
        raw_shape=(225, 225, 225),
        target_porosity=0.20,
        macro_blobiness=1.25,
        micro_blobiness=None,
    ),
    "indiana_limestone": PublicRockSpec(
        name="indiana_limestone",
        description="Calibrated bimodal vuggy + microporous carbonate XCT phantom (~18% total porosity).",
        lithology="Carbonate",
        reference="Calibrated bimodal carbonate phantom parameterized after Blunt et al., Adv. Water Resour. 51, 197-216 (2013)",
        license_name="CC-BY-4.0",
        url=None,
        sha256=None,
        md5=None,
        file_format="phantom",
        raw_dtype="float32",
        raw_shape=(128, 128, 128),
        target_porosity=0.18,
        macro_blobiness=0.65,
        micro_blobiness=2.6,
    ),
}

AVAILABLE_DUALSCAN_DATASETS: dict[str, PublicDualScanSpec] = {
    "bentheimer_dualscan_core1": PublicDualScanSpec(
        name="bentheimer_dualscan_core1",
        description=(
            "Bentheimer Sandstone Core 1 Subvol 1 physical dual-scan micro-CT pair "
            "(18 um 75^3 LR -> 6 um 225^3 HR, 3x super-resolution)."
        ),
        lithology="Sandstone",
        reference=(
            "Jackson, S. J., Niu, Y., Manoorkar, S., Mostaghimi, P. & Armstrong, R. T., "
            "Multi-resolution Bentheimer Sandstone micro-CT dataset, Zenodo (2021), "
            "https://doi.org/10.5281/zenodo.5542624"
        ),
        zenodo_doi="10.5281/zenodo.5542624",
        license_name="CC-BY-4.0",
        scale_factor=3,
        low_voxel_size_um=18.0,
        high_voxel_size_um=6.0,
        low_url=(
            "https://zenodo.org/records/5542624/files/"
            "Core1_Subvol1_18micron_75cube_16bit_LE_normalised.raw?download=1"
        ),
        low_filename="Core1_Subvol1_18micron_75cube_16bit_LE_normalised.raw",
        low_md5="6be38996684013a6a82ca181738bee48",
        low_raw_dtype="<u2",
        low_raw_shape=(75, 75, 75),
        high_url=(
            "https://zenodo.org/records/5542624/files/"
            "Core1_Subvol1_6micron_225cube_16bit_LE_normalised.raw?download=1"
        ),
        high_filename="Core1_Subvol1_6micron_225cube_16bit_LE_normalised.raw",
        high_md5="d26cb64dbe448c1330bdc00e424c011b",
        high_raw_dtype="<u2",
        high_raw_shape=(225, 225, 225),
        target_porosity=0.22,
        macro_blobiness=1.1,
    ),
    "bentheimer_dualscan_core2": PublicDualScanSpec(
        name="bentheimer_dualscan_core2",
        description=(
            "Laminated Bentheimer Sandstone Core 2 Subvol 2 physical dual-scan micro-CT pair "
            "(18 um 75^3 LR -> 6 um 225^3 HR, 3x super-resolution)."
        ),
        lithology="Sandstone",
        reference=(
            "Jackson, S. J., Niu, Y., Manoorkar, S., Mostaghimi, P. & Armstrong, R. T., "
            "Multi-resolution Bentheimer Sandstone micro-CT dataset, Zenodo (2021), "
            "https://doi.org/10.5281/zenodo.5542624"
        ),
        zenodo_doi="10.5281/zenodo.5542624",
        license_name="CC-BY-4.0",
        scale_factor=3,
        low_voxel_size_um=18.0,
        high_voxel_size_um=6.0,
        low_url=(
            "https://zenodo.org/records/5542624/files/"
            "Core2_Subvol2_18micron_75cube_16bit_LE_normalised.raw?download=1"
        ),
        low_filename="Core2_Subvol2_18micron_75cube_16bit_LE_normalised.raw",
        low_md5="2f2647f06ade06f21c5d11aaaa068e29",
        low_raw_dtype="<u2",
        low_raw_shape=(75, 75, 75),
        high_url=(
            "https://zenodo.org/records/5542624/files/"
            "Core2_Subvol2_6micron_225cube_16bit_LE_normalised.raw?download=1"
        ),
        high_filename="Core2_Subvol2_6micron_225cube_16bit_LE_normalised.raw",
        high_md5="3ee93a943a000b7635865780ac5c80e6",
        high_raw_dtype="<u2",
        high_raw_shape=(225, 225, 225),
        target_porosity=0.20,
        macro_blobiness=1.25,
    ),
}


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_https_file(url: str, dest: Path, timeout_s: float = 30.0) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"Refusing non-HTTPS benchmark download URL: {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": "OpenLSR-X/0.2.1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp, dest.open("wb") as out_f:  # noqa: S310
        shutil.copyfileobj(resp, out_f)


def _center_crop_3d(arr: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    d, h, w = (min(int(arr.shape[i]), int(shape[i])) for i in range(3))
    z0 = max(0, (arr.shape[0] - d) // 2)
    y0 = max(0, (arr.shape[1] - h) // 2)
    x0 = max(0, (arr.shape[2] - w) // 2)
    return np.ascontiguousarray(arr[z0 : z0 + d, y0 : y0 + h, x0 : x0 + w], dtype=np.float32)


def _synthesize_calibrated_rock_phantom(
    spec: PublicRockSpec,
    shape: tuple[int, int, int],
    seed: int = 42,
) -> np.ndarray:
    """Synthesize a calibrated greyscale 3D XCT digital-rock phantom when offline/air-gapped."""
    rng = np.random.default_rng(seed)

    if spec.micro_blobiness is None:
        pore_mask = _porespy_blobs(
            shape=shape,
            blobiness=spec.macro_blobiness,
            porosity=spec.target_porosity,
            seed=seed,
        )
        grain_dt = distance_transform_edt(~pore_mask).astype(np.float32)
        grain_dt = grain_dt / max(float(grain_dt.max()), 1.0)
        attenuation = np.where(
            pore_mask,
            0.14,
            0.78 + 0.12 * grain_dt,
        ).astype(np.float32)
    else:
        macro_pores = _porespy_blobs(
            shape=shape,
            blobiness=spec.macro_blobiness,
            porosity=spec.target_porosity * 0.55,
            seed=seed,
        )
        micro_pores = _porespy_blobs(
            shape=shape,
            blobiness=spec.micro_blobiness,
            porosity=spec.target_porosity * 0.65,
            seed=seed + 77,
        )
        microporous_zone = _porespy_blobs(
            shape=shape,
            blobiness=0.9,
            porosity=0.45,
            seed=seed + 203,
        )
        effective_micro = np.logical_and(micro_pores, microporous_zone)
        attenuation = np.full(shape, 0.86, dtype=np.float32)
        attenuation[microporous_zone] = 0.72
        attenuation[effective_micro] = 0.22
        attenuation[macro_pores] = 0.12

    attenuation = gaussian_filter(attenuation, sigma=0.5, mode="reflect")
    attenuation = attenuation + rng.normal(0.0, 0.012, size=shape).astype(np.float32)
    return np.clip(attenuation, 0.0, 1.0).astype(np.float32, copy=False)


def fetch_or_generate_rock_sample(
    dataset_name: str,
    dest_dir: str | Path = "data/benchmarks",
    shape: tuple[int, int, int] = (128, 128, 128),
    allow_offline_fallback: bool = True,
    prefer_offline: bool = False,
    timeout_s: float = 30.0,
    seed: int = 42,
) -> tuple[Path, dict[str, object]]:
    """Download a verified open micro-CT benchmark stack (or synthesize an explicitly labeled phantom).

    Writes ``<dest_dir>/<dataset_name>_hr.tif`` and ``<dest_dir>/<dataset_name>_manifest.json``
    atomically under an exclusive file lock (safe across concurrent SLURM array tasks on Lustre).
    Provenance is strictly recorded in ``manifest['source_kind']`` (``'downloaded_zenodo_5542624'``
    vs. ``'calibrated_carbonate_phantom'`` / ``'offline_calibrated_phantom'``).
    """
    if dataset_name not in AVAILABLE_PUBLIC_DATASETS:
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. Available: {sorted(AVAILABLE_PUBLIC_DATASETS)}"
        )

    spec = AVAILABLE_PUBLIC_DATASETS[dataset_name]
    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_path = out_dir / f"{spec.name}_hr.tif"
    manifest_path = out_dir / f"{spec.name}_manifest.json"
    lock_path = out_dir / f".{spec.name}.lock"

    with lock_path.open("w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        if tif_path.is_file() and manifest_path.is_file():
            cached_manifest: dict[str, object] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            shape_ok = cached_manifest.get("requested_shape", cached_manifest.get("shape")) == list(
                shape
            )
            offline_cached = cached_manifest.get("source_kind") == "offline_calibrated_phantom"
            want_real = (spec.url is not None) and (not prefer_offline)
            if shape_ok and not (want_real and offline_cached):
                return tif_path, cached_manifest

        pid = os.getpid()
        source_kind = (
            "calibrated_carbonate_phantom" if spec.url is None else "downloaded_zenodo_5542624"
        )
        vol: np.ndarray | None = None

        if spec.url is not None and not prefer_offline:
            raw_tmp = out_dir / f".{spec.name}.{pid}.download"
            try:
                _download_https_file(spec.url, raw_tmp, timeout_s=timeout_s)
                if spec.sha256 is not None:
                    actual_sha = _hash_file(raw_tmp, "sha256")
                    if actual_sha.lower() != spec.sha256.lower():
                        raise ValueError(
                            f"SHA-256 mismatch for {spec.name}: expected {spec.sha256}, got {actual_sha}"
                        )
                if spec.md5 is not None:
                    actual_md5 = _hash_file(raw_tmp, "md5")
                    if actual_md5.lower() != spec.md5.lower():
                        raise ValueError(
                            f"MD5 mismatch for {spec.name}: expected {spec.md5}, got {actual_md5}"
                        )

                if spec.file_format == "tif":
                    loaded = tifffile.imread(raw_tmp)
                    cropped = _center_crop_3d(loaded, shape=shape)
                    vol = normalize_volume(
                        cropped, mode="percentile", lower_pct=0.2, upper_pct=99.8
                    )
                else:
                    raw_bytes = np.fromfile(raw_tmp, dtype=spec.raw_dtype)
                    expected_elems = int(np.prod(spec.raw_shape))
                    if raw_bytes.size >= expected_elems:
                        arr = raw_bytes[:expected_elems].reshape(spec.raw_shape)
                        cropped = _center_crop_3d(arr, shape=shape)
                        vol = normalize_volume(cropped, mode="minmax")
            except Exception as exc:
                if not allow_offline_fallback:
                    raise RuntimeError(
                        f"Failed to download {spec.name} from {spec.url} and offline fallback is disabled."
                    ) from exc
                source_kind = "offline_calibrated_phantom"
            finally:
                raw_tmp.unlink(missing_ok=True)

        if vol is None:
            if spec.url is not None:
                if not allow_offline_fallback and not prefer_offline:
                    raise RuntimeError(
                        f"Real dataset {spec.name} was not loaded and offline fallback is disabled."
                    )
                source_kind = "offline_calibrated_phantom"
            vol = _synthesize_calibrated_rock_phantom(spec=spec, shape=shape, seed=seed)

        is_exp = source_kind.startswith("downloaded")
        tif_tmp = out_dir / f".{spec.name}_hr.{pid}.tif"
        man_tmp = out_dir / f".{spec.name}_manifest.{pid}.json"
        tifffile.imwrite(tif_tmp, vol.astype(np.float32), photometric="minisblack")
        sha = _hash_file(tif_tmp, "sha256")
        manifest: dict[str, object] = {
            "name": spec.name,
            "lithology": spec.lithology,
            "description": spec.description,
            "source_kind": source_kind,
            "is_experimental_xct": is_exp,
            "source_url": spec.url if is_exp else None,
            "reference": spec.reference,
            "license": spec.license_name,
            "requested_shape": list(shape),
            "shape": list(vol.shape),
            "sha256": sha,
            "seed": seed,
        }
        man_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        tif_tmp.replace(tif_path)
        man_tmp.replace(manifest_path)
        return tif_path, manifest


def _coaxial_center_crop_pair_3d(
    lr_arr: np.ndarray,
    hr_arr: np.ndarray,
    lr_shape: tuple[int, int, int],
    scale_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Coaxially center-crop an integer-scale ``(LR, HR)`` pair with zero voxel phase offset."""
    s = int(scale_factor)
    d_lr = min(int(lr_arr.shape[0]), int(hr_arr.shape[0]) // s, int(lr_shape[0]))
    h_lr = min(int(lr_arr.shape[1]), int(hr_arr.shape[1]) // s, int(lr_shape[1]))
    w_lr = min(int(lr_arr.shape[2]), int(hr_arr.shape[2]) // s, int(lr_shape[2]))
    z0_lr = max(0, (int(lr_arr.shape[0]) - d_lr) // 2)
    y0_lr = max(0, (int(lr_arr.shape[1]) - h_lr) // 2)
    x0_lr = max(0, (int(lr_arr.shape[2]) - w_lr) // 2)
    z0_hr, y0_hr, x0_hr = s * z0_lr, s * y0_lr, s * x0_lr
    lr_crop = np.ascontiguousarray(
        lr_arr[z0_lr : z0_lr + d_lr, y0_lr : y0_lr + h_lr, x0_lr : x0_lr + w_lr],
        dtype=np.float32,
    )
    hr_crop = np.ascontiguousarray(
        hr_arr[
            z0_hr : z0_hr + s * d_lr,
            y0_hr : y0_hr + s * h_lr,
            x0_hr : x0_hr + s * w_lr,
        ],
        dtype=np.float32,
    )
    return lr_crop, hr_crop


def _crop_and_register_linear_dualscan_pair(
    lo_full: np.ndarray,
    hi_full: np.ndarray,
    req_lr: tuple[int, int, int],
    scale_factor: int,
) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    """Align a dual-scan pair on the integer HR voxel grid in linear uint16/65535 space without interpolation."""
    s = int(scale_factor)
    d_lr = min(int(lo_full.shape[0]), int(hi_full.shape[0]) // s, int(req_lr[0]))
    h_lr = min(int(lo_full.shape[1]), int(hi_full.shape[1]) // s, int(req_lr[1]))
    w_lr = min(int(lo_full.shape[2]), int(hi_full.shape[2]) // s, int(req_lr[2]))

    z0_lr = max(0, (int(lo_full.shape[0]) - d_lr) // 2)
    y0_lr = max(0, (int(lo_full.shape[1]) - h_lr) // 2)
    x0_lr = max(0, (int(lo_full.shape[2]) - w_lr) // 2)
    z0_hr, y0_hr, x0_hr = s * z0_lr, s * y0_lr, s * x0_lr

    lr_init = np.ascontiguousarray(
        lo_full[z0_lr : z0_lr + d_lr, y0_lr : y0_lr + h_lr, x0_lr : x0_lr + w_lr],
        dtype=np.float32,
    )
    hr_init = np.ascontiguousarray(
        hi_full[z0_hr : z0_hr + s * d_lr, y0_hr : y0_hr + s * h_lr, x0_hr : x0_hr + s * w_lr],
        dtype=np.float32,
    )

    lo_reg_in = normalize_volume(lr_init, mode="percentile", lower_pct=0.5, upper_pct=99.5)
    hi_reg_in = normalize_volume(hr_init, mode="percentile", lower_pct=0.5, upper_pct=99.5)
    _, _, pre_reg = register_multires_pair_3d(
        lo_reg_in, hi_reg_in, scale_factor=s, local_refine_radius=2
    )
    raw_shift_obj = pre_reg.get("hr_shift_zyx", [0, 0, 0])
    assert isinstance(raw_shift_obj, list)
    raw_shift = [int(raw_shift_obj[0]), int(raw_shift_obj[1]), int(raw_shift_obj[2])]
    init_ssim_obj = pre_reg.get("initial_triplanar_ssim", 0.0)
    raw_init_ssim = float(init_ssim_obj) if isinstance(init_ssim_obj, (int, float)) else 0.0

    if tuple(raw_shift) == (0, 0, 0):
        return lr_init, hr_init, raw_shift, raw_init_ssim

    # Exact HR-grid alignment: hr_start_aligned = s * lr_start - shift_hr_zyx.
    # When hi_full has outer margin around the sub-crop (e.g., 36x32x32 inside 75^3 / 225^3),
    # shift the HR crop window directly inside hi_full to preserve the full requested shape;
    # otherwise (full 75^3 -> 225^3) slice the overlap via apply_hr_grid_shift_pair_3d.
    za_hr = z0_hr - raw_shift[0]
    ya_hr = y0_hr - raw_shift[1]
    xa_hr = x0_hr - raw_shift[2]
    if (
        0 <= za_hr
        and za_hr + s * d_lr <= int(hi_full.shape[0])
        and 0 <= ya_hr
        and ya_hr + s * h_lr <= int(hi_full.shape[1])
        and 0 <= xa_hr
        and xa_hr + s * w_lr <= int(hi_full.shape[2])
    ):
        hr_aligned = np.ascontiguousarray(
            hi_full[za_hr : za_hr + s * d_lr, ya_hr : ya_hr + s * h_lr, xa_hr : xa_hr + s * w_lr],
            dtype=np.float32,
        )
        return lr_init, hr_aligned, raw_shift, raw_init_ssim

    sliced = apply_hr_grid_shift_pair_3d(
        lr_init,
        hr_init,
        shift_hr_zyx=(raw_shift[0], raw_shift[1], raw_shift[2]),
        scale_factor=s,
        min_overlap_lr=min(8, int(min(d_lr, h_lr, w_lr))),
    )
    if sliced is not None:
        return sliced[0], sliced[1], raw_shift, raw_init_ssim
    return lr_init, hr_init, raw_shift, raw_init_ssim


def _read_verified_uint16_raw(
    raw_path: Path,
    expected_shape: tuple[int, int, int],
    raw_dtype: str = "<u2",
    expected_md5: str | None = None,
) -> np.ndarray:
    """Load a raw binary volume, verify MD5 when requested, and scale linearly by 1/65535."""
    if expected_md5 is not None:
        actual_md5 = _hash_file(raw_path, "md5")
        if actual_md5.lower() != expected_md5.lower():
            raise ValueError(
                f"MD5 mismatch for {raw_path.name}: expected {expected_md5}, got {actual_md5}"
            )
    expected_elems = int(np.prod(expected_shape))
    arr = np.fromfile(raw_path, dtype=raw_dtype)
    if arr.size != expected_elems:
        raise ValueError(
            f"Raw element count mismatch for {raw_path.name}: "
            f"expected {expected_elems} ({expected_shape}), got {arr.size}"
        )
    reshaped = arr.reshape(expected_shape)
    if np.dtype(raw_dtype).kind == "u" and np.dtype(raw_dtype).itemsize == 2:
        return (reshaped.astype(np.float32) / 65535.0).astype(np.float32, copy=False)
    return normalize_volume(reshaped.astype(np.float32), mode="minmax")


def _synthesize_calibrated_dualscan_phantom_pair(
    spec: PublicDualScanSpec,
    lr_shape: tuple[int, int, int],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize an explicitly labeled offline 3x dual-scan sandstone phantom pair."""
    s = int(spec.scale_factor)
    hr_shape = (int(lr_shape[0]) * s, int(lr_shape[1]) * s, int(lr_shape[2]) * s)
    rng = np.random.default_rng(seed)
    pore_mask = _porespy_blobs(
        shape=hr_shape,
        blobiness=spec.macro_blobiness,
        porosity=spec.target_porosity,
        seed=seed,
    )
    grain_dt = distance_transform_edt(~pore_mask).astype(np.float32)
    grain_dt = grain_dt / max(float(grain_dt.max()), 1.0)
    clean_hr = np.where(pore_mask, 0.14, 0.78 + 0.12 * grain_dt).astype(np.float32)

    hr_vol = gaussian_filter(clean_hr, sigma=0.55, mode="reflect")
    hr_vol = np.clip(
        hr_vol + rng.normal(0.0, 0.014, size=hr_shape).astype(np.float32), 0.0, 1.0
    ).astype(np.float32, copy=False)

    lr_deg = degrade_xct_volume(
        clean_hr,
        scale_factor=s,
        psf_sigma=1.05,
        noise_std=0.0,
        seed=seed + 17,
    )
    # Emulate polychromatic full-core vs interior-ROI beam-hardening tonal shift
    lr_shifted = 0.05 + 0.86 * np.power(np.clip(lr_deg, 0.0, 1.0), 1.25)
    lr_vol = np.clip(
        lr_shifted + rng.normal(0.0, 0.018, size=lr_deg.shape).astype(np.float32), 0.0, 1.0
    ).astype(np.float32, copy=False)
    return lr_vol, hr_vol


def prepare_bentheimer_dualscan_benchmark(
    dataset_name: str = "bentheimer_dualscan_core1",
    dest_dir: str | Path = "data/benchmarks",
    lr_shape: tuple[int, int, int] = (75, 75, 75),
    raw_dir: str | Path | None = None,
    allow_offline_fallback: bool = True,
    prefer_offline: bool = False,
    verify_checksum: bool = True,
    timeout_s: float = 90.0,
    seed: int = 42,
) -> tuple[Path, Path, dict[str, object]]:
    """Prepare a physical 18 um -> 6 um (3x) Bentheimer dual-scan micro-CT benchmark pair.

    Writes ``<dest_dir>/<dataset_name>_lr.tif``, ``<dest_dir>/<dataset_name>_hr.tif``, and
    ``<dest_dir>/<dataset_name>_manifest.json`` atomically under an exclusive file lock.

    Intensities of real 16-bit Zenodo ``.raw`` scans are scaled strictly linearly by
    ``1 / 65535.0`` without computing whole-volume percentiles, ensuring downstream
    ``PairedVolumeDataModule`` fits both ``[lower_pct, upper_pct]`` normalization bounds and
    ``cross_scan_calibration`` exclusively on the ``train`` slab with zero ``val``/``test`` leakage.
    """
    if dataset_name not in AVAILABLE_DUALSCAN_DATASETS:
        raise KeyError(
            f"Unknown dual-scan dataset {dataset_name!r}. "
            f"Available: {sorted(AVAILABLE_DUALSCAN_DATASETS)}"
        )

    spec = AVAILABLE_DUALSCAN_DATASETS[dataset_name]
    req_lr = (int(lr_shape[0]), int(lr_shape[1]), int(lr_shape[2]))
    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir = out_dir / "raw_cache"
    lr_tif = out_dir / f"{spec.name}_lr.tif"
    hr_tif = out_dir / f"{spec.name}_hr.tif"
    manifest_path = out_dir / f"{spec.name}_manifest.json"
    lock_path = out_dir / f".{spec.name}.lock"

    with lock_path.open("w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        if lr_tif.is_file() and hr_tif.is_file() and manifest_path.is_file():
            cached: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
            shape_ok = cached.get("requested_lr_shape") == list(req_lr)
            offline_cached = cached.get("source_kind") == "offline_calibrated_dualscan_phantom"
            want_real = (not prefer_offline) or (raw_dir is not None)
            reg_cached = cached.get("registration")
            aligned_ok = isinstance(reg_cached, dict) and reg_cached.get("hr_shift_zyx") == [
                0,
                0,
                0,
            ]
            if shape_ok and aligned_ok and not (want_real and offline_cached):
                return lr_tif, hr_tif, cached

        pid = os.getpid()
        lr_vol: np.ndarray | None = None
        hr_vol: np.ndarray | None = None
        reg_info: dict[str, object] = {}
        raw_detected_shift: list[int] | None = None
        raw_initial_ssim: float | None = None
        source_kind = "offline_calibrated_dualscan_phantom"

        # 1. Try explicit raw_dir or persistent raw_cache_dir first
        candidate_dirs: list[Path] = []
        if raw_dir is not None:
            candidate_dirs.append(Path(raw_dir))
        if not prefer_offline:
            candidate_dirs.append(raw_cache_dir)

        for cdir in candidate_dirs:
            lo_raw_p = cdir / spec.low_filename
            hi_raw_p = cdir / spec.high_filename
            if lo_raw_p.is_file() and hi_raw_p.is_file():
                lo_full = _read_verified_uint16_raw(
                    lo_raw_p,
                    expected_shape=spec.low_raw_shape,
                    raw_dtype=spec.low_raw_dtype,
                    expected_md5=spec.low_md5 if verify_checksum else None,
                )
                hi_full = _read_verified_uint16_raw(
                    hi_raw_p,
                    expected_shape=spec.high_raw_shape,
                    raw_dtype=spec.high_raw_dtype,
                    expected_md5=spec.high_md5 if verify_checksum else None,
                )
                lr_vol, hr_vol, raw_detected_shift, raw_initial_ssim = (
                    _crop_and_register_linear_dualscan_pair(
                        lo_full, hi_full, req_lr=req_lr, scale_factor=spec.scale_factor
                    )
                )
                source_kind = (
                    "local_zenodo_5542624_dualscan_raw"
                    if (raw_dir is not None and cdir == Path(raw_dir))
                    else "downloaded_zenodo_5542624_dualscan"
                )
                break

        # 2. Download from Zenodo 5542624 HTTPS and persist verified .raw files in raw_cache_dir
        if lr_vol is None and not prefer_offline:
            raw_cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_lo = raw_cache_dir / f".{spec.low_filename}.{pid}.part"
            tmp_hi = raw_cache_dir / f".{spec.high_filename}.{pid}.part"
            cached_lo = raw_cache_dir / spec.low_filename
            cached_hi = raw_cache_dir / spec.high_filename
            try:
                if not cached_lo.is_file():
                    _download_https_file(spec.low_url, tmp_lo, timeout_s=timeout_s)
                    if verify_checksum:
                        lo_md5 = _hash_file(tmp_lo, "md5")
                        if lo_md5.lower() != spec.low_md5.lower():
                            raise ValueError(
                                f"MD5 mismatch for {spec.low_filename}: "
                                f"expected {spec.low_md5}, got {lo_md5}"
                            )
                    tmp_lo.replace(cached_lo)
                if not cached_hi.is_file():
                    _download_https_file(spec.high_url, tmp_hi, timeout_s=timeout_s)
                    if verify_checksum:
                        hi_md5 = _hash_file(tmp_hi, "md5")
                        if hi_md5.lower() != spec.high_md5.lower():
                            raise ValueError(
                                f"MD5 mismatch for {spec.high_filename}: "
                                f"expected {spec.high_md5}, got {hi_md5}"
                            )
                    tmp_hi.replace(cached_hi)

                lo_full = _read_verified_uint16_raw(
                    cached_lo,
                    expected_shape=spec.low_raw_shape,
                    raw_dtype=spec.low_raw_dtype,
                    expected_md5=spec.low_md5 if verify_checksum else None,
                )
                hi_full = _read_verified_uint16_raw(
                    cached_hi,
                    expected_shape=spec.high_raw_shape,
                    raw_dtype=spec.high_raw_dtype,
                    expected_md5=spec.high_md5 if verify_checksum else None,
                )
                lr_vol, hr_vol, raw_detected_shift, raw_initial_ssim = (
                    _crop_and_register_linear_dualscan_pair(
                        lo_full, hi_full, req_lr=req_lr, scale_factor=spec.scale_factor
                    )
                )
                source_kind = "downloaded_zenodo_5542624_dualscan"
            except Exception as exc:
                if not allow_offline_fallback:
                    raise RuntimeError(
                        f"Failed to download dual-scan pair {spec.name} from Zenodo 5542624 "
                        "and offline fallback is disabled."
                    ) from exc
                source_kind = "offline_calibrated_dualscan_phantom"
            finally:
                tmp_lo.unlink(missing_ok=True)
                tmp_hi.unlink(missing_ok=True)

        # 3. Offline calibrated dual-scan phantom fallback
        if lr_vol is None or hr_vol is None:
            if not allow_offline_fallback and not prefer_offline:
                raise RuntimeError(
                    f"Real dual-scan dataset {spec.name} was not loaded and offline fallback is disabled."
                )
            lr_vol, hr_vol = _synthesize_calibrated_dualscan_phantom_pair(
                spec=spec, lr_shape=req_lr, seed=seed
            )
            source_kind = "offline_calibrated_dualscan_phantom"

        # Verify 3D registration diagnostics on [0, 1] contrast-matched copies without mutating
        # the split-isolated linear uint16 dynamic range stored on disk.
        lo_reg_in = normalize_volume(lr_vol, mode="percentile", lower_pct=0.5, upper_pct=99.5)
        hi_reg_in = normalize_volume(hr_vol, mode="percentile", lower_pct=0.5, upper_pct=99.5)
        _, _, reg_info = register_multires_pair_3d(
            lo_reg_in, hi_reg_in, scale_factor=spec.scale_factor, local_refine_radius=2
        )
        if raw_detected_shift is not None:
            reg_info["raw_detected_hr_shift_zyx"] = raw_detected_shift
            reg_info["initial_unaligned_triplanar_ssim"] = raw_initial_ssim

        is_exp = source_kind in {
            "downloaded_zenodo_5542624_dualscan",
            "local_zenodo_5542624_dualscan_raw",
        }
        lr_tmp = out_dir / f".{spec.name}_lr.{pid}.tif"
        hr_tmp = out_dir / f".{spec.name}_hr.{pid}.tif"
        man_tmp = out_dir / f".{spec.name}_manifest.{pid}.json"
        tifffile.imwrite(lr_tmp, lr_vol.astype(np.float32), photometric="minisblack")
        tifffile.imwrite(hr_tmp, hr_vol.astype(np.float32), photometric="minisblack")

        manifest: dict[str, object] = {
            "name": spec.name,
            "lithology": spec.lithology,
            "description": spec.description,
            "scale_factor": spec.scale_factor,
            "low_resolution_um": spec.low_voxel_size_um,
            "high_resolution_um": spec.high_voxel_size_um,
            "source_kind": source_kind,
            "is_experimental_xct": is_exp,
            "zenodo_doi": spec.zenodo_doi if is_exp else None,
            "low_source_url": spec.low_url if is_exp else None,
            "high_source_url": spec.high_url if is_exp else None,
            "reference": spec.reference,
            "license": spec.license_name,
            "requested_lr_shape": list(req_lr),
            "lr_shape": list(lr_vol.shape),
            "hr_shape": list(hr_vol.shape),
            "lr_sha256": _hash_file(lr_tmp, "sha256"),
            "hr_sha256": _hash_file(hr_tmp, "sha256"),
            "registration": reg_info,
            "seed": seed,
        }
        man_tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        lr_tmp.replace(lr_tif)
        hr_tmp.replace(hr_tif)
        man_tmp.replace(manifest_path)
        return lr_tif, hr_tif, manifest
