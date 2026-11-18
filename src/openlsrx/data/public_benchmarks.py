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
from .transforms import normalize_volume


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
    req = urllib.request.Request(url, headers={"User-Agent": "OpenLSR-X/0.2.0"})
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
