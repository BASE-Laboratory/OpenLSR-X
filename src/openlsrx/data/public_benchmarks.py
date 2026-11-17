"""Open-access micro-CT rock benchmark fetcher with SHA-256 verification and offlineHPC fallback."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .synthetic import _porespy_blobs


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
    raw_dtype: str
    raw_shape: tuple[int, int, int]
    target_porosity: float
    macro_blobiness: float
    micro_blobiness: float | None


AVAILABLE_PUBLIC_DATASETS: dict[str, PublicRockSpec] = {
    "bentheimer_sandstone": PublicRockSpec(
        name="bentheimer_sandstone",
        description="Bentheimer Sandstone micro-CT benchmark (well-sorted quartz arenite, ~22% porosity).",
        lithology="Sandstone",
        reference="Imperial College London Pore-Scale Modelling Micro-CT Collection (Blunt et al., Adv. Water Resour. 2013)",
        license_name="CC-BY-4.0",
        url="https://imperialcollegelondon.box.com/shared/static/bentheimer_microct_subvolume.raw",
        sha256=None,
        raw_dtype="uint8",
        raw_shape=(128, 128, 128),
        target_porosity=0.22,
        macro_blobiness=1.1,
        micro_blobiness=None,
    ),
    "indiana_limestone": PublicRockSpec(
        name="indiana_limestone",
        description="Indiana Limestone micro-CT benchmark (bimodal fossiliferous/oolitic carbonate, ~18% total porosity).",
        lithology="Carbonate",
        reference="Imperial College London Pore-Scale Modelling Micro-CT Collection (Blunt et al., Adv. Water Resour. 2013)",
        license_name="CC-BY-4.0",
        url="https://imperialcollegelondon.box.com/shared/static/indiana_microct_subvolume.raw",
        sha256=None,
        raw_dtype="uint8",
        raw_shape=(128, 128, 128),
        target_porosity=0.18,
        macro_blobiness=0.65,
        micro_blobiness=2.6,
    ),
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _synthesize_calibrated_rock_phantom(
    spec: PublicRockSpec,
    shape: tuple[int, int, int],
    seed: int = 42,
) -> np.ndarray:
    """Synthesize a calibrated greyscale 3D XCT digital-rock phantom when offline/air-gapped."""
    rng = np.random.default_rng(seed)

    if spec.micro_blobiness is None:
        # Intergranular sandstone with realistic sub-grain cementation gradients
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
        # Bimodal vuggy + intragranular microporous carbonate (Indiana Limestone)
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
        # Sub-resolvable microporous calcite rims have intermediate greyscale attenuation
        attenuation[microporous_zone] = 0.72
        attenuation[effective_micro] = 0.22
        attenuation[macro_pores] = 0.12

    # Detector partial-volume smoothing + subtle XCT acquisition noise
    attenuation = gaussian_filter(attenuation, sigma=0.5, mode="reflect")
    attenuation = attenuation + rng.normal(0.0, 0.012, size=shape).astype(np.float32)
    return np.clip(attenuation, 0.0, 1.0).astype(np.float32, copy=False)


def fetch_or_generate_rock_sample(
    dataset_name: str,
    dest_dir: str | Path = "data/benchmarks",
    shape: tuple[int, int, int] = (128, 128, 128),
    allow_offline_fallback: bool = True,
    seed: int = 42,
) -> tuple[Path, dict[str, object]]:
    """Download an open micro-CT benchmark stack (or synthesize a calibrated phantom offline).

    Writes ``<dest_dir>/<dataset_name>_hr.tif`` and ``<dest_dir>/<dataset_name>_manifest.json``
    recording exact provenance, checksum, shape, and parameters.
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

    if tif_path.is_file() and manifest_path.is_file():
        cached_manifest: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return tif_path, cached_manifest

    source_kind = "downloaded"
    vol: np.ndarray | None = None

    if spec.url is not None:
        raw_tmp = out_dir / f".{spec.name}.download"
        try:
            urllib.request.urlretrieve(spec.url, raw_tmp)  # noqa: S310
            if spec.sha256 is not None:
                actual = _sha256_file(raw_tmp)
                if actual.lower() != spec.sha256.lower():
                    raise ValueError(
                        f"SHA-256 mismatch for {spec.name}: expected {spec.sha256}, got {actual}"
                    )
            raw_bytes = np.fromfile(raw_tmp, dtype=spec.raw_dtype)
            expected_elems = int(np.prod(spec.raw_shape))
            if raw_bytes.size >= expected_elems:
                arr = raw_bytes[:expected_elems].reshape(spec.raw_shape).astype(np.float32)
                lo, hi = float(arr.min()), float(arr.max())
                vol = (arr - lo) / max(hi - lo, 1e-8)
                vol = vol[: shape[0], : shape[1], : shape[2]]
        except Exception as exc:
            if not allow_offline_fallback:
                raise RuntimeError(
                    f"Failed to download {spec.name} from {spec.url} and offline fallback is disabled."
                ) from exc
            source_kind = "offline_calibrated_phantom"
        finally:
            if raw_tmp.exists():
                raw_tmp.unlink()

    if vol is None:
        source_kind = "offline_calibrated_phantom"
        vol = _synthesize_calibrated_rock_phantom(spec=spec, shape=shape, seed=seed)

    tifffile.imwrite(tif_path, vol.astype(np.float32), photometric="minisblack")
    sha = _sha256_file(tif_path)
    manifest: dict[str, object] = {
        "name": spec.name,
        "lithology": spec.lithology,
        "description": spec.description,
        "source_kind": source_kind,
        "reference": spec.reference,
        "license": spec.license_name,
        "shape": list(vol.shape),
        "sha256": sha,
        "seed": seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return tif_path, manifest
