"""Data loading, synthetic porous-media generation, and XCT preprocessing modules."""

from .public_benchmarks import AVAILABLE_PUBLIC_DATASETS, fetch_or_generate_rock_sample
from .synthetic import (
    FeatureRegime,
    NoiseRegime,
    SyntheticPoreSpyDataModule,
    SyntheticRepresentation,
    generate_porespy_volume,
)
from .transforms import (
    DatasetObject,
    NormMode,
    PairedSliceDataset,
    VolumeAlignment,
    apply_alignment,
    degrade_xct_volume,
    downsample_orthogonal_slabs,
    extract_paired_slices_from_subvolume,
    normalize_volume,
    split_volume_spatially,
)
from .volumes import PairedVolumeDataModule, load_volume_any

__all__ = [
    "AVAILABLE_PUBLIC_DATASETS",
    "DatasetObject",
    "FeatureRegime",
    "NoiseRegime",
    "NormMode",
    "PairedSliceDataset",
    "PairedVolumeDataModule",
    "SyntheticPoreSpyDataModule",
    "SyntheticRepresentation",
    "VolumeAlignment",
    "apply_alignment",
    "degrade_xct_volume",
    "downsample_orthogonal_slabs",
    "extract_paired_slices_from_subvolume",
    "fetch_or_generate_rock_sample",
    "generate_porespy_volume",
    "load_volume_any",
    "normalize_volume",
    "split_volume_spatially",
]
