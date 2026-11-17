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
from .zn13x import (
    ZN13X_I13_RESOLUTIONS,
    ZN13X_SUBVOLUMES_ZENODO_RECORD,
    Zn13XPairName,
    Zn13XResolutionSpec,
    extract_aligned_slab_from_h5,
    prepare_zn13x_benchmark,
    register_multires_pair_3d,
    synthesize_zn13x_phantom_pair,
)

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
    "ZN13X_I13_RESOLUTIONS",
    "ZN13X_SUBVOLUMES_ZENODO_RECORD",
    "Zn13XPairName",
    "Zn13XResolutionSpec",
    "apply_alignment",
    "degrade_xct_volume",
    "downsample_orthogonal_slabs",
    "extract_aligned_slab_from_h5",
    "extract_paired_slices_from_subvolume",
    "fetch_or_generate_rock_sample",
    "generate_porespy_volume",
    "load_volume_any",
    "normalize_volume",
    "prepare_zn13x_benchmark",
    "register_multires_pair_3d",
    "split_volume_spatially",
    "synthesize_zn13x_phantom_pair",
]
