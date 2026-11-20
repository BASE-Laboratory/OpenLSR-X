# API Reference

## Models (SwinIR-2.5D, RRDB-ESRGAN, SRGAN & Native 3D CNN)

::: openlsrx.models.srgan.SRGAN

::: openlsrx.models.swinir.SwinIRGenerator

::: openlsrx.models.swinir.ResidualSwinBlock

::: openlsrx.models.rrdb.RRDBGenerator

::: openlsrx.models.rrdb.ArbitraryScaleUpsample

::: openlsrx.models.rrdb.RRDB

::: openlsrx.models.sr3d.SR3DResNet

::: openlsrx.models.sr3d.Generator3D

::: openlsrx.models.sr3d.super_resolve_volume_3d

::: openlsrx.models.submodels.Generator

::: openlsrx.models.submodels.Discriminator

::: openlsrx.models.blocks.ConvBlock

::: openlsrx.models.blocks.UpsampleBlock

::: openlsrx.models.blocks.ResidualBlock

---

## Physics, Topology & Perceptual Losses

::: openlsrx.losses.physics.AttenuationConservationLoss

::: openlsrx.losses.physics.SoftCLDiceLoss

::: openlsrx.losses.physics.SobelGradientLoss

::: openlsrx.losses.physics.FocalFrequencyLoss

::: openlsrx.losses.loss.VGGLoss

::: openlsrx.losses.loss.PixelNorm

::: openlsrx.losses.loss.custom_sigmoid

---

## 3D Orthogonal Inference & View-Disagreement Diagnostics

::: openlsrx.inference.orthogonal.super_resolve_volume_orthogonal

::: openlsrx.inference.orthogonal.super_resolve_volume_single_axis

::: openlsrx.inference.orthogonal.fuse_orthogonal_volumes

::: openlsrx.inference.orthogonal.compute_orthogonal_uncertainty

---

## Data Modules, Dual-Scan & Synchrotron Registration & Transforms

::: openlsrx.data.synthetic.SyntheticPoreSpyDataModule

::: openlsrx.data.synthetic.generate_porespy_volume

::: openlsrx.data.volumes.PairedVolumeDataModule

::: openlsrx.data.volumes.load_volume_any

::: openlsrx.data.public_benchmarks.prepare_bentheimer_dualscan_benchmark

::: openlsrx.data.zn13x.prepare_zn13x_benchmark

::: openlsrx.data.zn13x.SimilarityTransform3DSpec

::: openlsrx.data.zn13x.compose_similarity_transforms_3d

::: openlsrx.data.zn13x.resolve_zn13x_pair_7dof

::: openlsrx.data.zn13x.resample_lr_to_hr_grid_7dof

::: openlsrx.data.zn13x.register_multires_pair_3d

::: openlsrx.data.zn13x.apply_hr_grid_shift_pair_3d

::: openlsrx.data.zn13x.extract_aligned_slab_from_h5

::: openlsrx.data.transforms.split_volume_spatially

::: openlsrx.data.transforms.degrade_xct_volume

::: openlsrx.data.transforms.make_slice_dataloader

---

## Spectral Metrology, Petrophysics, Transport & OpenImpala I/O

::: openlsrx.metrics.spectral.compute_fsc_3d

::: openlsrx.metrics.spectral.compute_frc_2d

::: openlsrx.metrics.spectral.evaluate_spectral_metrics

::: openlsrx.metrics.image.evaluate_image_metrics

::: openlsrx.metrics.petrophysics.evaluate_petrophysical_metrics

::: openlsrx.metrics.petrophysics.evaluate_triphase_zn13x_metrics

::: openlsrx.metrics.transport.evaluate_transport_metrics

::: openlsrx.metrics.transport.compute_tortuosity_uncertainty_bounds

::: openlsrx.metrics.transport.compute_directional_tortuosity

::: openlsrx.metrics.transport.compute_drainage_rmse

::: openlsrx.io.openimpala.export_volume_ome_zarr

::: openlsrx.io.openimpala.export_for_openimpala

::: openlsrx.baselines.interpolation.upsample_2d

::: openlsrx.baselines.interpolation.upsample_3d
