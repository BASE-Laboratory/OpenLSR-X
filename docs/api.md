# API Reference

## Models (2D SRGAN, RRDB-ESRGAN & Native 3D CNN)

::: openlsrx.models.srgan.SRGAN

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

## Losses & Normalizations

::: openlsrx.losses.loss.VGGLoss

::: openlsrx.losses.loss.PixelNorm

::: openlsrx.losses.loss.custom_sigmoid

---

## 3D Orthogonal Inference

::: openlsrx.inference.orthogonal.super_resolve_volume_orthogonal

::: openlsrx.inference.orthogonal.super_resolve_volume_single_axis

::: openlsrx.inference.orthogonal.fuse_orthogonal_volumes

---

## Data Modules, Synchrotron Registration & Transforms

::: openlsrx.data.synthetic.SyntheticPoreSpyDataModule

::: openlsrx.data.synthetic.generate_porespy_volume

::: openlsrx.data.volumes.PairedVolumeDataModule

::: openlsrx.data.volumes.load_volume_any

::: openlsrx.data.zn13x.prepare_zn13x_benchmark

::: openlsrx.data.zn13x.register_multires_pair_3d

::: openlsrx.data.zn13x.extract_aligned_slab_from_h5

::: openlsrx.data.transforms.split_volume_spatially

::: openlsrx.data.transforms.degrade_xct_volume

---

## Metrics, Transport Physics & OpenImpala I/O

::: openlsrx.metrics.image.evaluate_image_metrics

::: openlsrx.metrics.petrophysics.evaluate_petrophysical_metrics

::: openlsrx.metrics.transport.evaluate_transport_metrics

::: openlsrx.metrics.transport.compute_directional_tortuosity

::: openlsrx.metrics.transport.compute_drainage_rmse

::: openlsrx.io.openimpala.export_for_openimpala

::: openlsrx.baselines.interpolation.upsample_2d

::: openlsrx.baselines.interpolation.upsample_3d
