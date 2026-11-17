# API Reference

## Models

::: openlsrx.models.srgan.SRGAN

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

## Data Modules & Transforms

::: openlsrx.data.synthetic.SyntheticPoreSpyDataModule

::: openlsrx.data.synthetic.generate_porespy_volume

::: openlsrx.data.volumes.PairedVolumeDataModule

::: openlsrx.data.volumes.load_volume_any

::: openlsrx.data.transforms.split_volume_spatially

::: openlsrx.data.transforms.degrade_xct_volume

---

## Metrics & Baselines

::: openlsrx.metrics.image.evaluate_image_metrics

::: openlsrx.metrics.petrophysics.evaluate_petrophysical_metrics

::: openlsrx.baselines.interpolation.upsample_2d

::: openlsrx.baselines.interpolation.upsample_3d
