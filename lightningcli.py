from pytorch_lightning.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    """
    Custom LightningCLI class that allows for handling of loading of older
    model variations that may not have been instantiated with the current
    model parameters such as 'use_vgg_loss' etc

    This class overrides the `instantiate_classes` method to handle
    loading of checkpoints with strict loading options and to instantiate
    the model and datamodule based on the provided configuration.
    """

    def instantiate_classes(self):
        subcmd_cfg = getattr(self.config, self.config.subcommand)
        model_kwargs = dict(getattr(subcmd_cfg, "model", {}))
        strict = model_kwargs.pop("strict_loading", True)
        ckpt_path = getattr(subcmd_cfg, "ckpt_path", None)
        if ckpt_path:
            print(f"Loading checkpoint with strict={strict}", flush=True)
            self.model = self.model_class.load_from_checkpoint(
                ckpt_path, strict=strict, strict_loading=strict, **model_kwargs
            )
        else:
            self.model = self.model_class(**model_kwargs)
        data_kwargs = dict(getattr(subcmd_cfg, "data", {}))
        if data_kwargs:
            self.datamodule = self.datamodule_class(**data_kwargs)
        else:
            self.datamodule = None
        # Set config_init for compatibility with instantiate_trainer and subcommands
        self.config_init = {}
        self.config_init["model"] = model_kwargs
        self.config_init["data"] = data_kwargs
        self.config_init["trainer"] = dict(getattr(subcmd_cfg, "trainer", {}))
        self.config_init[self.config.subcommand] = dict(subcmd_cfg)
        self.trainer = self.instantiate_trainer()
