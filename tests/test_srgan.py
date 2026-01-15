"""
This file tests the srgan.py file

To check:
1) SRGAN() -> class
"""

import unittest
import torch
from openlsrx.models.srgan import SRGAN
from openlsrx.models.submodels import Generator, Discriminator
import pytorch_lightning as pl
from openlsrx.data.dataset import DatasetObject
from torch.utils.data import DataLoader


class Test_SRGAN(unittest.TestCase):
    """
    Test the SRGAN class

    To test:

    1) Training Step
    2) On train epoch end
    3) Validation Step
    4) On validation epoch end
    """

    def setUp(self):

        self.gen = Generator()
        self.disc = Discriminator()
        self.learning_rate = 0.0001

        # takes [C,H,W] input
        self.low_res = torch.randn(1, 16, 16)
        self.high_res = torch.randn(1, 32, 32)

        self.srgan = SRGAN(
            self.learning_rate,
            adaptive_size=2,
            accumulate_n_batches=1,
            super_res_factor=2,
            num_blocks=1,
            disc_features=[8, 8],
        )

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.trainer = pl.Trainer(
            max_epochs=1,
            accelerator=accelerator,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            limit_train_batches=1,
        )

    def test_init(self):
        "This test is to check the initialisation of the SRGAN works"

        self.assertIsInstance(self.srgan.gen, Generator)
        self.assertIsInstance(self.srgan.disc, Discriminator)
        self.assertEqual(self.srgan.learning_rate, self.learning_rate)

    def test_training_step(self):
        """
        This is to test that the training step works as intended
        I want to check the losses are updated after a training step
        """
        # perform single step with trainer
        self.trainer.fit(
            self.srgan,
            train_dataloaders=DataLoader(
                DatasetObject(self.low_res, self.high_res), batch_size=1
            ),
        )

        # check losses are tensors (loss buffers reset at epoch end)
        assert isinstance(self.srgan.disc_losses, torch.Tensor)
        assert isinstance(self.srgan.gen_losses, torch.Tensor)

    def test_on_train_epoch_end(self):
        pass

    def test_validation_step(self):
        pass

    def test_on_validation_epoch_end(self):
        pass
