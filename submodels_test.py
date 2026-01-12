"""
This file tests the submodels.py file

To check:
1) Generator() -> class
2) Discriminator() -> class

"""

import unittest
import torch
from submodels import Generator, Discriminator


class Test_Generator(unittest.TestCase):
    """
    Test the Generator class
    1) Check output size
    2) Check output type
    """

    def setUp(self):
        self.batch_size = 2
        self.in_channels = 1
        self.out_channels = 64
        self.num_blocks = 10

        self.generator = Generator(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_blocks=self.num_blocks,
        )

        self.input_tensor = torch.randn(self.batch_size, self.in_channels, 100, 100)

    def test_output_size(self):

        output = self.generator(self.input_tensor)
        assert output.shape == torch.Size([self.batch_size, self.in_channels, 400, 400])

    def test_output_type(self):

        output = self.generator(self.input_tensor)
        assert isinstance(output, torch.Tensor)


class Test_Discriminator(unittest.TestCase):
    """
    This tests the Discriminator class

    1) Check output size
    2) Check output type
    """

    def setUp(self):

        self.in_channels = 1
        self.features = [64, 64, 128, 128, 256, 256, 512, 512]

        self.input_data = torch.randn(1, self.in_channels, 100, 100)

        self.discriminator = Discriminator(self.in_channels, self.features)

    def test_output_size(self):

        output = self.discriminator(self.input_data)
        assert output.shape == torch.Size([1, 1])

    def test_output_type(self):

        output = self.discriminator(self.input_data)
        assert isinstance(output, torch.Tensor)
