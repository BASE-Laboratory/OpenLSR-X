"""
This file test the blocks.py file

To check:
1) ConvBlock() -> class
2) UpsampleBlock() -> class
3) ResidualBlock() -> class

I will test each class by checking on a random input tensor

"""

import unittest
import torch
from blocks import ConvBlock, UpsampleBlock, ResidualBlock


class Test_ConvBlock(unittest.TestCase):
    """
    Test the ConvBlock class
    1) Check output size
    2) Check output type
    """

    def setUp(self):
        self.in_channels = 1
        self.out_channels = 64
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.padding_mode = "reflect"

        self.conv_block = ConvBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            padding_mode=self.padding_mode,
        )

        self.input_tensor = torch.randn(1, self.in_channels, 100, 100)

    def test_output_size(self):

        output = self.conv_block(self.input_tensor)
        assert output.shape == torch.Size([1, self.out_channels, 100, 100])

    def test_output_type(self):

        output = self.conv_block(self.input_tensor)
        assert isinstance(output, torch.Tensor)


class Test_UpsampleBlock(unittest.TestCase):
    """
    Test the UpsampleBlock class
    1) Check output size
    2) Check output type
    """

    def setUp(self):
        self.in_channels = 1
        self.scale_factor = 2

        self.upsample_block = UpsampleBlock(
            in_channels=self.in_channels,
            scale_factor=self.scale_factor,
        )

        self.input_tensor = torch.randn(1, self.in_channels, 100, 100)

    def test_output_size(self):

        output = self.upsample_block(self.input_tensor)
        assert output.shape == torch.Size([1, self.in_channels, 200, 200])

    def test_output_type(self):

        output = self.upsample_block(self.input_tensor)
        assert isinstance(output, torch.Tensor)


class Test_ResidualBlock(unittest.TestCase):
    """
    Test the ResidualBlock class
    1) Check output size
    2) Check output type
    """

    def setUp(self):
        self.in_channels = 1

        self.residual_block = ResidualBlock(
            in_channels=self.in_channels,
        )

        self.input_tensor = torch.randn(1, self.in_channels, 100, 100)

    def test_output_size(self):

        output = self.residual_block(self.input_tensor)
        assert output.shape == torch.Size([1, self.in_channels, 100, 100])

    def test_output_type(self):

        output = self.residual_block(self.input_tensor)
        assert isinstance(output, torch.Tensor)
