"""
This file defines the unit tests for the dataset.py file

To check:

data_processor() -> function
DatasetObject() -> class
MyDataModule() -> class
PoreSpy_3DVolume() -> class (later)
"""

import unittest
from openlsrx.data.dataset import data_processor, DatasetObject, MyDataModule

# from openlsrx.data.dataset import PoreSpy3D_Volume
import torch


class Test_data_processor(unittest.TestCase):
    """
    Test the data_processor function
    1) Check types of output -> torch.Tensor
    2) Check size of output -> [C,H,W]
    """

    def setUp(self):
        self.dataset_size = 10
        self.image_size = 100
        self.lf = False
        self.valid = self.test = False

        self.low_test_data, self.high_test_data = data_processor(
            self.dataset_size, self.image_size, self.lf, self.valid, self.test
        )

    def test_assert_is_type(self):

        assert isinstance(self.low_test_data, torch.Tensor)
        assert isinstance(self.high_test_data, torch.Tensor)

    def test_high_res_size(self):

        assert self.high_test_data.shape == torch.Size(
            [self.dataset_size, self.image_size, self.image_size]
        )

    def test_low_res_size(self):

        assert self.low_test_data.shape == torch.Size(
            [self.dataset_size, self.image_size // 4, self.image_size // 4]
        )


class Test_DatasetObject(unittest.TestCase):
    """
    Test the DatasetObject class
    1) Check length of datasets via __len__ method
    2) Check __getitem__ method returns tuple of high/low res images
    """

    def setUp(self):
        self.dataset_size = 10
        self.image_size = 100
        self.lf = False
        self.valid = self.test = False

        self.low_test_data, self.high_test_data = data_processor(
            self.dataset_size, self.image_size, self.lf, self.valid, self.test
        )

        self.dataset = DatasetObject(self.low_test_data, self.high_test_data)

    def test_len(self):
        assert (
            self.dataset.__len__()
            == self.dataset_size
            == self.low_test_data.shape[0]
            == self.high_test_data.shape[0]
        )

    def test_getitem(self):
        low = self.dataset.__getitem__(0)[0]
        high = self.dataset.__getitem__(0)[1]

        assert low.shape == torch.Size([self.image_size // 4, self.image_size // 4])
        assert high.shape == torch.Size([self.image_size, self.image_size])


class Test_MyDataModule(unittest.TestCase):
    """
    I want to test the MyDataModule class from openlsrx.data.dataset

    This module builds upon the DatasetObject, and data_processor above
    """

    def setUp(self):
        self.dataset_size = 10
        self.valid_size = 2
        self.test_size = 3
        self.image_size = 100
        self.lf = True
        # self.valid = self.test = True
        self.batch_size = 1
        self.num_workers = 1
        self.pin_memory = False
        self.persistent_workers = 1

        self.data_module = MyDataModule(
            self.dataset_size,
            self.valid_size,
            self.test_size,
            self.image_size,
            self.lf,
            # self.valid,
            # self.test,
            self.batch_size,
            self.num_workers,
            self.pin_memory,
            self.persistent_workers,
        )

    def test_fit(self, stage="fit"):
        self.data_module.setup(stage)

        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()

        # check the dataloaders exist
        assert train_loader is not None
        assert val_loader is not None

        # check the outputs are tensors
        assert isinstance(train_loader.dataset.__getitem__(0)[0], torch.Tensor)
        assert isinstance(train_loader.dataset.__getitem__(0)[1], torch.Tensor)

        assert isinstance(val_loader.dataset.__getitem__(0)[0], torch.Tensor)
        assert isinstance(val_loader.dataset.__getitem__(0)[1], torch.Tensor)

    def test_test(self, stage="test"):

        self.data_module.setup(stage)
        test_loader = self.data_module.test_dataloader()

        assert test_loader is not None
        assert isinstance(test_loader.dataset.__getitem__(0)[0], torch.Tensor)
        assert isinstance(test_loader.dataset.__getitem__(0)[1], torch.Tensor)

    def test_predict(self, stage="predict"):

        self.data_module.setup(stage)
        predict_loader = self.data_module.predict_dataloader()

        assert predict_loader is not None
        assert isinstance(predict_loader.dataset.__getitem__(0)[0], torch.Tensor)
        assert isinstance(predict_loader.dataset.__getitem__(0)[1], torch.Tensor)


class Test_PoreSpy3D_Volume(unittest.TestCase):

    pass
