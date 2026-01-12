# import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from porespy_funcs import generator, float32_transform
from porespy_funcs_3d import image_transform
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import imageio
from matplotlib.colors import Normalize
import h5py
import numpy as np
import gc
import cv2
import os
from torchvision.transforms import Resize, InterpolationMode


def data_processor(dataset_size, image_size, lf=False, valid=False, test=False):
    """
    This function does all pre-processing for passing to DatasetObject class
    This is used for generating corresponding high/low pairs of synthetic
    PoreSpy data

    dataset_size = size of dataset
    image_size = size of images
    seed = random seed
    lf = Bool (low-feature or high_feature)

    Returns tuple of low-res, high_res tensor

    Output -> tuple(torch.Tensor([C,H,W],dtype=float32))
    """
    low_res = []
    high_res = []

    for i in range(0, dataset_size):

        if not valid:
            seed = i
        elif valid:
            seed = i + 10000

        elif test:
            seed = i + 50000

        high, low = generator(size=image_size, seed=seed, lf=lf)
        # convert to Tensors
        high = float32_transform(high)
        low = float32_transform(low)
        low_res += low
        high_res += high

        # self.low_res/high_res contain 16bit tensors of all images in dataset_size

    return torch.stack(low_res), torch.stack(high_res)


class DatasetObject(Dataset):
    """
    This class takes takes input as low-res and high-res arrays,
    then creates a Dataset object with them as high-res and low-res
    tensors
    Input: np.array
    Output: torch.Tensor([B,C,H,W])
    """

    def __init__(self, low_res, high_res):
        super(DatasetObject, self).__init__()
        # load low_res and high_res stacks
        self.low_res = low_res
        self.high_res = high_res

    def __len__(self):
        return len(self.high_res)

    def __getitem__(self, index):
        return self.low_res[index], self.high_res[index]


class MyDataModule(pl.LightningDataModule):
    """
    This defines my Data module, which builds upon the DatasetObject class.

    It defines all of the preprocessing required,
    then also builds the correct dataloaders

    1) Use data_processor() to generate high/low res images for test/train
    2) Use DatasetObject to create Dataset objects
    3) Define the train/test/validation dataloaders using DataLoader

    All of the global parameters for the dataset are defined below in the __init__

    1) data_processor() takes:
        dataset_size = size of dataset
        image_size = size of images
        lf = Bool (low-feature or high_feature)

    2) DatasetObject takes: high_res, low_res from above

    3) DataLoader takes:
        dataset object from above
        batch_size
        num_workers
        pin_memory
        persistent_workers
    """

    def __init__(
        self,
        dataset_size,
        valid_size,
        test_size,
        image_size,
        lf,
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
    ):
        super().__init__()

        self.dataset_size = dataset_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.image_size = image_size
        self.lf = lf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage):

        if stage == "fit":

            low_train, high_train = data_processor(
                dataset_size=self.dataset_size,
                image_size=self.image_size,
                lf=self.lf,
                valid=False,
            )

            self.train = DatasetObject(low_train, high_train)

            low_valid, high_valid = data_processor(
                dataset_size=self.valid_size,
                image_size=self.image_size,
                lf=self.lf,
                valid=True,
            )

            self.valid = DatasetObject(low_valid, high_valid)

        if stage == "test":

            low_test, high_test = data_processor(
                dataset_size=self.test_size,
                image_size=self.image_size,
                lf=self.lf,
                test=True,
            )
            self.test = DatasetObject(low_test, high_test)

        if stage == "predict":
            low_predict, high_predict = data_processor(
                dataset_size=self.test_size,
                image_size=self.image_size,
                lf=self.lf,
                test=True,
            )
            self.predict = DatasetObject(low_predict, high_predict)

    def train_dataloader(self):

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):

        return DataLoader(
            self.valid,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):

        return DataLoader(
            self.predict,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


class PoreSpy3D_Volume(pl.LightningDataModule):
    """
    The purpose of this class is to load the 3D PoreSpy data

    The data will be loaded, and processed into corresponding
    high and low-res pairs, and also loaded orthogonally

    Task list:
    1) Load the tiff files
    2) Create Dataset objetcs using DatasetObject class
    3) Create DataLoader objects ready to be passed to model

    """

    def __init__(
        self,
        high_file,
        low_file,
        low_ortho1_file,
        low_ortho2_file,
        dataset_size,
        image_size,
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
    ):

        super().__init__()

        self.high_file = high_file
        self.low_file = low_file
        self.low_ortho1_file = low_ortho1_file
        self.low_ortho2_file = low_ortho2_file

        self.dataset_size = dataset_size
        self.image_size = image_size

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage):
        if stage == "fit":
            # training 1st
            self.high_res = []
            self.low_res = []

            high = imageio.volread(self.high_file)
            low = imageio.volread(self.low_file)

            for i in range(self.dataset_size):
                high_slice = high[
                    i,
                    self.image_size : self.image_size * 2,
                    self.image_size : self.image_size * 2,
                ]
                low_slice = low[
                    i,
                    self.image_size // 4 : self.image_size // 4 * 2,
                    self.image_size // 4 : self.image_size // 4 * 2,
                ]

                high_res = image_transform(high_slice)
                low_res = image_transform(low_slice)

                self.high_res += high_res
                self.low_res += low_res

            self.train = DatasetObject(self.low_res, self.high_res)

            # validation

            high_valid = high
            low_valid = imageio.volread(self.low_ortho1_file)

            slices = [0, 12, 72, 160, 292, 372, 500, 624, 812, 920]

            self.high_valid = []
            self.low_valid = []

            for slice in slices:
                high_ortho = high_valid[0 : self.image_size, slice, 0 : self.image_size]
                low_ortho = low_valid[
                    0 : self.image_size // 4, slice, 0 : self.image_size // 4
                ]

                high_res = image_transform(high_ortho)
                low_res = image_transform(low_ortho)

                self.high_valid += high_res
                self.low_valid += low_res

            self.valid = DatasetObject(self.low_valid, self.high_valid)

        if stage == "test":
            slices = [0, 12, 72, 160, 292, 372, 500, 624, 812, 920]

            high_test = imageio.volread(self.high_file)
            low_test = imageio.volread(self.low_ortho2_file)

            self.high_test = []
            self.low_test = []

            for slice in slices:
                high_ortho2 = high_test[0 : self.image_size, 0 : self.image_size, slice]
                low_ortho2 = low_test[
                    0 : self.image_size // 4, 0 : self.image_size // 4, slice
                ]

                high_res = image_transform(high_ortho2)
                low_res = image_transform(low_ortho2)

                self.high_test += high_res
                self.low_test += low_res

            self.test = DatasetObject(self.low_test, self.high_test)

        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        pass


# I13 XCT functions


class CustomNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, clip=True):
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result = super().__call__(value, clip)
        return result


# defines the colormaps for the different datasets and resolutions

# norm26 = CustomNormalize(vmin=-5.41e-4, vmax=1.68e-3)
norm16 = CustomNormalize(vmin=-5.23e-4, vmax=1.11e-3)
norm81 = CustomNormalize(vmin=-2.50e-4, vmax=6.79e-4)


def norm_01(volume, lower_bound, upper_bound):
    """
    Takes a volume and normalises the volume between [0,1]
    This preserves the dynamic range of the images without clipping
    Specify the lower and upper bounds for the percentile
    """

    # vol_min, vol_max = np.min(volume), np.max(volume)
    vol_min, vol_max = np.percentile(volume, lower_bound), np.percentile(
        volume, upper_bound
    )

    return CustomNormalize(vmin=vol_min, vmax=vol_max)(volume)


def i13_xct_processor(low_file, high_file, dataset_size):
    """
    The function takes the low-res and high-res files as input, loads them
    and processes them into a torch.stack of low_res, high_res images that are
    aligned spatially.

    The output is passsed to create the I13XCTDatasetModule object

    Input: High and Low-res files
    Output: tuple(torch.stack(low_res), torch.stackk(high_res)

    Arguments:
    low_file = low-res filename
    high_file = high-res filename
    dataset_size = required size of dataset

    Workflow:
    1) Load the data
    2) Normalise each dataset between [0,1]
    3) Align the data and define the dataset size
    4) Apply colormap to the data
    """

    # first load the data

    low_data = np.transpose(
        h5py.File(low_file, "r")["4-TomopyRecon-tomo"]["data"], (1, 2, 0)
    )
    high_data = np.transpose(
        h5py.File(high_file, "r")["4-TomopyRecon-tomo"]["data"], (1, 2, 0)
    )

    # apply the normalisation
    # data is float32 here and is normalised [0,1]
    # also align the dataset
    # low_data[894][1010:1650,858:1498], high_data[804][763:2043,466:1746]

    low_data_norm = norm16(low_data[:, 1010:1650, 858:1498])
    # low_data_norm = norm_01(low_data[:, 1010:1650, 858:1498], 0.005, 99.995)
    del low_data

    high_data_norm = norm81(high_data[:, 763:2043, 466:1746])
    # high_data_norm = norm_01(high_data[:, 763:2043, 466:1746], 0.005, 99.995)
    del high_data
    gc.collect()

    # data is loaded, normalised and aligned
    # now define the dataset

    # define dataset sizes and image indices for alignment
    dataset_size = dataset_size
    offset = 50
    low_idx = 895 - offset
    high_idx = 801 - (offset * 2)

    low_volume = low_data_norm[low_idx : low_idx + dataset_size, :, :]
    del low_data_norm

    high_volume = high_data_norm[
        high_idx : high_idx + (dataset_size * 2),
        :,
        :,
    ]
    del high_data_norm
    gc.collect()

    # 3D np.array -> torch.Tensor([C,H,W])
    low_stack = torch.tensor(low_volume)
    high_stack = torch.tensor(high_volume)

    # I now have the volumes, normalised and aligned

    return low_stack, high_stack


def i13_xct_manalign_processor(
    low_file, high_file, dataset_size, xy=False, xz=False, yz=False
):
    """
    Defines the processing for the manually aligned I13 data
    """

    # first load the data

    low_data = np.transpose(
        h5py.File(low_file, "r")["4-TomopyRecon-tomo"]["data"], (1, 2, 0)
    )
    high_data = np.transpose(
        h5py.File(high_file, "r")["4-TomopyRecon-tomo"]["data"], (1, 2, 0)
    )

    # apply the normalisation
    # data is float32 here and is normalised [0,1]
    # also align the dataset
    # low_data[894][1010:1650,858:1498], high_data[804][763:2043,466:1746]

    low_data_norm = norm16(low_data[:, 1010:1650, 858:1498])
    # low_data_norm = norm_01(low_data[:, 1010:1650, 858:1498], 0.005, 99.995)
    del low_data

    high_data_norm = norm81(high_data[:, 763:2043, 466:1746])
    # high_data_norm = norm_01(high_data[:, 763:2043, 466:1746], 0.005, 99.995)
    del high_data
    gc.collect()

    # data is loaded, normalised and aligned
    # now define the dataset

    # define dataset sizes and image indices for alignment
    dataset_size = dataset_size
    offset = 50
    low_idx = 895 - offset
    high_idx = 801 - (offset * 2)

    low_volume = low_data_norm[low_idx : low_idx + dataset_size, :, :]
    del low_data_norm

    high_volume = high_data_norm[
        high_idx : high_idx + (dataset_size * 2),
        :,
        :,
    ]
    del high_data_norm
    gc.collect()

    # 3D np.array -> torch.Tensor([C,H,W])
    low_stack = torch.tensor(low_volume)
    high_stack = torch.tensor(high_volume)

    low_slices = [
        1,
        4,
        9,
        11,
        15,
        17,
        23,
        25,
        28,
        31,
        38,
        44,
        49,
        54,
        57,
        63,
        67,
        71,
        76,
        82,
        87,
        90,
        92,
        98,
        104,
        107,
        114,
        120,
        128,
        134,
        137,
        142,
        148,
        155,
        162,
        164,
        171,
        181,
        185,
        190,
        197,
        199,
        205,
        215,
        220,
        225,
        231,
        234,
        242,
        247,
        251,
        256,
        258,
        264,
        267,
        273,
        281,
        287,
        293,
        295,
        298,
        301,
        306,
        308,
        314,
        320,
        324,
        330,
        339,
        344,
        349,
        355,
        361,
        366,
        370,
        375,
        378,
        382,
        385,
        387,
        396,
        401,
        404,
        413,
        420,
        426,
        432,
        438,
        444,
        451,
        459,
        467,
        471,
        476,
        485,
        489,
        492,
        500,
        507,
        510,
        516,
        522,
        524,
        533,
        540,
        546,
        553,
        559,
        564,
        569,
        576,
        582,
        592,
        597,
        602,
        607,
        610,
        616,
        624,
        628,
    ]
    high_slices = [
        2,
        5,
        11,
        13,
        17,
        19,
        25,
        27,
        30,
        33,
        40,
        46,
        51,
        56,
        59,
        65,
        70,
        74,
        79,
        85,
        90,
        93,
        95,
        101,
        107,
        110,
        117,
        123,
        131,
        137,
        140,
        145,
        152,
        159,
        166,
        168,
        175,
        185,
        189,
        194,
        201,
        203,
        210,
        220,
        225,
        230,
        236,
        239,
        247,
        252,
        256,
        261,
        263,
        269,
        273,
        279,
        287,
        292,
        299,
        301,
        303,
        307,
        312,
        314,
        320,
        326,
        330,
        336,
        345,
        351,
        356,
        362,
        367,
        373,
        377,
        382,
        385,
        389,
        392,
        394,
        403,
        408,
        411,
        421,
        428,
        434,
        440,
        446,
        452,
        459,
        467,
        475,
        479,
        485,
        493,
        498,
        501,
        508,
        516,
        519,
        524,
        531,
        533,
        542,
        549,
        556,
        562,
        569,
        574,
        579,
        587,
        592,
        602,
        607,
        612,
        617,
        620,
        627,
        634,
        640,
    ]

    low_aligned_stack = low_stack[torch.tensor(np.array(low_slices) - 1)]
    high_aligned_stack = high_stack[torch.tensor((np.array(high_slices) - 1) * 2)]

    if yz is True:
        return low_aligned_stack, high_aligned_stack

    elif xy is True:
        return torch.permute(
            low_stack[low_slices[0] - 1 : low_slices[-1] - 1], (2, 0, 1)
        ), torch.permute(
            high_stack[high_slices[0] - 1 : (high_slices[-1] - 1) * 2], (2, 0, 1)
        )

    elif xz is True:
        return torch.permute(
            low_stack[low_slices[0] - 1 : low_slices[-1] - 1], (1, 0, 2)
        ), torch.permute(
            high_stack[high_slices[0] - 1 : (high_slices[-1] - 1) * 2], (1, 0, 2)
        )


def i13_test_data_processor(low_file, high_file):

    low_data = h5py.File(low_file, "r")["data"]
    high_data = h5py.File(high_file, "r")["data"]
    print(low_data.shape, high_data.shape)
    print(type(low_data), type(high_data))

    return torch.tensor(low_data), torch.tensor(high_data)


def i13_xct_pred_processor(low_file, high_file, xy=False, xz=False, yz=False):
    low_data = np.transpose(
        h5py.File(low_file, "r")["4-TomopyRecon-tomo"]["data"], (1, 2, 0)
    )
    high_data = np.transpose(
        h5py.File(high_file, "r")["4-TomopyRecon-tomo"]["data"], (1, 2, 0)
    )

    low_data_norm = norm16(low_data[:, 1010:1650, 858:1498])
    # low_data_norm = norm_01(low_data[:, 1010:1650, 858:1498], 0.005, 99.995)
    del low_data

    high_data_norm = norm81(high_data[:, 763:2043, 466:1746])
    # high_data_norm = norm_01(high_data[:, 763:2043, 466:1746], 0.005, 99.995)
    del high_data
    gc.collect()

    # define dataset sizes and image indices for alignment

    low_size = 640
    high_size = 1280

    low_idx = 895 - 50
    high_idx = 801 - (50 * 2)

    low_volume = low_data_norm[
        low_idx : (low_idx + low_size), :, :
    ]  # [640,640,640] here
    del low_data_norm

    high_volume = high_data_norm[
        high_idx : (high_idx + high_size), :, :
    ]  # [1280,1280,1280] here
    del high_data_norm
    gc.collect()

    # 3D np.array -> torch.Tensor([C,H,W])
    if xy is True:
        low_pred = torch.tensor(np.transpose(low_volume, (2, 0, 1)))
        high_pred = torch.tensor(np.transpose(high_volume, (2, 0, 1))[::2, :, :])

    if xz is True:
        low_pred = torch.tensor(np.transpose(low_volume, (1, 0, 2)))
        high_pred = torch.tensor(np.transpose(high_volume, (1, 0, 2))[::2, :, :])

    if yz is True:
        low_pred = torch.tensor(low_volume)
        high_pred = torch.tensor(high_volume[::2, :, :])

    # I now have the volumes, normalised and aligned

    return low_pred, high_pred


class I13XCTDataModule(pl.LightningDataModule):
    """
    This module defines the dataloader process for the I13 XCT data that I collected
    It makes use of the DatasetObject class to create the dataset objects.

    The I13 data will exist in the high-res and low-res 3D volume.

    Workflow:
    1) Data processing (refer to Jupyter notebook on aligning/colormap)
        - need to create a new one

    2) Create DatasetObject of the high/low res data
        - should work provided I feed same input as output of data_processor function

    3) Create DataLoader objects for training/validation/testing
        - follow process from above with the different test/train/valid datasets
    """

    def __init__(
        self,
        low_file,
        high_file,
        dataset_size,
        train_size,
        valid_size,
        test_size,
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        train_axis="yz",
        predict_axis="yz",
        testing=False,
        predict=False,
        manual_train=False,
    ):
        super().__init__()
        self.low_file = low_file
        self.high_file = high_file
        self.testing = testing
        self.dataset_size = dataset_size
        self.train_size = int(train_size * dataset_size)
        self.valid_size = int(valid_size * dataset_size)
        self.test_size = int(test_size * dataset_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_axis = train_axis
        self.predict_axis = predict_axis
        self.testing = testing
        self.predict = predict
        self.remainder = self.dataset_size - (
            self.train_size + self.valid_size + self.test_size
        )
        self.manual_train = manual_train

    def setup(self, stage):

        if not self.testing:
            low_stack, high_stack = i13_xct_processor(
                self.low_file, self.high_file, self.dataset_size
            )
        elif self.testing:
            low_stack, high_stack = i13_test_data_processor(
                self.low_file, self.high_file
            )

        # define the sampler
        # create train, test, valid datasets
        sampler = torch.Generator().manual_seed(42)

        if not self.manual_train:
            stack = DatasetObject(low_stack, high_stack[::2, :, :])
            train, valid, test, _ = random_split(
                stack,
                lengths=[
                    self.train_size,
                    self.valid_size,
                    self.test_size,
                    self.remainder,
                ],
                generator=sampler,
            )

        elif self.manual_train and self.train_axis != "mixed":
            low_stack, high_stack = i13_xct_manalign_processor(
                self.low_file,
                self.high_file,
                self.dataset_size,
                yz=True if self.train_axis == "yz" else False,
                xy=True if self.train_axis == "xy" else False,
                xz=True if self.train_axis == "xz" else False,
            )

            low_stack = low_stack
            high_stack = high_stack[::2, :, :]

            # ensure that each slice is the correct size and resize if necessary
            if (low_stack.shape[1], low_stack.shape[2]) != (640, 640):
                low_stack = Resize(
                    size=(640, 640), interpolation=InterpolationMode.BICUBIC
                )(low_stack)

            if (high_stack.shape[1], high_stack.shape[2]) != (1280, 1280):
                high_stack = Resize(
                    size=(1280, 1280), interpolation=InterpolationMode.BICUBIC
                )(high_stack)

            stack = DatasetObject(low_stack, high_stack)

            if self.train_axis == "yz":
                train, valid, test = random_split(
                    stack,
                    lengths=[100, 10, 10],
                    generator=sampler,
                )
            elif self.train_axis == "xy" or self.train_axis == "xz":
                remainder = stack.__len__() - (100 + 10 + 10)
                train, valid, test, _ = random_split(
                    stack,
                    lengths=[100, 10, 10, remainder],
                    generator=sampler,
                )

        elif self.manual_train and self.train_axis == "mixed":
            low_yz, high_yz = i13_xct_manalign_processor(
                self.low_file,
                self.high_file,
                self.dataset_size,
                yz=True,
            )
            low_xy, high_xy = i13_xct_manalign_processor(
                self.low_file,
                self.high_file,
                self.dataset_size,
                xy=True,
            )
            low_xz, high_xz = i13_xct_manalign_processor(
                self.low_file,
                self.high_file,
                self.dataset_size,
                xz=True,
            )

            train_stacks = []
            valid_stacks = []
            test_stacks = []

            # ensure that each slice is the correct size and resize if necessary
            for low_stack, high_stack in zip(
                [low_xy, low_xz, low_yz], [high_xy, high_xz, high_yz]
            ):
                if (low_stack.shape[1], low_stack.shape[2]) != (640, 640):
                    low_stack = Resize(
                        size=(640, 640), interpolation=InterpolationMode.BICUBIC
                    )(low_stack)

                if (high_stack.shape[1], high_stack.shape[2]) != (1280, 1280):
                    high_stack = Resize(
                        size=(1280, 1280), interpolation=InterpolationMode.BICUBIC
                    )(high_stack)

                # take every second high-res slice to match low-res
                ortho_stack = DatasetObject(low_stack, high_stack[::2, :, :])

                remainder = ortho_stack.__len__() - (35 + 5 + 5)
                train_stack, valid_stack, test_stack, _ = random_split(
                    ortho_stack,
                    lengths=[35, 5, 5, remainder],
                    generator=sampler,
                )

                train_stacks.append(train_stack)
                valid_stacks.append(valid_stack)
                test_stacks.append(test_stack)

            train = ConcatDataset([train_stack for train_stack in train_stacks])
            valid = ConcatDataset([valid_stack for valid_stack in valid_stacks])
            test = ConcatDataset([test_stack for test_stack in test_stacks])

        if stage == "fit":
            self.train = train
            self.valid = valid

        if stage == "test":
            self.test = test

        if stage == "predict":
            low_pred, high_pred = i13_xct_pred_processor(
                self.low_file,
                self.high_file,
                xy=True if self.predict_axis == "xy" else False,
                xz=True if self.predict_axis == "xz" else False,
                yz=True if self.predict_axis == "yz" else False,
            )
            self.predict = DatasetObject(low_pred, high_pred)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


def rotate_image(image, angle):  # 254 degrees
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def norm_01_img(image, upper, lower):
    """
    Normalises an image between [upper, lower]

    image: np.array (uint16)
    upper: float (same dtype as image!)
    lower: float (same dtype as image!)

    return: np.array (float32)
    """

    image = image.astype(np.float32)

    # norm_image = (image - lower) / (upper - lower)

    norm_image = CustomNormalize(vmin=lower, vmax=upper)(image)

    return norm_image


def ind_lime_processor(low_folder, high_folder, dataset_size):
    """
    This function defines the processing done to the IL data

    1) Load the data
    2) Process
    3) Return tuple of high/low-res arrays as tensors
    """

    # define the index of the aligned images
    low_idx = 254
    high_idx = 637

    # define sizes of high_res and low_res
    low_size = 486
    high_size = 972  # = low_size * 2

    # specify [upper, lower] for normalisation
    upper = float(45000)
    lower = float(7500)

    # get list of all files in the high and low-folder

    low_files = sorted(os.listdir(low_folder))
    high_files = sorted(os.listdir(high_folder))

    low_array = []
    high_array = []

    for i in range(dataset_size):
        # load high and low-res images
        low_file = os.path.join(low_folder, low_files[low_idx + (i * 5)])
        high_file = os.path.join(high_folder, high_files[high_idx + (i * 9)])

        low_img = imageio.imread_v2(low_file)
        high_img = imageio.imread_v2(high_file)

        low_rot = rotate_image(low_img, 254)

        low_croprot = low_rot[175:710, 175:710]
        low_croprot = cv2.resize(
            low_croprot, (low_size, low_size), interpolation=cv2.INTER_CUBIC
        )
        high_rot = high_img[274 : 274 + high_size, 289 : 289 + high_size]

        low_norm = norm_01_img(low_croprot, upper, lower)
        high_norm = norm_01_img(high_rot, upper, lower)

        low_array.append(low_norm)
        high_array.append(high_norm)

    low_stack = torch.tensor(np.array(low_array))  # [dataset_size, low_size, low_size]
    high_stack = torch.tensor(
        np.array(high_array)
    )  # [dataset_size, high_size, high_size]

    return low_stack, high_stack


def ind_lime_processor_stack(low_file, xy=False, xz=False, yz=False):
    """
    Loads the orthogonal views for the predict dataloader
    Only need low-res as do not have enough high-res to correspond to
    """

    low_size = 486

    # specify [upper, lower] for normalisation
    upper = float(45000)
    lower = float(7500)

    low_volume = imageio.volread(low_file)  # [905, 876, 876]

    low_xy = low_volume.transpose((2, 0, 1)) if xy else None  # [876, 905, 876]
    low_xz = low_volume.transpose((1, 0, 2)) if xz else None  # [876, 905, 876]
    low_yz = low_volume if yz else None  # [905, 876, 876]

    # crop into [535,535] slices along axis and resize

    if xy:
        resized_xy = []
        for i in range(180, 690, 1):
            img = cv2.resize(
                low_xy[i, 175:710, 175:710],
                (low_size, low_size),
                interpolation=cv2.INTER_CUBIC,
            )
            img = norm_01_img(img, upper, lower)
            resized_xy.append(img)
        return torch.tensor(np.array(resized_xy))  # [905, 486, 486]

    elif xz:
        resized_xz = []
        for i in range(180, 690, 1):
            xz_slice = cv2.resize(
                low_xz[i, 175:710, 175:710],
                (low_size, low_size),
                interpolation=cv2.INTER_CUBIC,
            )
            xz_slice = norm_01_img(xz_slice, upper, lower)
            resized_xz.append(xz_slice)

        return torch.tensor(np.array(resized_xz))

    elif yz:
        resized_yz = []
        for i in range(175, 710, 1):
            yz_slice = cv2.resize(
                low_yz[i, 175:710, 180:690],
                (low_size, low_size),
                interpolation=cv2.INTER_CUBIC,
            )
            yz_slice = norm_01_img(yz_slice, upper, lower)
            resized_yz.append(yz_slice)

        return torch.tensor(np.array(resized_yz))  # [876, 486, 486]


class IndLimeDataModule(pl.LightningDataModule):
    """
    This module defines the dataloader process for the Indiana Limestone data
    It makes use of the DatasetObject class to create the dataset objects.

    The high and low-res data exists as a series of 2D images

    Workflow:
    1) Data processing - align & pre-process
    2) Create DatasetObject of the high/low res data
        - should work provided I feed same input as output of data_processor function

    3) Create DataLoader objects for training/validation/testing
        - follow process from above with the different test/train/valid datasets
    """

    def __init__(
        self,
        low_folder,
        high_folder,
        dataset_size,
        train_size,
        valid_size,
        test_size,
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        testing=False,
        predict_stack="",
        predict_axis="xy",
    ):
        super().__init__()
        self.low_folder = low_folder
        self.high_folder = high_folder
        self.testing = testing
        self.dataset_size = dataset_size
        self.train_size = int(train_size * dataset_size)
        self.valid_size = int(valid_size * dataset_size)
        self.test_size = int(test_size * dataset_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.predict_stack = predict_stack
        self.predict_axis = predict_axis

    def setup(self, stage):
        """
        Here we load the data from the folders and create the DatasetObjects

        The DatasetObjects are then split into train, validation and test datasets
        """

        low_stack, high_stack = ind_lime_processor(
            self.low_folder, self.high_folder, self.dataset_size
        )

        stack = DatasetObject(low_stack, high_stack)
        sampler = torch.Generator().manual_seed(42)
        train, valid, test = random_split(
            stack,
            lengths=[self.train_size, self.valid_size, self.test_size],
            generator=sampler,
        )

        if stage == "fit":
            self.train = train
            self.valid = valid

        if stage == "test":
            self.test = test

        if stage == "predict":
            pred_stack = ind_lime_processor_stack(
                self.predict_stack,
                xy=True if self.predict_axis == "xy" else False,
                xz=True if self.predict_axis == "xz" else False,
                yz=True if self.predict_axis == "yz" else False,
            )

            pred_dataset = DatasetObject(pred_stack, pred_stack)  # repeat low_res twice

            self.predict = pred_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


def bent_sand_processor(low_folder, high_folder, dataset_size):
    """
    This function describes the data processing for the Bentheimer Sandstone
    dataset.

    Input: str -> low_folder, high_folder & int -> dataset_size
    Output: torch.Tensor([C,H,W]), torch.Tensor([C,H,W])
    """
    # defines index for alignment
    low_idx = 82
    high_idx = 121

    low_size = 468

    # specify [upper, lower] for normalisation
    upper = float(45000)
    lower = float(7500)

    # get list of all files in the high and low-folder
    low_files = sorted(os.listdir(low_folder))
    high_files = sorted(os.listdir(high_folder))

    # loop over files in folders and append to list
    low_array = []
    high_array = []

    for i in range(dataset_size):

        # load high and low-res images
        low_file = os.path.join(low_folder, low_files[low_idx + (i * 5)])
        high_file = os.path.join(high_folder, high_files[high_idx + (i * 9)])

        low_img = imageio.imread_v2(low_file)
        high_img = imageio.imread_v2(high_file)

        # rotate low-res
        low_rot = rotate_image(low_img, 254)

        # crop high and low-res images
        low_croprot = low_rot[160:685, 165:690]
        high_crop = high_img[289:1225, 274:1210]

        # resize low-res image1
        low_resize = cv2.resize(
            low_croprot, (low_size, low_size), interpolation=cv2.INTER_CUBIC
        )

        # normalise images
        low_norm = norm_01_img(low_resize, upper, lower)
        high_norm = norm_01_img(high_crop, upper, lower)

        # append to list
        low_array.append(low_norm)
        high_array.append(high_norm)

    # convert to torch.Tensor
    low_stack = torch.tensor(np.array(low_array))
    high_stack = torch.tensor(np.array(high_array))

    return low_stack, high_stack


def bent_sand_processor_stack(low_folder, xy=False, xz=False, yz=False):
    """
    This function loads the orthogonal views for the predict dataloader
    Only need low-res as do not have enough high-res to correspond to
    """

    low_size = 468

    # specify [upper, lower] for normalisation
    upper = float(45000)
    lower = float(7500)

    low_volume = imageio.volread(low_folder)

    low_xy = low_volume.transpose((2, 0, 1)) if xy else None
    low_xz = low_volume.transpose((1, 0, 2)) if xz else None
    low_yz = low_volume if yz else None

    # crop into [300,300] slices along axis and resize

    if xy:
        resized_xy = []
        for i in range(165, 690, 1):
            img = cv2.resize(
                low_xy[i, 190:715, 160:685],
                (low_size, low_size),
                interpolation=cv2.INTER_CUBIC,
            )
            img = norm_01_img(img, upper, lower)
            resized_xy.append(img)
        return torch.tensor(np.array(resized_xy))

    if xz:
        resized_xz = []
        for i in range(160, 685, 1):
            xz_slice = cv2.resize(
                low_xz[i, 190:715, 165:690],
                (low_size, low_size),
                interpolation=cv2.INTER_CUBIC,
            )
            xz_slice = norm_01_img(xz_slice, upper, lower)
            resized_xz.append(xz_slice)

        return torch.tensor(np.array(resized_xz))

    if yz:
        resized_yz = []
        for i in range(190, 715, 1):
            yz_slice = cv2.resize(
                low_yz[i, 160:685, 165:690],
                (low_size, low_size),
                interpolation=cv2.INTER_CUBIC,
            )
            yz_slice = norm_01_img(yz_slice, upper, lower)
            resized_yz.append(yz_slice)

        return torch.tensor(np.array(resized_yz))


class BentSandDataModule(pl.LightningDataModule):
    """
    This class defines the dataloader process for the Bentheimer Sandstone data.

    1) Load and process data using bent_sand_processor
    2) Create DatasetObject of the high/low res data
    3) Create DataLoader objects for training/validation/testing
    """

    def __init__(
        self,
        low_folder,
        high_folder,
        dataset_size,
        train_size,
        valid_size,
        test_size,
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        predict_stack="",
        predict_axis="xy",
    ):
        super().__init__()
        self.low_folder = low_folder
        self.high_folder = high_folder
        self.dataset_size = dataset_size
        self.train_size = int(train_size * dataset_size)
        self.valid_size = int(valid_size * dataset_size)
        self.test_size = int(test_size * dataset_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.predict_stack = predict_stack
        self.predict_axis = predict_axis

    def setup(self, stage):

        low_stack, high_stack = bent_sand_processor(
            self.low_folder, self.high_folder, self.dataset_size
        )

        stack = DatasetObject(low_stack, high_stack)
        generator = torch.Generator().manual_seed(42)
        train, valid, test = random_split(
            stack,
            lengths=[self.train_size, self.valid_size, self.test_size],
            generator=generator,
        )

        if stage == "fit":
            self.train = train
            self.valid = valid

        if stage == "test":
            self.test = test

        if stage == "predict":
            pred_stack = bent_sand_processor_stack(
                self.predict_stack,
                xy=True if self.predict_axis == "xy" else False,
                xz=True if self.predict_axis == "xz" else False,
                yz=True if self.predict_axis == "yz" else False,
            )
            pred_dataset = DatasetObject(pred_stack, pred_stack)

            self.predict = pred_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


def porespy_data_processor(high_file, low_xy_file, low_xz_file, low_yz_file):
    """
    This function defines the processing of the porespy data

    1) Load the high-res data, and all three low-res files
    2) Return tuple of [low_res, high_res] for each dimension

    Input: str -> high_file, low_xy_file, low_xz_file, low_yz_file
    Output: tuple(torch.Tensor([C,H,W]), torch.Tensor([C,H,W]))
    """

    # load the data
    high_data = imageio.volread(high_file)
    low_xy_data = imageio.volread(low_xz_file)
    low_xz_data = imageio.volread(low_xy_file)
    low_yz_data = imageio.volread(low_yz_file)

    high_xy_stack = torch.tensor(
        high_data[400:800, 400:800, :].transpose(2, 0, 1)
    ).type(torch.float32)
    print(high_xy_stack.size())

    high_xz_stack = torch.tensor(
        high_data[400:800, :, 400:800].transpose(1, 0, 2)
    ).type(torch.float32)
    print(high_xz_stack.size())

    high_yz_stack = torch.tensor(high_data[:, 400:800, 400:800]).type(torch.float32)
    print(high_yz_stack.size())

    low_xy_stack = torch.tensor(
        low_xy_data[100:200, 100:200, :].transpose(2, 0, 1)
    ).type(torch.float32)
    print(low_xy_stack.size())

    low_xz_stack = torch.tensor(
        low_xz_data[100:200, :, 100:200].transpose(1, 0, 2)
    ).type(torch.float32)
    print(low_xz_stack.size())

    low_yz_stack = torch.tensor(low_yz_data[:, 100:200, 100:200]).type(torch.float32)
    print(low_yz_stack.size())

    return [
        [low_xy_stack, high_xy_stack],
        [low_xz_stack, high_xz_stack],
        [low_yz_stack, high_yz_stack],
    ]


class PoreSpyDataModule(pl.LightningDataModule):
    """
    This is a data module for the porespy data

    1) Load the relevant PoreSpy data files
    2) Create DatasetObjects of the high/low-res data
    3) Create DataLoader objects for training/validation/testing
        a. I want testing to be parallel
        b. I want validation to be processed in batches of 3
            i. each batch has image from different axis
    """

    def __init__(
        self,
        high_file,
        low_xy_file,
        low_xz_file,
        low_yz_file,
        dataset_size,
        train_size,
        valid_size,
        test_size,
        test_axis,
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
    ):
        super().__init__()
        # this defines the parameters for the config.yaml file
        self.high_file = high_file
        self.low_xy_file = low_xy_file
        self.low_xz_file = low_xz_file
        self.low_yz_file = low_yz_file
        self.dataset_size = dataset_size
        self.train_size = int(train_size * dataset_size)
        self.valid_size = int(valid_size * dataset_size)
        self.test_size = int(test_size * dataset_size)
        self.test_axis = test_axis
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage):

        # load the data

        stacks = porespy_data_processor(
            self.high_file, self.low_xy_file, self.low_xz_file, self.low_yz_file
        )

        xy_low, xy_high = stacks[0]
        xz_low, xz_high = stacks[1]
        yz_low, yz_high = stacks[2]

        # define the sampler
        # create train, test, valid datasets
        generator = torch.Generator().manual_seed(42)
        training_stack = DatasetObject(yz_low, yz_high)
        training_length = training_stack.__len__()

        train, valid, _ = random_split(
            training_stack,
            lengths=[
                self.train_size,
                self.valid_size,
                training_length - self.train_size - self.valid_size,
            ],
            generator=generator,
        )

        # concatenate the perpendicular low-res
        # define the high-res that correspon to the low-res
        if self.test_axis == "perpendicular":
            low_perp = torch.cat([xy_low, xz_low])
            high_perp = torch.cat([xy_high, xz_high])

            test_stack = DatasetObject(low_perp, high_perp)
            testing_size = test_stack.__len__()

            test, _ = random_split(
                test_stack,
                lengths=[self.test_size, (testing_size - self.test_size)],
                generator=generator,
            )

        elif self.test_axis == "parallel":
            test_stack = DatasetObject(yz_low, yz_high)
            testing_size = test_stack.__len__()

            test, _ = random_split(
                test_stack,
                lengths=[self.test_size, (testing_size - self.test_size)],
                generator=generator,
            )

        if stage == "fit":
            self.train = train
            self.valid = valid

        if stage == "test":
            self.test = test

        if stage == "predict":
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        pass


def porespy_data_processor_2x(high_file, low_xy_file, low_xz_file, low_yz_file):
    """
    This function defines the processing of the porespy data

    1) Load the high-res data, and all three low-res files
    2) Return tuple of [low_res, high_res] for each dimension

    Input: str -> high_file, low_xy_file, low_xz_file, low_yz_file
    Output: tuple(torch.Tensor([C,H,W]), torch.Tensor([C,H,W]))
    """

    # load the data
    high_data = imageio.volread(high_file)
    low_xy_data = imageio.volread(low_xz_file)
    low_xz_data = imageio.volread(low_xy_file)
    low_yz_data = imageio.volread(low_yz_file)

    high_xy_stack = torch.tensor(
        high_data[400:800, 400:800, :].transpose(2, 0, 1)
    ).type(torch.float32)
    print(high_xy_stack.size())

    high_xz_stack = torch.tensor(
        high_data[400:800, :, 400:800].transpose(1, 0, 2)
    ).type(torch.float32)
    print(high_xz_stack.size())

    high_yz_stack = torch.tensor(high_data[:, 400:800, 400:800]).type(torch.float32)
    print(high_yz_stack.size())

    low_xy_stack = torch.tensor(
        low_xy_data[200:400, 200:400, :].transpose(2, 0, 1)
    ).type(torch.float32)
    print(low_xy_stack.size())

    low_xz_stack = torch.tensor(
        low_xz_data[200:400, :, 200:400].transpose(1, 0, 2)
    ).type(torch.float32)
    print(low_xz_stack.size())

    low_yz_stack = torch.tensor(low_yz_data[:, 200:400, 200:400]).type(torch.float32)
    print(low_yz_stack.size())

    return [
        [low_xy_stack, high_xy_stack],
        [low_xz_stack, high_xz_stack],
        [low_yz_stack, high_yz_stack],
    ]


class PoreSpyDataModule_2x(pl.LightningDataModule):
    """
    This is a data module for the porespy data

    1) Load the relevant PoreSpy data files
    2) Create DatasetObjects of the high/low-res data
    3) Create DataLoader objects for training/validation/testing
        a. I want testing to be parallel
        b. I want validation to be processed in batches of 3
            i. each batch has image from different axis
    """

    def __init__(
        self,
        high_file,
        low_xy_file,
        low_xz_file,
        low_yz_file,
        dataset_size,
        train_size,
        valid_size,
        test_size,
        test_axis,
        predict_axis,
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
    ):
        super().__init__()
        # this defines the parameters for the config.yaml file
        self.high_file = high_file
        self.low_xy_file = low_xy_file
        self.low_xz_file = low_xz_file
        self.low_yz_file = low_yz_file
        self.dataset_size = dataset_size
        self.train_size = int(train_size * dataset_size)
        self.valid_size = int(valid_size * dataset_size)
        self.test_size = int(test_size * dataset_size)
        self.test_axis = test_axis
        self.predict_axis = predict_axis
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage):

        # load the data

        stacks = porespy_data_processor_2x(
            self.high_file, self.low_xy_file, self.low_xz_file, self.low_yz_file
        )

        xy_low, xy_high = stacks[0]
        xz_low, xz_high = stacks[1]
        yz_low, yz_high = stacks[2]

        # define the sampler
        # create train, test, valid datasets
        generator = torch.Generator().manual_seed(42)
        training_stack = DatasetObject(yz_low, yz_high)
        training_length = training_stack.__len__()

        train, valid, _ = random_split(
            training_stack,
            lengths=[
                self.train_size,
                self.valid_size,
                training_length - self.train_size - self.valid_size,
            ],
            generator=generator,
        )

        # concatenate the perpendicular low-res
        # define the high-res that correspon to the low-res
        if self.test_axis == "perpendicular":
            low_perp = torch.cat([xy_low, xz_low])
            high_perp = torch.cat([xy_high, xz_high])

            test_stack = DatasetObject(low_perp, high_perp)
            testing_size = test_stack.__len__()

            test, _ = random_split(
                test_stack,
                lengths=[self.test_size, (testing_size - self.test_size)],
                generator=generator,
            )

        elif self.test_axis == "parallel":
            test_stack = DatasetObject(yz_low, yz_high)
            testing_size = test_stack.__len__()

            test, _ = random_split(
                test_stack,
                lengths=[self.test_size, (testing_size - self.test_size)],
                generator=generator,
            )

        if stage == "fit":
            self.train = train
            self.valid = valid

        if stage == "test":
            self.test = test

        if stage == "predict":
            if self.predict_axis == "xy":
                predict = DatasetObject(xy_low, xy_high)

            elif self.predict_axis == "xz":
                predict = DatasetObject(xz_low, xz_high)

            elif self.predict_axis == "yz":
                predict = DatasetObject(yz_low, yz_high)

            self.predict = predict

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=1,
            num_workers=1,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
