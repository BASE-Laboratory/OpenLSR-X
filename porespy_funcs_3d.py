import numpy as np
import porespy as ps
import cv2

import matplotlib as mpl
from silx.math import colormap
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2


def contrast(volume, value):
    """
    takes an input image and adds a value to it
    """
    shape = volume.shape
    flat_volume = volume.flatten()
    for i in range(len(flat_volume)):
        if 0 < flat_volume[i] < 1 - value - 0.05:
            flat_volume[i] += value
    reshaped_volume = np.reshape(flat_volume, shape)

    return reshaped_volume


def min_max_norm(volume):
    shape = volume.shape
    flat_volume = volume.flatten()
    #     im_min = np.percentile(flat_image, 2.5)
    #     im_max = np.percentile(flat_image, 97.5)
    vol_min = min(flat_volume)
    vol_max = max(flat_volume)

    for i in range(len(flat_volume)):
        flat_volume[i] = (flat_volume[i] - vol_min) / (vol_max - vol_min)

    reshape_volume = np.reshape(flat_volume, shape)

    return reshape_volume


def static_noise(volume):
    # this noise is for high-res image ONLY
    # loop over all slices in volume
    for i in range(volume.shape[-1]):
        image = volume[:, :, i]
        row, col = image.shape
        mean = 0
        var = 0.0001
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        volume[:, :, i] += gauss
    return volume


def variable_noise(volume):
    # this noise is for high-res image ONLY
    for i in range(volume.shape[-1]):
        image = volume[:, :, i]
        row, col = image.shape
        mean = 0
        var = 0.01 * (float(np.random.random()))
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        volume[:, :, i] += gauss
    return volume


def check_por(image):
    porosity = ps.metrics.porosity(image)
    print(porosity)


def resize(volume, xy=False, xz=False, yz=False):
    # takes high-res input such as 400x400x400
    # and also the dimension the downsample wants to be performed in
    vol_shape = volume.shape
    ortho_volumes = []
    if xy is True:
        downsampled_xy = np.zeros(
            shape=(vol_shape[0] // 4, vol_shape[1] // 4, vol_shape[2])
        )
        long = 2
        for i in range(volume.shape[long]):
            image = volume[:, :, i]
            shape = image.shape
            low = cv2.resize(
                image,
                dsize=(shape[1] // 4, shape[0] // 4),
                interpolation=cv2.INTER_LANCZOS4,
            )
            # low = np.transpose(low,(1,0))
            downsampled_xy[:, :, i] += low
        ortho_volumes += [downsampled_xy]

    if xz is True:
        downsampled_xz = np.zeros(
            shape=(vol_shape[0] // 4, vol_shape[1], vol_shape[2] // 4)
        )
        long = 1
        for i in range(volume.shape[long]):
            image = volume[:, i, :]
            shape = image.shape
            low = cv2.resize(
                image,
                dsize=(shape[1] // 4, shape[0] // 4),
                interpolation=cv2.INTER_LANCZOS4,
            )
            # low = np.transpose(low,(1,0))
            downsampled_xz[:, i, :] += low
        ortho_volumes += [downsampled_xz]

    if yz is True:
        downsampled_yz = np.zeros(
            shape=(vol_shape[0], vol_shape[1] // 4, vol_shape[2] // 4)
        )
        long = 0
        for i in range(volume.shape[long]):
            image = volume[i, :, :]
            shape = image.shape
            low = cv2.resize(
                image,
                dsize=(shape[1] // 4, shape[0] // 4),
                interpolation=cv2.INTER_LANCZOS4,
            )
            # low = np.transpose(low,(1,0))
            downsampled_yz[i, :, :] += low
        ortho_volumes += [downsampled_yz]

    return ortho_volumes


def ortho_resize(volume, x=False, y=False, z=False):
    # specify the long axis
    vol_shape = volume.shape
    # volume will have gone from [1000,400,400] to [1000,100,100], want [250,100,100]
    if x is True:
        dim_range = vol_shape[1]
        downsampled = np.zeros(shape=(vol_shape[0] // 4, vol_shape[1], vol_shape[2]))
        for i in range(dim_range):
            image = volume[:, :, i]
            shape = image.shape
            resized = cv2.resize(
                image, dsize=(shape[0] // 4, shape[1]), interpolation=cv2.INTER_LANCZOS4
            )
            downsampled[:, :, i] += resized

        return downsampled

    elif y is True:
        dim_range = vol_shape[2]
        downsampled = np.zeros(shape=(vol_shape[0], vol_shape[1] // 4, vol_shape[2]))
        for i in range(dim_range):
            image = volume[i, :, :]
            shape = image.shape
            resized = cv2.resize(
                image, dsize=(shape[0] // 4, shape[1]), interpolation=cv2.INTER_LANCZOS4
            )
            downsampled[i, :, :] += resized

        return downsampled

    elif z is True:
        dim_range = vol_shape[0]
        downsampled = np.zeros(shape=(vol_shape[0], vol_shape[1], vol_shape[2] // 4))
        print(downsampled.shape)
        for i in range(dim_range):
            image = volume[:, i, :]
            shape = image.shape
            resized = cv2.resize(
                image, dsize=(shape[1] // 4, shape[0]), interpolation=cv2.INTER_LANCZOS4
            )
            # resized = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
            downsampled[:, i, :] += resized

        return downsampled


def generator(size=1000, seed=1, lf=True):
    """
    takes high-res size as input - 1000 by default
    seed = 1 default
    """
    seed = np.random.seed(seed)
    if lf is True:
        im_lf = ps.generators.blobs(shape=size, blobiness=1, porosity=0.4)
        thk_lf = ps.filters.local_thickness(im_lf, mode="hybrid", sizes=25)
        thk_lf_norm = min_max_norm(thk_lf)
        thk_lf_static = static_noise(thk_lf_norm)
        thk_lf_stat_norm = min_max_norm(thk_lf_static)
        print(thk_lf_stat_norm.shape)

        # prepare low-res
        thk_lf_norm = contrast(thk_lf_norm, 0.3)
        thk_lf_noisy = variable_noise(thk_lf_norm)
        thk_lf_low = resize(thk_lf_noisy, xz=True, xy=True, yz=True)
        print(len(thk_lf_low))
        thk_lf_low_xz = min_max_norm(thk_lf_low[0])
        thk_lf_low_xy = min_max_norm(thk_lf_low[1])
        thk_lf_low_yz = min_max_norm(thk_lf_low[2])

        return thk_lf_stat_norm, [thk_lf_low_xz, thk_lf_low_xy, thk_lf_low_yz]

    elif lf is False:
        im1_hf = ps.generators.blobs(shape=size, blobiness=0.5, porosity=0.2)
        im2_hf = ps.generators.blobs(shape=size, blobiness=2.5, porosity=0.5)
        im3_hf = ~(~im1_hf * im2_hf)
        thk_hf = ps.filters.local_thickness(im3_hf, mode="hybrid")
        thk_hf_norm = min_max_norm(thk_hf)
        thk_hf_static = static_noise(thk_hf_norm)
        thk_hf_stat_norm = min_max_norm(thk_hf_static)

        # prepare low-res
        thk_hf_norm = contrast(thk_hf_norm, 0.3)
        thk_hf_noisy = variable_noise(thk_hf_norm)
        thk_hf_low = resize(thk_hf_noisy, xz=True, xy=True, yz=True)
        thk_hf_low_xz = min_max_norm(thk_hf_low[0])
        thk_hf_low_xy = min_max_norm(thk_hf_low[1])
        thk_hf_low_yz = min_max_norm(thk_hf_low[2])

        return thk_hf_stat_norm, [thk_hf_low_xz, thk_hf_low_xy, thk_hf_low_yz]


def float16_transform(image):
    # takes an image from generator
    greys = mpl.colormaps["gray"]
    colors = greys.resampled(100)(range(100))
    transform = A.Compose([ToTensorV2()])
    vmin = 0
    vmax = 1
    image_cmap = colormap.cmap(image, colors=colors, vmin=vmin, vmax=vmax)
    image_cmap = image_cmap[:, :, :3]  # RGB
    image_tensor = transform(image=image_cmap)["image"]
    image_gray = torchvision.transforms.functional.rgb_to_grayscale(image_tensor)
    # convert to 16bit precision here
    # image_gray16 = image_gray.to(torch.float16)

    return image_gray  # grayscale 16 bit tensor in CHW form


def image_transform(image):
    greys = mpl.colormaps["gray"]
    colors = greys.resampled(100)(range(100))
    transform = A.Compose([ToTensorV2()])
    vmin = 0
    vmax = 1
    image_cmap = colormap.cmap(image, colors=colors, vmin=vmin, vmax=vmax)
    image_cmap = image_cmap[:, :, :3]
    image_tensor = transform(image=image_cmap)["image"]
    image_gray = torchvision.transforms.functional.rgb_to_grayscale(image_tensor)

    return image_gray
