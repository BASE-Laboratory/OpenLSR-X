import numpy as np
import porespy as ps
import cv2

import matplotlib as mpl
from silx.math import colormap
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

ps._Settings.tqdm["disable"] = "True"


def contrast(image, value):
    """
    takes an input image and adds a value to it
    """
    shape = image.shape
    flat_image = image.flatten()
    for i in range(len(flat_image)):
        if 0 < flat_image[i] < 1 - value - 0.05:
            flat_image[i] += value
    reshaped_image = np.reshape(flat_image, shape)

    return reshaped_image


def min_max_norm(image):
    shape = image.shape
    flat_image = image.flatten()
    #     im_min = np.percentile(flat_image, 2.5)
    #     im_max = np.percentile(flat_image, 97.5)
    im_min = min(flat_image)
    im_max = max(flat_image)

    for i in range(len(flat_image)):
        flat_image[i] = (flat_image[i] - im_min) / (im_max - im_min)

    reshape_image = np.reshape(flat_image, shape)

    return reshape_image


def static_noise(image):
    # this noise is for high-res image ONLY
    row, col = image.shape
    mean = 0
    var = 0.0001
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


def variable_noise(image):
    # this noise is for high-res image ONLY
    random = float(np.random.random())
    row, col = image.shape
    mean = 0
    var = 0.01 * random
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    image = image + gauss

    s_vs_p = 0.5
    amount = 0.01 * random
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0

    return out


def check_por(image):
    porosity = ps.metrics.porosity(image)
    print(porosity)


def generator(size=1000, seed=1, lf=True):
    """
    takes high-res size as input - 1000 by default
    seed = 1 default
    """
    seed = np.random.seed(seed)
    if lf is True:
        im_lf = ps.generators.blobs(shape=[size, size], blobiness=1, porosity=0.4)
        thk_lf = ps.filters.local_thickness(im_lf, mode="hybrid", sizes=25)
        thk_lf_norm = min_max_norm(thk_lf)
        thk_lf_static = static_noise(thk_lf_norm)
        thk_lf_stat_norm = min_max_norm(thk_lf_static)

        # prepare low-res
        thk_lf_norm = contrast(thk_lf_norm, 0.3)
        thk_lf_noisy = variable_noise(thk_lf_norm)
        thk_lf_low = cv2.resize(
            thk_lf_noisy, dsize=(size // 4, size // 4), interpolation=cv2.INTER_LANCZOS4
        )
        thk_lf_low = min_max_norm(thk_lf_low)

        return thk_lf_stat_norm, thk_lf_low

    elif lf is False:
        im1_hf = ps.generators.blobs(shape=[size, size], blobiness=0.5, porosity=0.2)
        im2_hf = ps.generators.blobs(shape=[size, size], blobiness=2.5, porosity=0.5)
        im3_hf = ~(~im1_hf * im2_hf)
        thk_hf = ps.filters.local_thickness(im3_hf, mode="hybrid")
        thk_hf_norm = min_max_norm(thk_hf)
        thk_hf_static = static_noise(thk_hf_norm)
        thk_hf_stat_norm = min_max_norm(thk_hf_static)

        # prepare low-res
        thk_hf_norm = contrast(thk_hf_norm, 0.3)
        thk_hf_noisy = variable_noise(thk_hf_norm)
        thk_hf_low = cv2.resize(
            thk_hf_noisy,
            dsize=(size // 4, size // 4),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        thk_hf_low = min_max_norm(thk_hf_low)

        return thk_hf_stat_norm, thk_hf_low


def float32_transform(image):
    # takes an image from generator
    greys = mpl.colormaps["gray"]
    colors = greys.resampled(100)(range(100))
    transform = A.Compose([ToTensorV2()])
    vmin = 0
    vmax = 1
    image_cmap = colormap.cmap(
        image.astype(np.float32), colors=colors.astype(np.float32), vmin=vmin, vmax=vmax
    )
    image_cmap = image_cmap[:, :, :3]  # RGB
    image_tensor = transform(image=image_cmap)["image"]
    image_gray = torchvision.transforms.functional.rgb_to_grayscale(image_tensor)

    return image_gray  # grayscale 32 bit tensor in CHW form
