import numpy as np
import porespy as ps
from pathlib import Path

# pip install tifffile
import tifffile as tiff

# --------------------
# Settings
# --------------------
out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)

shape = (500, 500, 500)   # (z, y, x)
target_porosity = 0.5
blobiness = 1
seed = 0

# Parallel/chunk settings (tune cores for your CPU)
sigma = np.mean(shape) / (40 * blobiness)
overlap = int(3 * sigma)   # helps avoid seams between chunks

parallel_kw = {
    "divs": 2,       # try 2 first; 3 if you have lots of cores
    "cores": 16,     # set to your physical cores (or None for all)
    "overlap": overlap,
}

# --------------------
# Generate greyscale blobs
# --------------------
im = ps.generators.blobs(
    shape=shape,
    porosity=None,       # keep greyscale (NOT boolean)
    blobiness=blobiness,
    seed=seed,
    periodic=True,
    parallel_kw=parallel_kw,
).astype(np.float32, copy=False)

# --------------------
# Make a 0.5-porosity pore mask (optional)
# --------------------
thr = np.quantile(im, target_porosity)
pores = im < thr  # pores.mean() ~= 0.5

print("Greyscale:", im.shape, im.dtype, float(im.min()), float(im.max()))
print("Porosity:", float(pores.mean()))

# --------------------
# Save as a single concatenated TIFF stack (BigTIFF)
# --------------------
# NOTE: 1000^3 float32 is ~4 GB, so BigTIFF is important.
out_gray = out_dir / "blobs_1000_float32.tif"
tiff.imwrite(
    out_gray,
    im,                       # 3D array -> multi-page TIFF
    dtype=np.float32,
    bigtiff=True,
    photometric="minisblack",
)

# Optional: also save the pore mask as uint8 (0/255) for easy viewing
out_mask = out_dir / "pores_porosity0p5_uint8.tif"
tiff.imwrite(
    out_mask,
    (pores.astype(np.uint8) * 255),
    dtype=np.uint8,
    bigtiff=True,
    photometric="minisblack",
)

print("Saved:", out_gray)
print("Saved:", out_mask)
