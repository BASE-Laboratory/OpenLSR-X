import numpy as np
import porespy as ps
import tifffile as tf


def main():
    volume = ps.generators.blobs(
        shape=[1000, 1000, 1000],
        blobiness=1,
        porosity=0.4,
    ).astype(np.float32)
    tf.imwrite("data/porespy_1000x1000x1000_f32.tiff", volume)


if __name__ == "__main__":
    main()
