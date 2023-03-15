import rasterio
from rasterio.windows import Window
from rasterio.plot import show

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from utils import paths_train_reference_images

x_paths, y_paths = paths_train_reference_images(type='flood')

x_paths = [x for x in x_paths if 'Myanmar' in x]
y_paths = [y for y in y_paths if 'Myanmar' in y]

print(x_paths)
print(y_paths)

found = 0

with rasterio.open(x_paths[0]) as x_src, rasterio.open(y_paths[0]) as y_src:
    x_img = x_src.read()
    y_img = y_src.read()

    x_img[x_img == x_src.nodata] = 0.
    y_img[y_img == y_src.nodata] = 0.

    print(x_src.bounds)
    print(y_src.bounds)

    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    # ax[0].imshow(x_img[0])
    # ax[1].imshow(y_img[0])
    show(x_src)
    show(y_src)
    exit()

    window_size = (200, 200)
    stride = (200, 200)

    # iterate over the image array using a nested loop
    for i in tqdm(range(0, x_img.shape[1] - window_size[0], stride[0]), position=0):
        for j in tqdm(range(0, x_img.shape[2] - window_size[1], stride[1]), position=1, leave=False):
            # define the window using the Window class
            window = Window.from_slices(
                (i, i + window_size[0]), (j, j + window_size[1]))

            # read the data within the window
            x_data_window = x_src.read(window=window)
            y_data_window = y_src.read(window=window)

            # replace -999 with 0
            x_data_window[x_data_window == x_src.nodata] = 0
            # y_data_window[y_data_window == y_src.nodata] = 0

            if (np.sum(x_data_window) != 0) and (np.sum(x_data_window[:, 0, :]) != 0) and (np.sum(x_data_window[:, -1, :]) != 0) and (np.sum(x_data_window[:, :, 0]) != 0) and (np.sum(x_data_window[:, :, -1]) != 0):
                if np.sum(y_data_window) != 0:
                    # pass
                    fig, ax = plt.subplots(1, 2, squeeze=True, figsize=(8, 8))
                    ax[0].imshow(x_data_window[0])
                    ax[1].imshow(y_data_window[0])
                    plt.show()
                    found += 1

print(found)
