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


def intersection_window(src1, src2):
    left = max(src1.bounds.left, src2.bounds.left)
    right = min(src1.bounds.right, src2.bounds.right)
    bottom = max(src1.bounds.bottom, src2.bounds.bottom)
    top = min(src1.bounds.top, src2.bounds.top)

    if left < right and bottom < top:
        src1_window = rasterio.windows.from_bounds(
            left, bottom, right, top, src1.transform)
        src2_window = rasterio.windows.from_bounds(
            left, bottom, right, top, src2.transform)

        src1_col_off = src1.index(left, top)[1]
        src1_row_off = src1.index(left, top)[0]

        src2_col_off = src2.index(left, top)[1]
        src2_row_off = src2.index(left, top)[0]

        return (src1_window, src2_window, src1_col_off, src1_row_off, src2_col_off, src2_row_off)
    else:
        return None


def plot_sample(sar_band, reference_band):

    # Create a figure and axis objects for side-by-side plots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plot the SAR patch on the left axis
    axes[0].imshow(sar_band, cmap='gray')
    axes[0].set_title('SAR Patch')

    # Plot the reference patch on the right axis
    axes[1].imshow(reference_band, cmap='gray')
    axes[1].set_title('Reference Patch')

    # Adjust layout and display the figure
    fig.tight_layout()
    plt.show()


with rasterio.open(x_paths[0]) as x_src, rasterio.open(y_paths[0]) as y_src:
    x_img = x_src.read()
    x_transform = x_src.transform

    y_img = y_src.read()
    y_transform = y_src.transform

    intersect_result = intersection_window(x_src, y_src)

    if intersect_result is None:
        print("No overlapping area found")
        exit()
    else:
        sar_window, reference_window, sar_col_off, sar_row_off, ref_col_off, ref_row_off = intersect_result

    x_img[x_img == x_src.nodata] = 0.
    y_img[y_img == y_src.nodata] = 0.

    # print(x_src.bounds)
    # print(y_src.bounds)

    # fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    # # ax[0].imshow(x_img[0])
    # # ax[1].imshow(y_img[0])
    # show(x_src)
    # show(y_src)
    # exit()

    window_size = (200, 200)
    stride = (200, 200)

    for row in range(0, int(sar_window.height) - window_size[0] + 1, stride[0]):
        for col in range(0, int(sar_window.width) - window_size[1] + 1, stride[1]):
            sar_patch = x_img[:, sar_row_off+row:sar_row_off+row +
                              window_size[0], sar_col_off+col:sar_col_off+col+window_size[1]]
            reference_patch = y_img[:, ref_row_off+row:ref_row_off +
                                    row+window_size[0], ref_col_off+col:ref_col_off+col+window_size[1]]
            # training_samples.append((sar_patch, reference_patch))

            # plot_sample(sar_patch[0], reference_patch[0])
            if (np.sum(sar_patch) != 0) and (np.sum(sar_patch[:, 0, :]) != 0) and (np.sum(sar_patch[:, -1, :]) != 0) and (np.sum(sar_patch[:, :, 0]) != 0) and (np.sum(sar_patch[:, :, -1]) != 0):
                if np.sum(reference_patch) != 0:
                    # pass
                    fig, ax = plt.subplots(1, 2, squeeze=True, figsize=(8, 8))
                    ax[0].imshow(sar_patch[0])
                    ax[1].imshow(reference_patch[0])
                    plt.show()
    #                 found += 1

            # training_samples.append((sar_patch, reference_patch))

    # # iterate over the image array using a nested loop
    # for i in tqdm(range(0, x_img.shape[1] - window_size[0], stride[0]), position=0):
    #     for j in tqdm(range(0, x_img.shape[2] - window_size[1], stride[1]), position=1, leave=False):
    #         # define the window using the Window class
    #         window = Window.from_slices(
    #             (i, i + window_size[0]), (j, j + window_size[1]))

    #         # read the data within the window
    #         x_data_window = x_src.read(window=window)
    #         y_data_window = y_src.read(window=window)

    #         # replace -999 with 0
    #         x_data_window[x_data_window == x_src.nodata] = 0
    #         # y_data_window[y_data_window == y_src.nodata] = 0


print(found)
