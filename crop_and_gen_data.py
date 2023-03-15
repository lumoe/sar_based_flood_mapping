import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
from rasterio.plot import show

from rasterio.warp import calculate_default_transform, reproject, Resampling


from utils import paths_train_reference_images

x_paths, y_paths = paths_train_reference_images(type='flood')

found_samples = 0

# go through x_paths and y_paths and remove the ones that conatin "Myanmar2019" anywhere in the string
x_paths = [x for x in x_paths if 'Greece' in x]
y_paths = [y for y in y_paths if 'Greece' in y]


def get_reference_extent_and_window(ref_image_path):
    with rasterio.open(ref_image_path) as src:
        left, bottom, right, top = src.bounds
        width, height = src.width, src.height
        window = Window(col_off=0, row_off=0, width=width, height=height)
    return left, bottom, right, top, window


def crop_sar_to_reference(sar_image_path, extent):
    left, bottom, right, top, window = extent
    with rasterio.open(sar_image_path) as src:
        sar_transform = src.transform
        sar_crs = src.crs
        col_start, row_start = src.index(left, top)
        col_stop, row_stop = src.index(right, bottom)
        width, height = col_stop - col_start, row_stop - row_start
        sar_window = Window(col_off=row_start,
                            row_off=col_start, width=height, height=width)
        sar_data = src.read(1, window=sar_window)
        sar_transform = src.window_transform(sar_window)
    return sar_data, sar_transform, sar_crs, sar_window


reference_image_path = y_paths[0]
with rasterio.open(reference_image_path) as src:
    reference_data = src.read(1)
    reference_transform = src.transform
    reference_crs = src.crs

sar_image_path = x_paths[0]
extent = get_reference_extent_and_window(
    reference_image_path)
sar_data, sar_transform, sar_crs, sar_window = crop_sar_to_reference(
    sar_image_path, extent)


def plot_images(reference_data, reference_transform, sar_data, sar_transform):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    show(reference_data, transform=reference_transform, ax=ax[0])
    ax[0].set_title('Reference Image')

    show(sar_data, transform=sar_transform, ax=ax[1])
    ax[1].set_title('Cropped SAR Image')

    plt.show()


def matplot_overlapping_images(image1, image2, alpha=0.5):
    fig, ax = plt.subplots(
        1, 2, squeeze=True, figsize=(8, 8))
    ax[0].imshow(image1, alpha=alpha, cmap='viridis')
    ax[0].imshow(image2, alpha=alpha, cmap='jet')
    plt.show()


def plot_overlapping_images(image1, image2, transform1, transform2, alpha=0.5):
    fig, ax = plt.subplots(figsize=(10, 10))
    show(image1, transform=transform1, ax=ax, alpha=alpha, cmap='viridis')
    show(image2, transform=transform2, ax=ax, alpha=alpha, cmap='jet')
    plt.show()


plot_overlapping_images(
    reference_data, sar_data, reference_transform, sar_transform)

left, bottom, right, top, window = extent


def sliding_window(width, height, window_size=(200, 200), stride=(200, 200)):
    for row in range(0, height - window_size[1] + 1, stride[1]):
        for col in range(0, width - window_size[0] + 1, stride[0]):
            yield col, row, window_size[0], window_size[1]


def extract_samples(image_data, width, height, window_size=(200, 200), stride=(200, 200)):
    samples = []
    for col, row, win_width, win_height in sliding_window(width, height, window_size, stride):
        sample = image_data[row:row + win_height, col:col + win_width]
        samples.append(sample)
    return samples


def print_image_bounds(image_path, name):
    with rasterio.open(image_path) as src:
        left, bottom, right, top = src.bounds
    print(f"{name} image bounds: left={left}, bottom={bottom}, right={right}, top={top}")


print_image_bounds(reference_image_path, "Reference")
print_image_bounds(sar_image_path, "SAR")


width, height = window.width, window.height

ref_samples = extract_samples(
    reference_data, reference_data.shape[1], reference_data.shape[0])
sar_samples = extract_samples(sar_data, sar_data.shape[1], sar_data.shape[0])


for idx, sample in enumerate(zip(ref_samples, sar_samples)):
    # plot_images(sample[0], reference_transform, sample[1], sar_transform)
    plot_overlapping_images(
        sample[0], sample[1], reference_transform, sar_transform)
    # matplot_overlapping_images(sample[0], sample[1])
