import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
from rasterio.plot import show
import tempfile
import os


from rasterio.warp import calculate_default_transform, reproject, Resampling


from utils import paths_train_reference_images

x_paths, y_paths = paths_train_reference_images(type='flood')

found_samples = 0

target_train = 'Texas'

target_path = os.path.join('data', 'train', 'flood', target_train)

os.makedirs(target_path, exist_ok=True)

# go through x_paths and y_paths and remove the ones that conatin "Myanmar2019" anywhere in the string
x_paths = [x for x in x_paths if target_train in x]
y_paths = [y for y in y_paths if target_train in y]

print(x_paths[0])
print(y_paths[0])


def reproject_sar_to_reference_crs(sar_image_path, reference_crs):
    with rasterio.open(sar_image_path) as src:
        src_crs = src.crs
        if src_crs != reference_crs:
            transform, width, height = calculate_default_transform(
                src_crs, reference_crs, src.width, src.height, *src.bounds)
            sar_data = np.empty((height, width), dtype=src.meta['dtype'])
            reproject(
                source=rasterio.band(src, 1),
                destination=sar_data,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=reference_crs,
                resampling=Resampling.nearest)
            sar_transform = transform
        else:
            sar_data = src.read(1)
            sar_transform = src.transform
    return sar_data, sar_transform, reference_crs


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
        col_start, row_start = map(round, src.index(left, top))
        col_stop, row_stop = map(round, src.index(right, bottom))
        width, height = col_stop - col_start, row_stop - row_start
        sar_window = Window(col_off=row_start,
                            row_off=col_start, width=height, height=width)
        sar_data = src.read(1, window=sar_window)
        sar_transform = src.window_transform(sar_window)
        profile = src.profile
        profile.update({
            'width': height,
            'height': width,
            'transform': sar_transform,
        })

        temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        temp_file.close()

        with rasterio.open(temp_file.name, 'w', **profile) as sar_img:
            sar_img.write(sar_data, 1)

        cropped_sar_image = rasterio.open(temp_file.name)

    return cropped_sar_image
    # return sar_data, sar_transform, sar_crs, sar_window

# def crop_sar_to_reference(sar_data, sar_transform, sar_crs, extent):
#     left, bottom, right, top, window = extent
#     sar_crs = src.crs
#     with rasterio.open(sar_image_path) as src:
#         sar_transform = src.transform
#         col_start, row_start = map(round, src.index(left, top))
#         col_stop, row_stop = map(round, src.index(right, bottom))
#         width, height = col_stop - col_start, row_stop - row_start
#         sar_window = Window(col_off=row_start,
#                             row_off=col_start, width=height, height=width)
#         # sar_window = Window(col_off=col_start,
#         #                     row_off=row_start, width=width, height=height)
#         sar_data = src.read(1, window=sar_window)
#         sar_transform = src.window_transform(sar_window)
#     return sar_data, sar_transform, sar_crs, sar_window


reference_image_path = y_paths[0]
ref_src = rasterio.open(reference_image_path)
reference_data = ref_src.read(1)
reference_transform = ref_src.transform
reference_crs = ref_src.crs

sar_image_path = x_paths[0]

sar_data, sar_transform, sar_crs = reproject_sar_to_reference_crs(
    sar_image_path, reference_crs)

extent = get_reference_extent_and_window(
    reference_image_path)
# sar_data, sar_transform, sar_crs, sar_window = crop_sar_to_reference(
#     sar_image_path, extent)
# sar_data, sar_transform, sar_crs, sar_window = crop_sar_to_reference(
#     sar_image_path, extent)

cropped_sar_image = crop_sar_to_reference(sar_image_path, extent)
sar_data = cropped_sar_image.read(1)
sar_transform = cropped_sar_image.transform
sar_crs = cropped_sar_image.crs


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


# def extract_samples(image_data, width, height, window_size=(200, 200), stride=(200, 200)):
#     samples = []
#     for col, row, win_width, win_height in sliding_window(width, height, window_size, stride):
#         sample = image_data[row:row + win_width, col:col + win_height]
#         samples.append(sample)
#     return samples

def extract_samples(src, width, height, window_size=(200, 200), stride=(50, 50)):
    samples = []
    # width, height = src.width, src.height

    for col, row, win_width, win_height in sliding_window(width, height, window_size, stride):
        window = Window(col_off=col, row_off=row,
                        width=win_width, height=win_height)
        sample_data = src.read(1, window=window)
        sample_transform = src.window_transform(window)
        samples.append((sample_data, sample_transform))

    return samples


def print_image_bounds(image_path, name):
    with rasterio.open(image_path) as src:
        left, bottom, right, top = src.bounds
    print(f"{name} image bounds: left={left}, bottom={bottom}, right={right}, top={top}")


print_image_bounds(reference_image_path, "Reference")
print_image_bounds(sar_image_path, "SAR")


width, height = window.width, window.height

ref_samples = extract_samples(
    ref_src, width=ref_src.width, height=ref_src.height)
sar_samples = extract_samples(
    cropped_sar_image, width=ref_src.width, height=ref_src.height)


def plot_samples(ref_samples, sar_samples, alpha=0.8):
    n_samples = len(ref_samples)

    for i, (ref_sample, sar_sample) in enumerate(zip(ref_samples, sar_samples)):
        fig, ax = plt.subplots(figsize=(8, 8))

        ref_data, ref_transform = ref_sample
        sar_data, sar_transform = sar_sample

        show(ref_data, transform=ref_transform, ax=ax, alpha=alpha,
             cmap='gray', title=f'Reference Sample {i + 1}')
        show(sar_data, transform=sar_transform, ax=ax,
             alpha=alpha, cmap='jet', title=f'SAR Sample {i + 1}')

        plt.tight_layout()
        plt.show()
        plt.close(fig)

        break


def save_samples(ref_samples, sar_samples, sar_profile, ref_profile):
    for i, (ref_sample, sar_sample) in enumerate(zip(ref_samples, sar_samples)):
        ref_data, ref_transform = ref_sample
        sar_data, sar_transform = sar_sample

        # Save both samples as geotiffs to disk
        with rasterio.open(os.path.join(target_path, f"{i}_sar.tif"), 'w', **sar_profile) as writer:
            writer.write(sar_data, 1)

        with rasterio.open(os.path.join(target_path, f"{i}_ref.tif"), 'w', **ref_profile) as writer:
            writer.write(ref_data, 1)

        break


plot_samples(ref_samples, sar_samples)
# save_samples(ref_samples, sar_samples, cropped_sar_image.meta, ref_src.meta)


def create_chips(raster_array, raster_meta, window_size, output_dir, image_type):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rasterio.open('temp.tif', 'w', **raster_meta) as temp:
        temp.write(raster_array)

    with rasterio.open('temp.tif') as src:
        ncols, nrows = src.meta['width'], src.meta['height']
        offsets = product(range(0, ncols, window_size),
                          range(0, nrows, window_size))
        for idx, (col_off, row_off) in enumerate(offsets):
            window = Window(
                col_off=col_off, row_off=row_off, width=window_size, height=window_size)
            transform = src.window_transform(window)
            chip = src.read(window=window)
            chip_meta = src.meta.copy()
            chip_meta.update({
                'transform': transform,
                'width': window.width,
                'height': window.height
            })

            with rasterio.open(os.path.join(output_dir, f'{image_type}_{idx}.tif'), 'w', **chip_meta) as dst:
                dst.write(chip)

            break

    os.remove('temp.tif')


create_chips(reference_data, ref_src.meta, 200, target_path, 'ref')
create_chips(sar_data, cropped_sar_image.meta, 200, target_path, 'sar')

# for idx, sample in enumerate(zip(ref_samples, sar_samples)):
#     # plot_images(sample[0], reference_transform, sample[1], sar_transform)
#     plot_overlapping_images(
#         sample[0], sample[1], reference_transform, sar_transform)
#     # matplot_overlapping_images(sample[0], sample[1])
