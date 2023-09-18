import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def visualize_matching_chips(reference_folder, sentinel_folder, chip_name):
    reference_path = os.path.join(reference_folder, chip_name)
    sentinel_path = os.path.join(sentinel_folder, chip_name)

    with rasterio.open(reference_path) as ref_src:
        reference_data = ref_src.read()

    with rasterio.open(sentinel_path) as sen_src:
        sentinel_data = sen_src.read()

    # Replace non-data with 0
    reference_data[reference_data == -9999] = 0
    sentinel_data[sentinel_data == -9999] = 0

    # Normalize the data for better visualization
<<<<<<< HEAD
    reference_data = (reference_data - np.min(reference_data)) / \
        (np.max(reference_data) - np.min(reference_data))
    sentinel_data = (sentinel_data - np.min(sentinel_data)) / \
        (np.max(sentinel_data) - np.min(sentinel_data))
=======
    reference_data = (reference_data - np.min(reference_data)) / (
        np.max(reference_data) - np.min(reference_data)
    )
    sentinel_data = (sentinel_data - np.min(sentinel_data)) / (
        np.max(sentinel_data) - np.min(sentinel_data)
    )
>>>>>>> master

    # Visualize the reference and Sentinel images side by side and on top of each other with 0.5 opacity
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

<<<<<<< HEAD
    ax1.imshow(reference_data[0], cmap='gray')
=======
    ax1.imshow(reference_data[0], cmap="gray")
>>>>>>> master
    ax1.set_title("Reference Image")

    ax2.imshow(sentinel_data[0])
    ax2.set_title("Sentinel Image")

<<<<<<< HEAD
    ax3.imshow(reference_data[0], cmap='gray', alpha=0.5)
=======
    ax3.imshow(reference_data[0], cmap="gray", alpha=0.5)
>>>>>>> master
    ax3.imshow(sentinel_data[0], alpha=0.5)
    ax3.set_title("Overlay")

    for ax in [ax1, ax2, ax3]:
<<<<<<< HEAD
        ax.axis('off')
=======
        ax.axis("off")
>>>>>>> master

    plt.suptitle(chip_name)
    plt.show()


<<<<<<< HEAD
target_path = os.path.join('data', 'tmp')
output_path_gt = os.path.join(target_path, 'gt')
output_path_sentinel = os.path.join(target_path, 'sentinel')
=======
target_path = os.path.join("data", "tmp")
output_path_gt = os.path.join(target_path, "gt")
output_path_sentinel = os.path.join(target_path, "sentinel")
>>>>>>> master

# Example chip name

range_it = range(0, 1000, 1000)

for x in range_it:
    for y in range_it:
        chip_name = f"{x}_{y}.tif"
<<<<<<< HEAD
        visualize_matching_chips(
            output_path_gt, output_path_sentinel, chip_name)
=======
        visualize_matching_chips(output_path_gt, output_path_sentinel, chip_name)
>>>>>>> master

# chip_name = "1100_4700.tif"

# visualize_matching_chips(output_path_gt, output_path_sentinel, chip_name)
