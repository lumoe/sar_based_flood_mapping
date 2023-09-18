from config import config
from utils.utils import (
    paths_train_reference_images,
    paths_train_flood_images,
    get_crs_for_file,
)
import rastervision.core
from rastervision.core.data import ClassConfig
from rastervision.core.box import Box
from rastervision.pytorch_learner import (
    SemanticSegmentationSlidingWindowGeoDataset,
    SemanticSegmentationVisualizer,
    TransformType,
    SemanticSegmentationRandomWindowGeoDataset,
)

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import numpy as np

import albumentations as A

from pprint import pprint

from rastervision.core.data import (
    RasterioSource,
    ClassConfig,
    SemanticSegmentationLabelSource,
    RasterioCRSTransformer,
    RasterioSource,
    Scene,
)
from rastervision.pytorch_learner.dataset import (
    SemanticSegmentationSlidingWindowGeoDataset,
)
from rastervision.pytorch_learner.dataset.visualizer import (
    SemanticSegmentationVisualizer,
)

# print(f"RaserVision Version: {rastervision.core.__version__}")


def show_windows(img, windows, title=""):
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))
    ax.imshow(img)
    ax.axis("off")
    # draw windows on top of the image
    for w in windows:
        p = patches.Polygon(w.to_points(), color="r", linewidth=1, fill=False)
        ax.add_patch(p)
    ax.autoscale()
    ax.set_title(title)
    plt.show()


def visualise_overlap(sentinel_data, reference_data, chip_name=""):
    reference_data = reference_data.numpy()
    sentinel_data = sentinel_data.numpy()

    # Normalize data between 0 and 1
    reference_data = (reference_data - reference_data.min()) / (
        reference_data.max() - reference_data.min()
    )
    sentinel_data = (sentinel_data - sentinel_data.min()) / (
        sentinel_data.max() - sentinel_data.min()
    )

    # Visualize the reference and Sentinel images side by side and on top of each other with 0.5 opacity
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(reference_data, cmap="gray")
    ax1.set_title("Sentinel Image")

    ax2.imshow(sentinel_data)
    ax2.set_title("Reference Image")

    ax3.imshow(reference_data, cmap="gray", alpha=0.5)
    ax3.imshow(sentinel_data, alpha=0.5)
    ax3.set_title("Overlay")

    for ax in [ax1, ax2, ax3]:
        ax.axis("off")

    plt.suptitle(chip_name)
    plt.show()


# x_paths, y_paths = paths_train_reference_images(type='flood')
x_paths, y_paths = paths_train_flood_images("Texas2017")
class_config = ClassConfig(
    names=["other", "flood"], colors=["lightgray", "darkred"], null_class="other"
)

found_samples = 0


pprint(x_paths)
pprint(y_paths)

x_crs = [get_crs_for_file(x) for x in x_paths]
y_crs = [get_crs_for_file(y) for y in y_paths]

# Verify that all elements in x_crs are the same as in y_crs
assert all(x == y for x, y in zip(x_crs, y_crs))


# Correct for the fact that the labels have a different extent than the imagery which
# RV does not handle automatically.
rs_img = RasterioSource(y_paths)
crs_tf_img = rs_img.crs_transformer
crs_tf_label = RasterioCRSTransformer.from_uri(x_paths[0])
crs_img = RasterioCRSTransformer.from_uri(y_paths[0])
extent_pixel_img = rs_img.extent
extent_map = crs_tf_img.pixel_to_map(extent_pixel_img)
extent_pixel_label = crs_tf_label.map_to_pixel(extent_map)
rs_label = RasterioSource(x_paths)  # , extent=extent_pixel_label)

print(rs_img.extent)
print(rs_label.extent)

scene = Scene(
    id="my_scene",
    raster_source=rs_img,
    label_source=SemanticSegmentationLabelSource(rs_label, class_config),
)

ds = SemanticSegmentationSlidingWindowGeoDataset(
    scene,
    size=600,
    stride=600,
    to_pytorch=True,
    normalize=True,
    transform=A.Resize(128, 128),
)

# for x, y in zip(x_paths, y_paths):
found_inside = 0
# https://docs.rastervision.io/en/0.20/api_reference/_generated/rastervision.core.data.raster_source.rasterio_source_config.RasterioSourceConfig.html
# ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
#     class_config=class_config,
#     image_uri=x_paths,
#     label_raster_uri=y_paths,
#     # label_vector_default_class_id=class_config.get_class_id('water'),
#     image_raster_source_kw=dict(allow_streaming=True),
#     label_raster_source_kw=dict(allow_streaming=True),

#     size=500,
#     stride=100,
#     transform=A.Resize(128, 128),

#     to_pytorch=True,
#     normalize=True,

#     # Random window settings
#     # out_size=256,
#     # allow windows to overflow the extent by 100 pixels
#     # padding=100,
#     # size_lims=(200, 300),

# )

# print(len(ds))
for i in tqdm(range(0, len(ds), 1)):
    # for i in tqdm(range(62509, len(ds), 1)):
    # for i in tqdm(range(350_000, len(ds), 2)):
    x, y = ds[i]
    # Convert x, y to torch tensors

    # x = torch.from_numpy(x)
    # y = torch.from_numpy(y.astype(np.uint8))
    # print(x.shape, y.shape)

    # # Show x, y as images to check if they are correct in one plot
    # fig, ax = plt.subplots(1, 2, squeeze=True, figsize=(8, 8))
    # ax[0].imshow(x)
    # ax[1].imshow(y)
    # plt.show()

    # Check if x is all zeroes using torch, check if first or last column is all zeroes or if first or last row is all zeroes
    if (
        (torch.sum(x) != 0)
        and (torch.sum(x[:, 0, :]) != 0)
        and (torch.sum(x[:, -1, :]) != 0)
        and (torch.sum(x[:, :, 0]) != 0)
        and (torch.sum(x[:, :, -1]) != 0)
    ):
        # if (torch.sum(x) != 0):
        found_samples += 1
        # if found_inside < 5 and torch.sum(y) != 0:
        channel_display_groups = {"SAR": (0,)}
        if torch.sum(y) != 0:
            # viz = SemanticSegmentationVisualizer(
            #     class_names=class_config.names, class_colors=class_config.colors, channel_display_groups=channel_display_groups)
            # viz.plot_batch(x.unsqueeze(0), y.unsqueeze(0), show=True)
            visualise_overlap(x.unsqueeze(0)[0][0], y.unsqueeze(0)[0])

            # check with torch if x is only zeros
            print(i if torch.sum(x) != 0 else "", end=",")
            found_inside += 1

# break
print()
print(found_samples)
print(found_inside)
# img_full = ds.scene.raster_source[:, :]
# show_windows(img_full, ds.windows, title='Sliding windows')
