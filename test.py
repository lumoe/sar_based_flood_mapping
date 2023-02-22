from config import config
from utils.utils import paths_train_reference_images

from rastervision.core.data import ClassConfig
from rastervision.core.box import Box
from rastervision.pytorch_learner import (
    SemanticSegmentationSlidingWindowGeoDataset, SemanticSegmentationVisualizer, TransformType)

import torch
import numpy
from matplotlib import pyplot as plt
from tqdm import tqdm

import numpy as np

import albumentations as A

import inspect


DEBUG = any(True for frame in inspect.stack()
            if frame[1].endswith('pydevd.py'))

print(DEBUG)


def show_windows(img, windows, title=''):
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))
    ax.imshow(img)
    ax.axis('off')
    # draw windows on top of the image
    for w in windows:
        p = patches.Polygon(w.to_points(), color='r', linewidth=1, fill=False)
        ax.add_patch(p)
    ax.autoscale()
    ax.set_title(title)
    plt.show()


x_paths, y_paths = paths_train_reference_images(type='flood')

class_config = ClassConfig(
    names=['other', 'water'],
    colors=['lightgray', 'darkred'],
    null_class='other'
)

found_samples = 0

for x, y in zip(x_paths, y_paths):
    found_inside = 0
    ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
        class_config=class_config,
        image_uri=x_paths,
        label_raster_uri=y_paths,
        # label_vector_default_class_id=class_config.get_class_id('water'),
        image_raster_source_kw=dict(allow_streaming=True),
        label_raster_source_kw=dict(allow_streaming=True),
        size=200,
        stride=200,
        to_pytorch=True,
        normalize=True,
        transform=A.Resize(128, 128),

    )

    # print(len(ds))
    for i in tqdm(range(0, len(ds))):
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
        if (torch.sum(x) != 0) and (torch.sum(x[:, 0, :]) != 0) and (torch.sum(x[:, -1, :]) != 0) and (torch.sum(x[:, :, 0]) != 0) and (torch.sum(x[:, :, -1]) != 0):
            found_samples += 1
            if found_inside < 5 and torch.sum(y) != 0:
                channel_display_groups = {'SAR': (0,)}

                viz = SemanticSegmentationVisualizer(
                    class_names=class_config.names, class_colors=class_config.colors, channel_display_groups=channel_display_groups)
                viz.plot_batch(x.unsqueeze(0), y.unsqueeze(0), show=True)

                # check with torch if x is only zeros
                print(i if torch.sum(x) != 0 else '', end="")
                found_inside += 1

    # break
print(found_samples)
# img_full = ds.scene.raster_source[:, :]
# show_windows(img_full, ds.windows, title='Sliding windows')