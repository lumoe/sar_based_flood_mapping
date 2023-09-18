from pprint import pprint

import rastervision

from utils import get_train_image_pairs, visualise_overlap
from rastervision.core.data import ClassConfig
from rastervision.pytorch_learner import (
    SemanticSegmentationSlidingWindowGeoDataset,
    SemanticSegmentationVisualizer,
)

from rastervision.core.data import (
    RasterioSource,
    RasterioCRSTransformer,
    Scene,
    SemanticSegmentationLabelSource,
)

import albumentations as A

import torch
from tqdm import tqdm


train_image_paths = get_train_image_pairs()

image_uri = train_image_paths["flood"]["Greece2018"]["images"]
label_uri = train_image_paths["flood"]["Greece2018"]["reference"]
# image_uri = label_uri

pprint(image_uri)
pprint(label_uri)

# image_uri = "data/train/Sentinel-1/floods/Greece2018/EQUI7_EU020M/E054N006T3/SIG0_20180301T043036__VV_D007_E054N006T3_EU020M_V1M1R1_S1BIWGRDH_TUWIEN.tif"
# label_uri = image_uri
# label_uri = "data/train/reference/floods/Greece2018/S2-GroundTruth_FloodExtent_Greece201802_repro.tif"

class_config = ClassConfig(
    names=["other", "flood"], colors=["lightgray", "darkred"], null_class="other"
)


# ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
#     class_config=class_config,
#     image_uri=image_uri,
#     label_raster_uri=label_uri,
#     # label_vector_default_class_id=class_config.get_class_id("building"),
#     image_raster_source_kw=dict(allow_streaming=True),
#     label_raster_source_kw=dict(allow_streaming=True),
#     size=200,
#     stride=200,
#     transform=A.Resize(256, 256),
#     to_pytorch=True,
#     normalize=True,
# )


rs_img = RasterioSource(label_uri)
# crs_tf_img = rs_img.crs_transformer
# crs_tf_label = RasterioCRSTransformer.from_uri(image_uri[0])
# crs_img = RasterioCRSTransformer.from_uri(label_uri[0])
# extent_pixel_img = rs_img.extent
# extent_map = crs_tf_img.pixel_to_map(extent_pixel_img)
# extent_pixel_label = crs_tf_label.map_to_pixel(extent_map)
rs_label = RasterioSource(image_uri)  # , extent=extent_pixel_label)


print(rs_img.crs_transformer)
print(rs_label.crs_transformer)

scene = Scene(
    id="my_scene",
    raster_source=rs_label,
    label_source=SemanticSegmentationLabelSource(rs_img, class_config),
)

ds = SemanticSegmentationSlidingWindowGeoDataset(
    scene,
    size=200,
    stride=200,
    to_pytorch=True,
    normalize=True,
    transform=A.Resize(128, 128),
)

for i in tqdm(range(0, len(ds), 1)):
    x, y = ds[i]
    # print(x.shape, y.shape)
    # x, y = ds[0]
    # print(x.shape, y.shape)
    if (
        (torch.sum(x) != 0)
        and (torch.sum(x[:, 0, :]) != 0)
        and (torch.sum(x[:, -1, :]) != 0)
        and (torch.sum(x[:, :, 0]) != 0)
        and (torch.sum(x[:, :, -1]) != 0)
        and (torch.sum(y) != 0)
    ):
        visualise_overlap(x.unsqueeze(0)[0][0], y.unsqueeze(0)[0])

        # channel_display_groups = {"SAR": (0,)}
        # viz = SemanticSegmentationVisualizer(
        #     class_names=class_config.names,
        #     class_colors=class_config.colors,
        #     channel_display_groups={"SAR": (0,)},
        # )
        # viz.plot_batch(
        #     x.unsqueeze(0),
        #     y.unsqueeze(0),
        #     show=True,
        # )
