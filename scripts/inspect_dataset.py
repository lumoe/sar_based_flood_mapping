import rasterio

from utils import paths_train_reference_images
import matplotlib.pyplot as plt

x_paths, y_paths = paths_train_reference_images(type="flood")

for path in x_paths + y_paths:
    with rasterio.open(path) as src:
        # load and display image using matplotlib
        print(path)
        # print(src.crs)
        # Get pixel spacing
        print(src.res)
        # img_data = src.read(1)
        # # replace -999 with 0
        # img_data[img_data == src.nodata] = -1

        # print(img_data.shape)
        # print(src.nodata)
        # plt.imshow(img_data)
        # plt.show()
