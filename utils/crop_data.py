import os
import rasterio
import numpy as np
from PIL import Image
from sys import stdout


class Printer:
    """Print things to stdout on one line dynamically"""

    def __init__(self, data):
        stdout.write("\r\x1b[K" + data.__str__())
        stdout.flush()


def get_windows(crop_size, img_size, sliding_len):
    cropped_windows = []
    step_size = crop_size
    iterator_col = 0
    iterator_row = 0
    for col in range(0, img_size, step_size):
        if iterator_col == sliding_len:
            col = img_size - crop_size
        else:
            iterator_col = iterator_col + 1
        for row in range(0, img_size, step_size):
            if iterator_row == sliding_len:
                row = img_size - crop_size
            else:
                iterator_row = iterator_row + 1
            if row + crop_size <= img_size and col + crop_size <= img_size:
                cropped_windows.append(((row, row + crop_size), (col, col + crop_size)))
        iterator_row = 0
    return cropped_windows


def crop_geo_transform(
    temp_image, cropped_windows, cropped_image_dimension, temp_label=None
):
    cropped_data = {}

    for index, tiff_window in zip(range(0, len(cropped_windows)), cropped_windows):
        cropped_image = temp_image.read(window=tiff_window)
        if temp_label is not None:
            cropped_label = temp_label.read(window=tiff_window)
            kwargs_label = temp_label.meta.copy()
            kwargs_label.update(
                {
                    "crs": "EPSG:4326",
                    "height": cropped_image_dimension,
                    "width": cropped_image_dimension,
                    "transform": temp_label.window_transform(tiff_window),
                }
            )

            if not Image.fromarray(cropped_label[0]).getbbox():
                continue
        else:
            cropped_label = None
            kwargs_label = None

        kwargs_image = temp_image.meta.copy()
        kwargs_image.update(
            {
                "crs": "EPSG:4326",
                "height": cropped_image_dimension,
                "width": cropped_image_dimension,
                "transform": temp_image.window_transform(tiff_window),
            }
        )

        cropped_data[index] = {
            "image": cropped_image,
            "label": cropped_label,
            "kwargs_image": kwargs_image,
            "kwargs_label": kwargs_label,
        }

    return cropped_data


def crop_image(temp_image, cropped_windows):
    cropped_images = []
    for num_win, win in enumerate(cropped_windows):
        cropped_images.append(temp_image[win[0][0] : win[0][1], win[1][0] : win[1][1]])
    return cropped_images


def adjust_prediction_overlap(
    final_image, prediction, part_1_x, part_1_y, part_2_x, part_2_y
):
    cropped_image = final_image[part_1_x:part_1_y, part_2_x:part_2_y]
    prediction = np.add(cropped_image, prediction)
    cropped_x, cropped_y = np.nonzero(cropped_image)
    for iterator in range(len(cropped_x)):
        prediction[cropped_x[iterator], cropped_y[iterator]] = float(
            prediction[cropped_x[iterator], cropped_y[iterator]] / 2
        )
    final_image[part_1_x:part_1_y, part_2_x:part_2_y] = prediction
    return final_image


def get_cropped_data(image, label, crop_size, sliding_len=10, keep_original=False):
    img_size, img_size = image.shape
    if label is not None:
        lbl_size, lbl_size = label.shape
        assert lbl_size == img_size

    cropped_windows = get_windows(crop_size, img_size=img_size, sliding_len=sliding_len)

    if keep_original:
        crop_size = img_size
    cropped_data = crop_geo_transform(
        temp_image=image,
        temp_label=label,
        cropped_windows=cropped_windows,
        cropped_image_dimension=crop_size,
    )

    return cropped_data, cropped_windows, img_size


def create_folder(save_dir):
    current_save_dir = save_dir
    save_dir = os.path.join(current_save_dir, "cropped")

    image_dir = os.path.join(save_dir, "images")
    label_dir = os.path.join(save_dir, "labels")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    return image_dir, label_dir


def perform_crop_image_label(
    image_dir,
    label_dir,
    crop_size=384,
    image_dimension=512,
    save_dir=os.getcwd(),
    keep_original=False,
):
    save_image_dir, save_label_dir = create_folder(save_dir)
    all_the_files = [files for files in os.listdir(image_dir)]
    sliding_len = get_sliding_len(crop_size, image_dimension)

    for iterator, file_name in enumerate(all_the_files):
        Printer(str(iterator) + "/" + str(len(all_the_files)))

        image_file = os.path.join(image_dir, file_name)
        label_file = os.path.join(label_dir, file_name)

        image_rasterio = rasterio.open(image_file, driver="GTiff")
        label_rasterio = rasterio.open(label_file, driver="GTiff")

        cropped_data, cropped_windows, img_size = get_cropped_data(
            image=image_rasterio,
            label=label_rasterio,
            crop_size=crop_size,
            sliding_len=sliding_len,
            keep_original=keep_original,
        )

        for key, value in cropped_data.items():
            save_image_path = os.path.join(save_image_dir, str(key) + "_" + file_name)
            save_label_path = os.path.join(save_label_dir, str(key) + "_" + file_name)

            save_image = value["image"]
            save_label = value["label"]
            kwargs_image = value["kwargs_image"]
            kwargs_label = value["kwargs_label"]
            with rasterio.open(save_image_path, "w", **kwargs_image) as dst:
                dst.write(save_image)

            with rasterio.open(save_label_path, "w", **kwargs_label) as dst:
                dst.write(save_label)


def get_sliding_len(crop_dimension, image_dimension):
    if crop_dimension == image_dimension:
        return 1
    else:
        return image_dimension // crop_dimension


def perform_crop_image(
    image_dir,
    crop_size=384,
    image_dimension=512,
    save_dir=os.getcwd(),
    keep_original=False,
):
    save_image_dir, save_label_dir = create_folder(save_dir)
    all_the_files = [files for files in os.listdir(image_dir)]

    sliding_len = get_sliding_len(crop_size, image_dimension)
    for iterator, file_name in enumerate(all_the_files):
        Printer(str(iterator) + "/" + str(len(all_the_files)))
        image_file = os.path.join(image_dir, file_name)

        image_rasterio = rasterio.open(image_file, driver="GTiff")

        cropped_data, cropped_windows, img_size = get_cropped_data(
            image=image_rasterio,
            label=None,
            crop_size=crop_size,
            sliding_len=sliding_len,
            keep_original=keep_original,
        )

        for key, value in cropped_data.items():
            save_image_path = os.path.join(save_image_dir, str(key) + "_" + file_name)

            save_image = value["image"]
            kwargs_image = value["kwargs_image"]
            with rasterio.open(save_image_path, "w", **kwargs_image) as dst:
                dst.write(save_image)


perform_crop_image_label(
    "/home/palnak/Dataset/scratch_inria/NEW2-AerialImageDataset/AerialImageDataset/train/images",
    "/home/palnak/Dataset/scratch_inria/NEW2-AerialImageDataset/AerialImageDataset/train/gt",
    crop_size=512,
    image_dimension=5000,
    save_dir="/home/palnak/Dataset/scratch_inria/512x512",
    keep_original=False,
)
# perform_crop_image(
#     image_dir="/home/palnak/Temp/prediction/test/images",
#     save_dir="/home/palnak/Temp/prediction/test/",
#     crop_size=256,
#     image_dimension=512,
#     keep_original=False,
# )
