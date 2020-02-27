import numpy as np


def get_sliding_len(original_size, step_size):
    return original_size // step_size


def get_windows(crop_size, img_size, step_size):

    cropped_windows = []
    step_size = crop_size if step_size is None or step_size == 0 else step_size
    sliding_len = get_sliding_len(img_size, step_size)
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


def crop_image(temp_image, cropped_windows):
    cropped_images = []
    for num_win, win in enumerate(cropped_windows):
        cropped_images.append(temp_image[win[0][0] : win[0][1], win[1][0] : win[1][1]])
    return cropped_images


def adjust_prediction_overlap(
    final_image, prediction, part_1_x, part_1_y, part_2_x, part_2_y
):
    cropped_image = final_image[part_1_x:part_1_y, part_2_x:part_2_y]

    temp = np.zeros(cropped_image.shape)
    final_image[part_1_x:part_1_y, part_2_x:part_2_y] = temp

    prediction = cropped_image + prediction
    cropped_x, cropped_y = np.nonzero(cropped_image)
    for iterator in range(len(cropped_x)):
        prediction[cropped_x[iterator], cropped_y[iterator]] = float(
            prediction[cropped_x[iterator], cropped_y[iterator]] / 2
        )
    final_image[part_1_x:part_1_y, part_2_x:part_2_y] = prediction
    return final_image


def perform_sliding_prediction(image, crop_size, step_size=None):
    img_size, img_size, bands = image.shape
    cropped_windows = get_windows(crop_size, img_size=img_size, step_size=step_size)
    cropped_images = crop_image(temp_image=image, cropped_windows=cropped_windows)
    return cropped_images, cropped_windows, img_size
