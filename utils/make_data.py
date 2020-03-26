import shutil
from os.path import join, exists
from os import makedirs, listdir
from random import shuffle


def create_data_set_dirs(
    train_images_dir,
    train_labels_dir,
    test_images_dir,
    test_labels_dir,
    val_images_dir,
    val_labels_dir,
):
    """

    :param train_images_dir: train_image_dir path
    :param train_labels_dir: tarin_labels_dir path
    :param test_images_dir: test_images_dir path
    :param test_labels_dir: test_labels_dir path
    :param val_images_dir: val_images_dir path
    :param val_labels_dir: val_labels_dir path
    :return:
    """
    try:
        if not exists(train_images_dir):

            makedirs(train_images_dir)
        if not exists(train_labels_dir):
            makedirs(train_labels_dir)

        if not exists(test_images_dir):
            makedirs(test_images_dir)
        if not exists(test_labels_dir):
            makedirs(test_labels_dir)

        if not exists(val_images_dir):
            makedirs(val_images_dir)
        if not exists(val_labels_dir):
            makedirs(val_labels_dir)

        return True

    except Exception as ex:
        print(str(ex))
        return False


def distribute_images(
    images_dir, labels_dir, output_dir, train_percentage=0.96, test_percentage=0.02
):
    """

    :param images_dir: input_images dir path
    :param labels_dir: labels_dir path
    :param output_dir: output_dir_path
    :param train_percentage: get following percentage of images from dataset for training
    :param test_percentage: get following percentage of images from dataset for training
    :return: Boolean True after successful completion
    """

    if not exists(output_dir):
        makedirs(output_dir)

    train_dir = join(output_dir, "train")
    train_images_dir = join(train_dir, "images")
    train_labels_dir = join(train_dir, "labels")

    test_dir = join(output_dir, "test")
    test_images_dir = join(test_dir, "images")
    test_labels_dir = join(test_dir, "labels")

    val_dir = join(output_dir, "val")
    val_images_dir = join(val_dir, "images")
    val_labels_dir = join(val_dir, "labels")

    data_set_created = create_data_set_dirs(
        train_images_dir,
        train_labels_dir,
        test_images_dir,
        test_labels_dir,
        val_images_dir,
        val_labels_dir,
    )

    if not data_set_created:
        print("Dataset couldn't be created")
        exit(1)

    all_images = listdir(images_dir)

    # shuffle all images
    shuffle(all_images)

    total_images = len(all_images)
    print("Total Number of Images: {}".format(total_images))

    train_count = int(total_images * train_percentage)
    print("Train Images : {}".format(train_count))

    test_count = int(total_images * test_percentage)
    print("Test Images : {}".format(test_count))

    val_count = total_images - (train_count + test_count)
    print("Validation Images : {}".format(val_count))

    for i in range(0, train_count):
        shutil.copy(
            join(images_dir, all_images[i]), join(train_images_dir, all_images[i])
        )
        shutil.copy(
            join(labels_dir, all_images[i]), join(train_labels_dir, all_images[i])
        )

    print("Done with Training Images")

    for i in range(train_count, train_count + test_count):
        shutil.copy(
            join(images_dir, all_images[i]), join(test_images_dir, all_images[i])
        )
        shutil.copy(
            join(labels_dir, all_images[i]), join(test_labels_dir, all_images[i])
        )

    print("Done with Test Images")

    for i in range(train_count + test_count, total_images):
        shutil.copy(
            join(images_dir, all_images[i]), join(val_images_dir, all_images[i])
        )
        shutil.copy(
            join(labels_dir, all_images[i]), join(val_labels_dir, all_images[i])
        )

    print("Done with Validation Images")

    return True


path_to_images = "/home/palnak/Dataset/scratch_inria/512x512/cropped/images"
path_to_labels = "/home/palnak/Dataset/scratch_inria/512x512/cropped/labels"
data_set_dir = "/home/palnak/Dataset/temp"
distribute_images(
    images_dir=path_to_images,
    labels_dir=path_to_labels,
    output_dir=data_set_dir,
    train_percentage=0.80,
    test_percentage=0.00,
)
