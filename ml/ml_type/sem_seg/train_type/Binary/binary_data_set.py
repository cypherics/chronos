from ml.base.base_pt_dataset import BaseDataSetPt
from utils.data_image_and_label_loader import load_image, load_mask
from utils.torch_tensor_conversion import to_input_image_tensor, to_label_image_tensor


class BinaryDataSet(BaseDataSetPt):
    def __init__(
        self,
        root,
        model_input_dim=None,
        mode="train",
        transform=None,
        normalization=None,
    ):
        super().__init__(root, model_input_dim, mode, transform, normalization)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file_name = self.images[idx]
        img = load_image(str(img_file_name))

        if self.mode in ["train", "val"]:
            label_file_name = self.labels[idx]
            mask = load_mask(str(label_file_name))

            input_dictionary = self.perform_image_operation_train_and_val(
                img=img, mask=mask
            )
            assert isinstance(input_dictionary, dict), "Return type should be dict"

            assert (
                "image" in input_dictionary and "label" in input_dictionary
            ), "while passing image use key-image and for label use key-label"

            return input_dictionary

        elif self.mode == "test":
            # predict mode
            input_dictionary = self.perform_image_operation_test(img=img)
            assert isinstance(input_dictionary, dict), "Return type should be dict"
            assert "image" in input_dictionary, "while passing image use key-image"

            return input_dictionary, str(img_file_name)

        else:
            raise NotImplementedError

    def perform_image_operation_train_and_val(self, img, mask):
        img, mask = super().perform_transformation(img, mask)
        img, mask = super().handle_image_size(img, mask, self.model_input_dimension)

        img = super().perform_normalization(img)
        mask = self.get_label_normalization(mask)
        return {
            "image": to_input_image_tensor(img),
            "label": to_label_image_tensor(mask),
        }

    def perform_image_operation_test(self, img):
        if self.model_input_dimension != (img.shape[0], img.shape[1]):
            height, width = super().get_random_crop_x_and_y(
                self.model_input_dimension, img.shape
            )
            img = super().crop_image(img, self.model_input_dimension, (height, width))

        img = super().perform_normalization(img)
        input_dictionary = {"image": to_input_image_tensor(img)}
        return input_dictionary

    @staticmethod
    def get_label_normalization(mask):
        normalized_mask = mask / 255
        return normalized_mask
