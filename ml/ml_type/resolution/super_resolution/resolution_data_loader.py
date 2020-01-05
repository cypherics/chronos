import cv2

from ml.base import BaseDataLoader
from utils.data_image_and_label_loader import load_image
from utils.torch_tensor_conversion import to_input_image_tensor


class ResolutionDataLoader(BaseDataLoader):
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
        if len(self.images) == 0:
            return len(self.labels)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        lbl_file_name = self.labels[idx]
        lbl = load_image(str(lbl_file_name))

        if self.mode in ["train", "val"]:
            img = None
            mask = lbl

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
            img = lbl
            img_file_name = lbl_file_name
            input_dictionary = self.perform_image_operation_test(img=img)
            assert isinstance(input_dictionary, dict), "Return type should be dict"
            assert "image" in input_dictionary, "while passing image use key-image"

            return input_dictionary, str(img_file_name)

        else:
            raise NotImplementedError

    def perform_image_operation_train_and_val(self, img, mask):
        if img is None:
            w, h, _ = mask.shape
            img = self.perform_scale(mask, dimension=(w // 4, h // 4))

        img, mask = super().perform_transformation(img, mask)

        img = super().perform_normalization(img)
        mask = self.get_label_normalization(mask)

        return {
            "image": to_input_image_tensor(img),
            "label": to_input_image_tensor(mask),
        }

    def perform_image_operation_test(self, img):
        w, h, _ = img.shape
        img = self.perform_scale(img, dimension=(w // 4, h // 4))
        img = super().perform_normalization(img)
        input_dictionary = {"image": to_input_image_tensor(img)}
        return input_dictionary

    def get_label_normalization(self, mask):
        return mask / 255

    @staticmethod
    def perform_scale(img, dimension):
        new_height, new_width = dimension
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        return img
