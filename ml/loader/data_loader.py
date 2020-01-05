import sys
import traceback

import torch
from torch.utils.data import DataLoader

from ml import ml_type


class DominusDataLoader:
    def __init__(self, training_configuration, logger):
        self.training_configuration = training_configuration
        self.logger = logger

        self.train_data_set, self.val_data_set, self.test_data_set = (
            self.load_data_set()
        )
        self.train_loader, self.val_loader, self.test_loader = self.load_loader()

    def load_data_set(self):
        try:
            root = self.training_configuration["root"]
            model_input_dimension = self.training_configuration["initial_assignment"][
                "model_input_dimension"
            ]
            normalization = self.training_configuration["normalization"]
            transformation = self.training_configuration["transformation"]
            training_problem = self.training_configuration["problem_type"]

            train_data_set = getattr(getattr(ml_type, training_problem), "Dataloader")(
                root, model_input_dimension, "train", transformation, normalization
            )

            val_data_set = getattr(getattr(ml_type, training_problem), "Dataloader")(
                root, model_input_dimension, "val", transformation, normalization
            )

            test_data_set = getattr(getattr(ml_type, training_problem), "Dataloader")(
                root, model_input_dimension, "test", transformation, normalization
            )
            return train_data_set, val_data_set, test_data_set
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))

    def load_loader(self):
        try:
            batch_size = self.training_configuration["initial_assignment"]["batch_size"]
            train_loader = DataLoader(
                dataset=self.train_data_set,
                shuffle=True,
                num_workers=0,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available(),
            )

            val_loader = DataLoader(
                dataset=self.val_data_set,
                shuffle=True,
                num_workers=0,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available(),
            )

            test_loader = DataLoader(
                dataset=self.test_data_set,
                shuffle=True,
                num_workers=0,
                batch_size=batch_size,
                pin_memory=torch.cuda.is_available(),
            )
            self.logger.log_info("Inference and Data loader - DataLoader complete")

            return train_loader, val_loader, test_loader
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))
