import sys
import traceback

from ml import ml_type


class DominusValidation:
    def __init__(self, training_configuration, logger):
        self.training_configuration = training_configuration
        self.logger = logger
        self.validation = self.load_validation()

    def load_validation(self):
        try:
            problem_type = self.training_configuration["problem_type"]
            current_problem_type_loader = getattr(
                getattr(ml_type, problem_type), "Validation"
            )(problem_type)
            self.logger.log_info(
                "Validation - {} Problem Type loaded".format(problem_type)
            )

            return current_problem_type_loader
        except Exception as ex:
            self.logger.log_exception(ex)
            sys.exit(str(traceback.format_exc()))
