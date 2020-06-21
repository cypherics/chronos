from abc import ABCMeta, abstractmethod


class BaseEvaluator(metaclass=ABCMeta):
    def handle_prediction(self, prediction):
        prediction = self.classifier_activation(prediction)
        prediction = self.generate_image(prediction)
        return prediction

    @abstractmethod
    def generate_image(self, prediction):
        raise NotImplementedError

    @abstractmethod
    def classifier_activation(self, prediction):
        raise NotImplementedError

    @abstractmethod
    def create_prediction_grid(self, inputs, prediction):
        raise NotImplementedError
