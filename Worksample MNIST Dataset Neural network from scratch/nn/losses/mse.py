from .loss import Loss
import numpy as np


class MeanSquaredError(Loss):

    def loss(self, y_pred: np.array, y_true: np.array) -> np.array:
        """
        Calculates the Mean Squared Error loss for the given predicted probability distribution.

        :param y_pred: numpy array of predicted probabilities. Has dimensions [n_samples, n_classes].
        :param y_true: numpy array of true probabilities. Has dimensions [n_samples, n_classes]. # TODO: true probabilities or hot encode?
        :return: numpy array of loss values for the current batch.
        """
        return np.sum((y_true - y_pred) ** 2) / len(y_pred)

    def derivative(self, y_pred: np.array, y_true: np.array) -> np.array:
        """
        Calculates the Mean Squared Error loss derivative in respect to the given predicted probability distribution.

        :param y_pred: numpy array of predicted probabilities. Has dimensions [n_samples, n_classes].
        :param y_true: numpy array of true probabilities. Has dimensions [n_samples, n_classes]. # TODO: true probabilities or hot encode?
        :return: numpy array of loss values for the current batch.
        """

        return 2 * (y_pred - y_true) / len(y_pred)
