from .loss import Loss
import numpy as np


class CategoricalCrossEntropy(Loss):

    def loss(self, y_pred: np.array, y_true: np.array) -> np.array:
        """
        Calculates the Categorical Cross Entropy loss of the predicted probability distributions over real values.

        :param y_pred: numpy array of predicted probabilities. Has dimensions [n_samples, n_classes].
        :param y_true: numpy array of true probabilities. Has dimensions [n_samples, n_classes]. # TODO: true probabilities or hot encode?
        :return: numpy array of loss values for the current batch.
        """

        epsilon = 1e-15
        clipped_y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Assuming y_true is one-hot encoded
        loss = -np.sum(y_true * np.log(clipped_y_pred), axis=1) / len(y_pred)
        return loss

    def derivative(self, y_pred: np.array, y_true: np.array) -> np.array:
        """
        Calculates the derivative of Categorical Cross Entropy loss in respect to the y_pred.

        :param y_pred: numpy array of predicted probabilities. Has dimensions [n_samples, n_classes].
        :param y_true: numpy array of true probabilities. Has dimensions [n_samples, n_classes]. # TODO: true probabilities or hot encode?
        :return: dE_dY.
        """

        return y_pred - y_true
