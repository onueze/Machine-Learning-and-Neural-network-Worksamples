import numpy as np


class Loss:

    """
    Abstract class for the loss functions to be inherited from.
    """

    def loss(self, y_pred: np.array, y_true: np.array) -> np.array:
        """
        Abstract method for loss calculation.
        """
        pass

    def derivative(self, y_pred: np.array, y_true: np.array) -> np.array:
        """
        Abstract method for the calculation of loss derivative against the predicted values.

        dLoss/dY_pred.

        """
        pass
