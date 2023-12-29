import numpy as np

from .activation import Activation


class ReLU(Activation):
    """
    ReLU activation function.
    """

    def act(self, x: np.array) -> np.array:
        """
        Apply ReLU to x.
        :param x: input, numpy array
        :return: ReLU-activated input
        """
        return np.maximum(0, x)

    def derivative(self, x: np.array) -> np.array:
        """
        ReLU derivative in respect to x.
        :param x: input
        :return: dReLU/dX
        """
        return np.where(x > 0, 1, 0)
