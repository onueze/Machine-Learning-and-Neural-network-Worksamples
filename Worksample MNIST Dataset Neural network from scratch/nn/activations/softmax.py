from .activation import Activation
import numpy as np


class Softmax(Activation):
    """
    Softmax activation function.

    """

    def act(self, x: np.array) -> np.array:
        """
        Apply Softmax to x.
        :param x: input, numpy array
        :return: Softmax-activated input
        """
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        denominator = np.sum(e_x, axis=-1, keepdims=True)

        # subtracting max from x to prevent overflow from happening, will still have the same ratio
        epsilon = 1e-15
        result = e_x / (denominator + epsilon)

        return result

    def derivative(self, x: np.array) -> np.array:
        """
        Compute the derivative of the Softmax function.
        :param x: input
        :return: dSoftmax/dY
        """

        s = self.act(x)
        return s * (1 - s)
