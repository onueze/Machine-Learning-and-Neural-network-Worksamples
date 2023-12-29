import numpy as np

from .activation import Activation


class Sigmoid(Activation):
    """
    Sigmoid activation function to be used in ActivationLayer.

    I just love making docstrings.
    """

    def act(self, x: np.array) -> np.array:
        """
        Apply Sigmoid function.
        :param x: input
        :return: Sigmoid-activated input
        """
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.array) -> np.array:
        """
        Sigmoid derivative in respect to x.
        :param x: input
        :return: dSigmoid/dX
        """
        
        sigmoid_output = self.act(x)
        return sigmoid_output * (1 - sigmoid_output)
