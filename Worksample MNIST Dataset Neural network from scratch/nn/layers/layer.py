import numpy as np


class Layer:
    """
    Abstract layer class to be inherited from.
    """
    input: np.array
    training: bool

    def __init__(self):
        """
        Initially created to be trained.


        Motivating...

        """

        self.training = True

    def forward(self, x: np.array) -> np.array:
        """
        Forward propagation abstract layer.
        :param x: input array
        :return: input passed through the layer
        """
        pass

    def backprop(self, x: np.array) -> np.array:
        """
        Backpropagation abstract layer.
        :param x: input array, matrix
        :return: input reverse-passed through the layer
        """
        pass
