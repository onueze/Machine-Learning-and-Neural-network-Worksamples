import numpy as np

from ..activations.activation import Activation
from .layer import Layer


class ActivationLayer(Layer):

    def __init__(self, activation: Activation):
        """
        Initialises the activation layer
        :param activation: chosen activation function. Inherits from "Activation" class.
        """
        super().__init__()
        self.activation = activation

    def forward(self, x: np.array) -> np.array:
        """
        Pass the weighted product through the activation layer. activation(y = Wx + B)
        :param x: pre-activated input array.
        :return: activated input array.
        """
        self.input = x
        return self.activation.act(x)

    def backprop(self, x: np.array) -> np.array:
        """
        Reverse-pass the x through the activation layer. dE/dy = dE/dA * dA/dy
        :param x: np array, dE/dA (previous dE/dX). Derivative of the loss function with respect to the activation function.
        :return: np array, dE / dy. derivative of the loss function with respect to y.
        """

        return x * self.activation.derivative(self.input)
