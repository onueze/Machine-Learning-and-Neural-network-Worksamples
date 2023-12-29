import numpy as np

from . import Layer


class DropoutLayer(Layer):

    mask: np.array = None

    def __init__(self, dropout_rate: float, scale_during_train: bool = False):
        """
        Dropout Layer to be used during training, for less over-fit likelihood.
        :param dropout_rate: float, determines the proportion of neurons that will "hibernate" during pass.
        :param scale_during_train: *NEW* optimisation that will attempt to maintain the signal intensity during training, neglecting the signal de-amplification.
        """

        super().__init__()
        self.rate = dropout_rate
        self.scale_during_train = scale_during_train

    def forward(self, x: np.array) -> np.array:
        """
        Applies dropout mask to the input x, neglecting a proportion of neurons determined by the dropout_rate.
        Applied during training stage only.

        :param x: input array of activations.
        :return: partially-hibernated activations.
        """

        if self.training and self.rate > 0:
            self.mask = np.random.rand(*x.shape) < (1 - self.rate)
            x *= self.mask
            if self.scale_during_train:
                x /= (1 - self.rate)

        return x

    def backprop(self, x: np.array) -> np.array:
        """
        Applies the mask to the gradients, making sure that the "hibernated" neurons do not update.

        :param x: input array of dE/dY.
        :return: filtered gradients.
        """

        if self.training:
            if self.mask is not None:
                x *= self.mask  # Apply mask to the gradients

        return x
