import numpy as np


class Activation:
    """
    Abstract class for the activations to be inherited from.
    """

    def act(self, x: np.array) -> np.array:
        """
        Abstract activation function method
        :param x: input array
        :return: "activated" input array
        """
        pass

    def derivative(self, x: np.array) -> np.array:
        """
        Abstract derivative of activation with respect to the x
        :param x: input array
        :return: activation function derived in respect to the input array
        """
        pass
