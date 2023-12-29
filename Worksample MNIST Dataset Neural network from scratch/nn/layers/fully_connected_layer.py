from .layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):

    lr: float
    l1_lambda: float
    l2_lambda: float

    def __init__(self, in_size: int, out_size: int, lr: float = 0.01, l1_lambda: float = 0.0, l2_lambda: float = 0.0, momentum_rate: float = 0.9):
        """
        Fully-connected layer of N(=out_size) neurons. Using He weights initialisation.
        :param in_size: number of features.
        :param out_size: number of neurons.
        :param l1_lambda: coefficient for L1 regularization.
        :param l2_lambda: coefficient for L2 regularization.
        :param momentum_rate: coefficient for momentum rate.
        """
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.lr = lr
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        self.weights = np.random.randn(in_size, out_size) * np.sqrt(2. / in_size)
        self.biases = np.zeros(out_size)

        self.momentum_rate = momentum_rate
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.biases)

    def forward(self, x: np.array) -> np.array:
        """
        Forward pass.
        :param x: input numpy array.
        :return: y, where y = Wx + B.
        """
        self.input = x
        return x.dot(self.weights) + self.biases

    def backprop(self, x: np.array) -> np.array:
        """
        Back pass.
        :param x: input array of dE_dY.
        :return: numpy array of dE_dX.
        """

        dEdW = np.dot(self.input.T, x) + self.l1_lambda * np.sign(self.weights) + self.l2_lambda * self.weights
        dEdB = np.sum(x, axis=0)

        self.velocity_w = self.momentum_rate * self.velocity_w + (1 - self.momentum_rate) * dEdW
        self.velocity_b = self.momentum_rate * self.velocity_b + (1 - self.momentum_rate) * dEdB

        self.weights -= self.lr * self.velocity_w
        self.biases -= self.lr * self.velocity_b

        return np.dot(x, self.weights.T)
