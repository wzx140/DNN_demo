from enum import Enum, unique
from util import *


@unique
class ACTIVATIONS(Enum):
    RELU = 0
    SIGMOID = 1


class Dnn(object):
    """
    deep neural network
    """

    def __init__(self, layer_dims: list):
        """
        init the class

        :param layer_dims: the dimensions of each layer in dnn
        """
        # store w and b
        self.__parameters = {}
        self.__m = 0
        self.__layer_dims = layer_dims
        # number of layers except input layer
        self.__L = len(layer_dims) - 1
        self.__AL = None
        # store z and a_prev
        Z_cache = [0] * self.__L
        a_prev_cache = [0] * self.__L
        self.__cache = (a_prev_cache, Z_cache)
        self.__grads = {}

    def initialize_parameters(self):
        # np.random.seed(1)
        for l in range(1, self.__L + 1):
            # avoid gradient explosion and gradient disappear
            self.__parameters['W' + str(l)] = np.random.randn(self.__layer_dims[l], self.__layer_dims[l - 1]) * np.sqrt(
                2/self.__layer_dims[l - 1])
            self.__parameters['b' + str(l)] = np.zeros((self.__layer_dims[l], 1))
        # print(self.__parameters)

    def __forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the single layer

        :param A_prev: input data from previous layer
        :param W: weights matrix
        :param b: bias vector
        :param activation: activation used for this layer
        :return:
        """

        if activation == ACTIVATIONS.SIGMOID:
            Z = np.dot(W, A_prev) + b
            A = sigmoid(Z)

        elif activation == ACTIVATIONS.RELU:
            Z = np.dot(W, A_prev) + b
            A = relu(Z)

        return A, Z

    def forward(self, X):
        """
        forward propagation

        :param X: input data
        """
        assert (X.shape[0] == self.__parameters['W1'].shape[1])
        A = X
        self.__m = X.shape[1]
        a_prev_cache, Z_cache = self.__cache
        # First L-1 layers use relu function
        for l in range(1, self.__L):
            A_prev = A
            A, Z = self.__forward(A_prev, self.__parameters['W' + str(l)], self.__parameters['b' + str(l)],
                                  ACTIVATIONS.RELU)
            a_prev_cache[l - 1] = A_prev
            Z_cache[l - 1] = Z

        # last layer use sigmoid function
        AL, Z = self.__forward(A, self.__parameters['W' + str(self.__L)], self.__parameters['b' + str(self.__L)],
                               ACTIVATIONS.SIGMOID)
        a_prev_cache[-1] = A
        Z_cache[-1] = Z
        self.__AL = AL

    def cost(self, Y) -> ():
        """
        compute the cost

        :return:
        """
        AL = self.__AL
        assert (self.__m == Y.shape[1])
        cost = (1. / self.__m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
        cost = np.squeeze(cost)
        return cost

    def __backward(self, dA, A_prev, Z, W, activation):
        """
        Implement the backward propagation for the singer layer

        :param dA: post activation gradient
        :param A_prev: A of previous layer
        :param Z: Z of current layer
        :param W: W of current layer
        :param activation: activation of current layer
        :return:
        """

        if activation == ACTIVATIONS.SIGMOID:
            dZ = sigmoid_backward(dA, Z)
            dW = 1. / self.__m * np.dot(dZ, A_prev.T)
            db = 1. / self.__m * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)
        elif activation == ACTIVATIONS.RELU:
            dZ = relu_backward(dA, Z)
            dW = 1. / self.__m * np.dot(dZ, A_prev.T)
            db = 1. / self.__m * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def backward(self, Y):
        """
        backward propagation

        :param Y: except value
        """
        assert (self.__m == Y.shape[1])
        grads = self.__grads
        L = self.__L
        A_prev, Z, = self.__cache
        AL = self.__AL
        # Y = Y.reshape(AL.shape)
        dAL = - (Y / AL - (1 - Y) / (1 - AL))

        # the last layer
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = \
            self.__backward(dAL, A_prev[-1], Z[-1], self.__parameters['W' + str(L)], ACTIVATIONS.SIGMOID)

        # First L-1 layers
        for l in reversed(range(L - 1)):
            dA_prev_temp, dW_temp, db_temp = \
                self.__backward(grads["dA" + str(l + 2)], A_prev[l], Z[l], self.__parameters['W' + str(l + 1)],
                                ACTIVATIONS.RELU)
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        # print(grads)

    def update_parameters(self, learning_rate: float):
        """
        Update parameters using gradient descent
        :param learning_rate
        :return:
        """
        for l in range(self.__L):
            self.__parameters["W" + str(l + 1)] -= learning_rate * self.__grads["dW" + str(l + 1)]
            self.__parameters["b" + str(l + 1)] -= learning_rate * self.__grads["db" + str(l + 1)]

    def predict(self, X, Y=None):
        """
        return predict_Y if it is not given Y
        return accuracy and predict_Y if it is given Y

        :param X: data set of examples
        :param Y: actual label of the X
        :return:
        """

        m = X.shape[1]
        p = np.zeros((1, m))

        A = X

        for l in range(1, self.__L):
            A_prev = A
            A, Z = self.__forward(A_prev, self.__parameters['W' + str(l)], self.__parameters['b' + str(l)],
                                  ACTIVATIONS.RELU)

        # last layer use sigmoid function
        AL, Z = self.__forward(A, self.__parameters['W' + str(self.__L)], self.__parameters['b' + str(self.__L)],
                               ACTIVATIONS.SIGMOID)

        probas = AL

        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        if Y is not None:
            return np.sum((p == Y) / m), p
        else:
            return p
