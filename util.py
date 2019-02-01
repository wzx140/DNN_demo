import numpy as np
import csv


def sigmoid(Z):
    """
    Implement sigmoid activation in numpy

    :param Z: matrix in numpy
    :return: output of sigmoid(Z), same shape as Z
    """
    A = 1 / (1 + np.exp(-Z))

    return A


def relu(Z):
    """
    Implement the Rectified Linear Unit activation in numpy

    :param Z: matrix in numpy
    :return: output of relu(Z), same shape as Z
    """
    A = np.maximum(0, Z)

    return A


def sigmoid_backward(dA, Z):
    """
     Implement the backward propagation for a single SIGMOID unit.

    :param dA: post gradient
    :param Z: current linear forward value
    :return: the derivative value of the sigmoid(Z)
    """
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ


def relu_backward(dA, Z):
    """
     Implement the backward propagation for a single RELU unit.

    :param dA: post gradient
    :param Z: current linear forward value
    :return: the derivative value of the relu(Z)
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def load_data():
    """
    load welding test and train data
    :return: train_x, train_y, test_x, test_y
    """
    # each sample consists of 6*1508 data
    length = 1508
    length_flatten = 9048

    data_x = np.zeros((length_flatten, 550), dtype=np.float32)
    data_y = np.array([1] * 500 + [0] * 50, dtype=np.float32).reshape((1, 550))
    # print(len(data_y))

    with open('dataSets/data_plain.csv') as f:
        f_csv = csv.reader(f)
        for i, row in enumerate(f_csv):
            sample_index = i // length
            vector_st_index = i % 1508 * 6
            vector_end_index = i % 1508 * 6 + 6
            data_x[vector_st_index:vector_end_index, sample_index] = [float(l) for l in row]

    # split train and test
    permutation = list(np.random.permutation(550))
    shuffled_x = data_x[:, permutation]
    shuffled_y = data_y[:, permutation]
    train_x = shuffled_x[:, 0:500]
    train_y = shuffled_y[:, 0:500]
    test_x = shuffled_x[:, 500:550]
    test_y = shuffled_y[:, 500:550]
    return train_x, train_y, test_x, test_y
