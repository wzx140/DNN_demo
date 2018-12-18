import numpy as np
import h5py
import os
import scipy
from scipy import ndimage


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
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_pic(path: str = None):
    if path is None or not os.path.isdir(path):
        path = 'pic'

    files = os.listdir(path)
    files.sort()
    X = None
    for file in files:
        if file.startswith('data_'):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                image = np.array(ndimage.imread(file_path, flatten=False, mode='RGB'))
                temp = scipy.misc.imresize(image, size=(64, 64)).reshape((64 * 64 * 3, 1))
                if X is None:
                    X = temp
                else:
                    X = np.hstack((X, temp))

    # print(X.shape)
    return X
