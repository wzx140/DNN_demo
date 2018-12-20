from config import *
from dnn import Dnn
from util import load_data, load_pic
import numpy as np

if __name__ == '__main__':

    # this is the data to test dnn
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    dnn = Dnn(layer_dims, lambd=lambd, keep_prob=keep_prob)
    # np.random.seed(1)
    dnn.initialize_parameters()
    for i in range(1, num_iterations + 1):
        dnn.forward(train_x)

        dnn.backward(train_y)
        dnn.update_parameters(learning_rate)

        if print_detail and (i % 100 == 0 or i == 1):
            cost = dnn.cost(train_y)
            train_accuracy, train_predict_y = dnn.predict(train_x, train_y)
            test_accuracy, test_predict_y = dnn.predict(test_x, test_y)

            print('Iteration %d' % i)
            if cost:
                print('Cost: %f' % cost)
            print('Train accuracy: %f' % train_accuracy)
            print('Test accuracy: %f\n' % test_accuracy)

    X = load_pic(pic_path)
    if X is not None:
        predict_y = dnn.predict(X)
        print('The output: ' + str(predict_y))
