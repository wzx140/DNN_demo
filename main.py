from config import *
from dnn import Dnn
from util import load_data
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # this is the data to test dnn
    train_x, train_y, test_x, test_y = load_data()

    # log the cost and accuracy
    cost_log = []
    test_log = []
    train_log = []

    dnn = Dnn(layer_dims, lambd=lambd, keep_prob=keep_prob)
    dnn.initialize_parameters()
    for i in range(1, num_iterations + 1):
        dnn.forward(train_x)

        dnn.backward(train_y)
        dnn.update_parameters(learning_rate)

        cost = dnn.cost(train_y)
        train_accuracy, train_predict_y = dnn.predict(train_x, train_y)
        test_accuracy, test_predict_y = dnn.predict(test_x, test_y)
        cost_log.append(cost)
        train_log.append(train_accuracy)
        test_log.append(test_accuracy)

        if print_detail and (i % 10 == 0 or i == 1):
            print('Iteration %d' % i)
            if cost:
                print('Cost: %f' % cost)
            print('Train accuracy: %f' % train_accuracy)
            print('Test accuracy: %f\n' % test_accuracy)

    plt.figure()
    plt.subplot(131)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.plot(range(len(cost_log)), cost_log)

    plt.subplot(132)
    plt.xlabel('Iteration')
    plt.ylabel('Train accuracy')
    plt.plot(range(len(train_log)), train_log)

    plt.subplot(133)
    plt.xlabel('Iteration')
    plt.ylabel('Test accuracy')
    plt.plot(range(len(test_log)), test_log)

    plt.show()
