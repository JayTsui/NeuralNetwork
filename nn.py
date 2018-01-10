#!/usr/bin/env python
# coding: utf-8

import gzip
import pickle
import numpy as np
from helper import one_hot
from layer import LinearLayer, Activation, MSECostLayer

class NeutraNetwork(object):

    """
    Neutral Network Model
    """

    def __init__(self, layers, cost_fun):
        self.layers = layers
        self.cost_fun = cost_fun
        self.n_classes = None


    def setup(self, input_shape, learning_rate):
        """
        init layer
        """
        input_n = input_shape
        for layer in self.layers:
            layer.setup(input_n, learning_rate)
            input_n = layer.n_out
        self.n_classes = input_n


    def gradient_descent(self, input_x, input_y):
        """
        grandient descent
        """
        input_a = input_x
        for layer in self.layers:
            input_a = layer.forward(input_a)

        input_grad = self.cost_fun.grad(one_hot(input_y, self.n_classes), input_a)
        for layer in reversed(self.layers):
            input_grad = layer.backward(input_grad)


    def train(self, input_x, labels, num_epochs=10, batch_size=32, learning_rate=0.01):
        """
        train function
        """
        n_batch = input_x.shape[1] // batch_size
        input_shape = input_x.shape[0]
        self.setup(input_shape, learning_rate)

        num_now = 0
        while num_now < num_epochs:
            num_now += 1

            for i in range(n_batch):
                batch_begin = i * batch_size
                batch_end = batch_begin + batch_size
                input_y = labels[batch_begin: batch_end]
                input_a = input_x[:, batch_begin: batch_end]

                self.gradient_descent(input_a, input_y)

            v_loss = self.loss(input_x, labels)
            v_error = self.error(input_x, labels)
            print "iter: %i, loss %.4f, train error %.4f" % (num_now, v_loss, v_error)


    def loss(self, input_x, labels):
        """
        loss function
        """
        labels = one_hot(labels, self.n_classes)
        for layer in self.layers:
            input_x = layer.forward(input_x)
        return self.cost_fun.loss(labels, input_x)


    def predict(self, input_x):
        """
        model predict
        """
        for layer in self.layers:
            input_x = layer.forward(input_x)
        predicts = np.zeros(input_x.shape, dtype=int)
        predicts[input_x > 0.5] = 1
        return predicts


    def error(self, input_x, labels):
        """
        error value
        """
        labels = one_hot(labels, self.n_classes)
        predicts = self.predict(input_x)
        error = predicts != labels
        return np.mean(error)


    def get_model_params(self):
        """
        get model parameters
        """
        model_params = np.zeros(0)
        for layer in self.layers:
            layer_params = np.concatenate((layer.param_w.ravel(), layer.param_b.ravel()))
            model_params = np.concatenate((model_params, layer_params))
        return model_params


    def get_grad_params(self):
        """
        get grad parameters
        """
        grad_params = np.zeros(0)
        for layer in self.layers:
            layer_params = np.concatenate((layer.grad_w.ravel(), layer.grad_b.ravel()))
            grad_params = np.concatenate((grad_params, layer_params))
        return grad_params


    def update_model_params(self, model_params):
        """
        update model parameters
        """
        index = 0
        for layer in self.layers:
            begin_index = index
            end_index = begin_index + layer.param_w.size
            temp_data = model_params[begin_index: end_index]

            temp_data.shape = layer.param_w.shape
            layer.param_w = temp_data

            begin_index = end_index
            end_index = begin_index + layer.param_b.size
            temp_data = model_params[begin_index: end_index]

            temp_data.shape = layer.param_b.shape
            layer.param_b = temp_data
            index = end_index


    def grad_check(self, input_x, labels):
        """
        grad check
        """
        count = 100
        epsilon = 1e-4
        grad_error = 0

        for _ in range(count):
            self.setup(input_x.shape[0], learning_rate=0.1)

            params = self.get_model_params()

            self.gradient_descent(input_x, labels)
            grad_params = self.get_grad_params()

            index = np.random.randint(0, len(params))
            params_a = params.copy()
            params_b = params.copy()
            params_a[index] += epsilon
            params_b[index] -= epsilon

            self.update_model_params(params_a)
            loss_a = self.loss(input_x, labels)
            self.update_model_params(params_b)
            loss_b = self.loss(input_x, labels)

            grad = (loss_a - loss_b) / float(2.0 * epsilon)
            # print grad, grad_params[index], np.abs(grad_params[index] - grad)
            grad_error += np.abs(grad_params[index] - grad)
        # Debug grad check
        print "grad check error: %e\n" % (grad_error / float(count))



def get_data():
    """
    load data by local
    """
    with gzip.open('mnist.pkl.gz', 'rb') as data_file:
        train_set, valid_set, test_set = pickle.load(data_file)
    return train_set, valid_set, test_set


def get_data_range(data_set, num_range=10):
    """
    get data range
    """
    data_x = data_set[0]
    data_y = data_set[1]

    idx = np.where(data_y < num_range)
    data_x = data_x[idx, :]
    data_y = data_y[idx]
    data_x.shape = (data_x.shape[1], data_x.shape[2])
    data_y.shape = (data_y.shape[0])
    return data_x.T, data_y



def main():
    """
    main function
    """
    num_range = 2
    # Load the dataset
    train_set, valid_set, _ = get_data()

    train_x, train_y = get_data_range(train_set, num_range=num_range)
    valid_x, valid_y = get_data_range(valid_set, num_range=num_range)
    print "size x: %s, y: %s" % (train_x.shape, train_y.shape)

    # n_classes = np.unique(train_y).size
    model = NeutraNetwork(
        [
            LinearLayer(64, Activation('relu')),
            # LinearLayer(64, Activation('relu')),
            LinearLayer(32, Activation('relu')),
            # LinearLayer(32, Activation('relu')),
            LinearLayer(num_range, Activation('sigmoid')),
        ],
        MSECostLayer()
    )

    # model.grad_check(valid_x[:, : 1000], valid_y[:1000])

    # Train neural network
    print 'Training neural network'
    model.train(train_x, train_y, num_epochs=50, batch_size=32, learning_rate=0.1)

    # Evaluate on training data
    error = model.error(valid_x, valid_y)
    print 'valid error rate: %.4f' % error


if __name__ == '__main__':
    main()
