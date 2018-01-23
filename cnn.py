#!/usr/bin/env python
# coding: utf-8

"""
神经网络算法
"""

import gzip
import pickle
import numpy as np
from helper import one_hot
from layer import LinearLayer, PoolLayer, FlattenLayer, \
                    ConvolutionLayer, Activation, SoftMaxCostLayer

class ConvNeutraNetwork(object):

    """
    Neutral Network Model
    """

    def __init__(self, layers, cost_fun, learning_rate=0.001):
        self.n_classes = None
        self.layers = layers
        self.cost_fun = cost_fun
        self.learning_rate = learning_rate


    def setup(self, input_shape):
        """
        init layer
        """
        data_shape = input_shape
        for layer in self.layers:
            layer.setup(data_shape)
            data_shape = layer.get_output_shape()


    def gradient_descent(self, input_x, input_y):
        """
        grandient descent
        """
        for layer in self.layers:
            print("forward: %s" % layer.__class__.__name__)
            input_x = layer.forward(input_x)

        input_grad = self.cost_fun.grad(one_hot(input_y, self.n_classes), input_x)
        for layer in reversed(self.layers):
            input_grad = layer.backward(input_grad)

        for layer in self.layers:
            layer.param_w -= self.learning_rate * layer.grad_w
            layer.param_b -= self.learning_rate * layer.grad_b


    def train(self, input_x, labels, num_epochs=10, batch_size=32, model_file=None):
        """
        train function
        """
        self.setup(input_x.shape)
        if model_file:
            self.load_model(model_file)
        n_batch = input_x.shape[0] // batch_size

        num_now = 0
        while num_now < num_epochs:
            num_now += 1

            for i in range(n_batch):
                batch_begin = i * batch_size
                batch_end = batch_begin + batch_size
                batch_input_y = labels[batch_begin: batch_end]
                batch_input_x = input_x[:, batch_begin: batch_end]
                self.gradient_descent(batch_input_x, batch_input_y)

            v_loss = self.loss(input_x, labels)
            v_error = self.error(input_x, labels)
            print("iter: %i, loss %.4f, train error %.4f" % (num_now, v_loss, v_error))


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


    def save_model(self, filename):
        """
        save model
        """
        model_data = self.get_model_params()
        model_data = [str(d) for d in model_data]
        with open(filename, 'w') as _file:
            _file.write(" ".join(model_data))


    def load_model(self, filename):
        """
        load model
        """
        with open(filename, 'r') as _file:
            model_data = _file.read()
            model_data = np.array(model_data.split(), np.float)
            self.update_model_params(model_data)


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
            self.setup(input_x[1])

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
        print("grad check error: %e\n" % (grad_error / float(count)))



def get_data():
    """
    load data by local
    """
    with gzip.open('mnist.pkl.gz', 'rb') as data_file:
        train_set, valid_set, test_set = pickle.load(data_file)
    return train_set, valid_set, test_set


def get_data_range(data_set):
    """
    get data range
    """
    data_x = data_set[0]
    data_y = data_set[1]
    data_x = data_x.reshape((data_x.shape[0], 1, 28, 28))
    return data_x, data_y

def main():
    """
    main function
    """
    # Load the dataset
    train_set, valid_set, _ = get_data()

    train_x, train_y = get_data_range(train_set)
    valid_x, valid_y = get_data_range(valid_set)
    print("size x: %s, y: %s" % (train_x.shape, train_y.shape))

    # n_classes = np.unique(train_y).size
    model = ConvNeutraNetwork(
        [
            ConvolutionLayer((32, 1, 3, 3), Activation('relu')),
            PoolLayer((2, 2)),
            ConvolutionLayer((64, 1, 3, 3), Activation('relu')),
            PoolLayer((2, 2)),
            FlattenLayer(),
            LinearLayer(32, Activation('relu')),
            LinearLayer(10, Activation('softmax')),
        ],
        SoftMaxCostLayer(),
        learning_rate=0.05
    )

    # model.grad_check(valid_x[:, : 1000], valid_y[:1000])

    # Train neural network
    print('Training neural network')
    model.train(train_x[1000:], train_y[1000:], num_epochs=10, batch_size=32)

    # Evaluate on training data
    error = model.error(valid_x[100:], valid_y[100:])
    print('valid error rate: %.4f' % error)

    # model.save_model('data.rc')


if __name__ == '__main__':
    main()
