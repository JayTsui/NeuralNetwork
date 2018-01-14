#!/usr/bin/env python
# coding: utf-8

import numpy as np
from helper import relu, relu_grad, sigmoid, sigmoid_grad, tanh, tanh_grad, softmax, softmax_grad



class CostLayer(object):

    """
    Cost Layer Interface
    """

    def loss(self, labels, predicts):
        """
        loss function
        """
        raise NotImplementedError()

    def grad(self, labels, predicts):
        """
        grad function
        """
        raise NotImplementedError()



class MSECostLayer(CostLayer):

    """
    MSE Cost Layer
    """

    def loss(self, labels, predicts):
        """
        loss function
        """
        return np.sum((labels - predicts) ** 2) / (2.0 * predicts.shape[1])


    def grad(self, labels, predicts):
        """
        grad function
        """
        return predicts - labels


class SoftMaxCostLayer(CostLayer):

    """
    Softmax Cost Layer
    """

    def loss(self, labels, predicts):
        """
        loss function
        """
        return 0 - np.sum(np.sum(labels * np.log(predicts), axis=0))


    def grad(self, labels, predicts):
        """
        grad function
        """
        return predicts - labels



class Activation(object):

    """
    Activation Layer
    relu tanh sigmoid
    """

    def __init__(self, func='relu'):
        func_map = {
            "relu": (relu, relu_grad),
            "tanh": (tanh, tanh_grad),
            "sigmoid": (sigmoid, sigmoid_grad),
            "softmax": (softmax, softmax_grad)
        }
        self.fun, self.fun_d = func_map[func]



class CoverLayer(object):

    """
    Base Layer Interface
    """

    def setup(self, input_shape, learning_rate):
        """
        setup function
        """
        raise NotImplementedError()


    def forward(self, input_x):
        """
        forward function
        """
        raise NotImplementedError()


    def backward(self, input_grad):
        """
        backward function
        """
        raise NotImplementedError()



class LinearLayer(object):

    """
    Linear Layer
    """

    def __init__(self, n_out, act_fun):
        self.n_out = n_out
        self.act_fun = act_fun
        self.param_w = None
        self.param_b = None
        self.cache_x = None
        self.cache_z = None
        self.grad_w = None
        self.grad_b = None


    def setup(self, input_shape):
        """
        setup function
        """
        self.param_w = np.random.rand(self.n_out, input_shape) * .01
        # self.param_b = np.random.rand(self.n_out, 1) * .01
        self.param_b = np.zeros((self.n_out, 1))


    def forward(self, input_x):
        """
        forward function
        """
        self.cache_x = input_x
        self.cache_z = np.dot(self.param_w, input_x) + self.param_b
        return self.act_fun.fun(self.cache_z)


    def backward(self, input_grad):
        """
        backward function
        """
        grad_z = input_grad * self.act_fun.fun_d(self.cache_z)
        grad_w = np.dot(grad_z, self.cache_x.T) / input_grad.shape[1]
        grad_b = np.mean(grad_z, axis=1, keepdims=True)

        self.grad_w = grad_w
        self.grad_b = grad_b
        return np.dot(self.param_w.T, grad_z)



class ConvolutionLayer(CoverLayer):

    """
    Conv Layer
    """

    def __init__(self, stride, padding, act_fun):
        self.stride = stride
        self.padding = padding
        self.act_fun = act_fun
        self.conv_w = None
        self.conv_b = None
        self.learning_rate = None
        self.cache_x = None
        self.cache_z = None
        self.cache_pad_h = 0
        self.cache_pad_w = 0


    def setup(self, input_shape, learning_rate):
        """
        setup function
        """
        self.conv_w = np.random.rand(input_shape)
        self.conv_b = np.zeros((1, input_shape[0]))
        self.learning_rate = learning_rate


    def calc_padding(self, input_shape, kernel_shape):
        """
        calculate padding
        """
        _, _, input_h, input_w = input_shape
        _, _, kernel_h, kernel_w = kernel_shape
        rows = ((self.stride - 1) * input_h - self.stride + kernel_h) / 2
        cols = ((self.stride - 1) * input_w - self.stride + kernel_w) / 2
        return rows, cols


    def padding_input(self, input_x):
        """
        padding input
        """
        input_n, channel, rows, cols = input_x.shape
        if self.padding == "SAME":
            pad_h, pad_w = self.calc_padding(input_x.shape, self.conv_w.shape)
            # Cache padding
            self.cache_pad_h = pad_h
            self.cache_pad_w = pad_w
            res_xp = np.zeros(input_n, channel, rows + pad_h, cols + pad_w)
            res_xp[:, :, pad_h: rows + pad_h, pad_w: cols + pad_w] = input_x
            output = np.zeros(input_x.shape)
        else:
            res_xp = input_x
            conv_w_n, _, conv_w_rows, conv_w_cols = self.conv_w[2:]
            out_h = (rows - conv_w_rows) / self.stride + 1
            out_w = (cols - conv_w_cols) / self.stride + 1
            output = np.zeros((input_n, conv_w_n, out_h, out_w))
        return res_xp, output


    def forward(self, input_x):
        """
        convolution forward function
        """
        self.cache_x, output = self.padding_input(input_x)

        input_rows, input_cols = self.conv_w.shape[2:]
        conv_w_n, _, conv_w_rows, conv_w_cols = self.conv_w.shape

        for f in range(conv_w_n):
            for row in range(input_rows):
                for col in range(input_cols):
                    begin_h = self.stride * row
                    begin_w = self.stride * col
                    end_h = begin_h + conv_w_rows
                    end_w = begin_w + conv_w_cols
                    window_x = self.cache_x[:, f, begin_h: end_h, begin_w: end_w]
                    output[:, f, row, col] = np.sum(window_x * self.conv_w[f], axis=[1, 2, 3])
            output[:, f, :, :] += self.conv_b[f]
        self.cache_z = output
        return self.act_fun.fun(output)


    def backward(self, input_grad):
        """
        convolution backward function
        """
        grad_z = input_grad * self.act_fun.fun_d(self.cache_z)

        output_n, _, output_rows, output_cols = self.cache_z.shape
        conv_w_n, _, conv_w_rows, conv_w_cols = self.conv_w.shape

        grad_x = np.zeros(self.cache_x.shape)
        grad_w = np.zeros(self.conv_w.shape)
        grad_b = np.sum(grad_z, axis=[0, 2, 3])

        for i in range(output_n):
            for row in range(output_rows):
                for col in range(output_cols):
                    begin_h = self.stride * row
                    begin_w = self.stride * col
                    end_h = begin_h + conv_w_rows
                    end_w = begin_w + conv_w_cols
                    window_x = self.cache_x[i, :, begin_h: end_h, begin_w: end_w]
                    grad_w += window_x * grad_z

                    for f in range(conv_w_n):
                        grad_x[i, :, begin_h: end_h, begin_w: end_w] += self.conv_w[f] * grad_z

        self.conv_w -= self.learning_rate * grad_w
        self.conv_b -= self.learning_rate * grad_b
        return grad_x[:, :, self.cache_pad_h: output_rows, self.cache_pad_w: output_cols]
