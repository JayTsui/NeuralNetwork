#!/usr/bin/env python
# coding: utf-8

"""
test unit
"""

import numpy as np
from layer import PoolLayer, FlattenLayer, ConvolutionLayer, Activation


def pool_layer():
    """
    PoolLayer
    """
    input_x = np.arange(0, 64).reshape((2, 2, 4, 4))
    pool = PoolLayer((2, 2))
    output = pool.forward(input_x)
    # print(input_x)
    # print(output)
    # print(pool.backward(output))


def flatten_layer():
    """
    FlattenLayer
    """
    input_x = np.arange(0, 64).reshape((2, 2, 4, 4))
    flatten = FlattenLayer()
    output = flatten.forward(input_x)
    print input_x
    print output
    print flatten.backward(output)


def conv_layer():
    """
    ConvolutionLayer
    """
    x_shape = (2, 3, 4, 4)
    w_shape = (2, 3, 3, 3)
    input_x = np.ones(x_shape)
    conv = ConvolutionLayer(w_shape, Activation('same'))
    conv.setup(x_shape)
    print input_x
    output = conv.forward(input_x)
    print output.shape, output


def main():
    """
    main function
    """
    conv_layer()

if __name__ == '__main__':
    main()
