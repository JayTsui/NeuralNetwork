#!/usr/bin/env python
# coding: utf-8

"""
test unit
"""

import numpy as np
from layer import PoolLayer


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


def main():
    """
    main function
    """
    pool_layer()

if __name__ == '__main__':
    main()
