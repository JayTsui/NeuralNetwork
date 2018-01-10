#!/usr/bin/env python
# coding: utf-8

import numpy as np


def sigmoid(input_x):
    """
    sigmoid
    """
    return 1.0 / (1.0 + np.exp(-input_x))


def sigmoid_grad(input_x):
    """
    sigmoid grad
    """
    res = sigmoid(input_x)
    return res * (1 - res)


def tanh(input_x):
    """
    tanh
    """
    return np.tanh(input_x)


def tanh_grad(input_x):
    """
    tanh grad
    """
    res = tanh(input_x)
    return 1 - res ** 2


def relu(input_x):
    """
    relu
    """
    return np.maximum(0.0, input_x)


def relu_grad(input_x):
    """
    relu grad
    """
    grad = np.zeros(input_x.shape)
    grad[input_x >= 0.0] = 1
    return grad


def one_hot(labels, n_classes):
    """
    encode one hot
    """
    classes = np.unique(labels)
    # n_classes = classes.size
    one_hot_labels = np.zeros((n_classes,) + labels.shape)
    for _cls in classes:
        one_hot_labels[_cls, labels == _cls] = 1
    return one_hot_labels


def unhot(one_hot_labels):
    """
    decode one hot
    """
    return np.argmax(one_hot_labels, axis=0)
