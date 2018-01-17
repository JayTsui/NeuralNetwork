#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.misc import imread, imresize
from layer import ConvolutionLayer, Activation
from matplotlib import pyplot as plt


def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')


def main():
    """
    main function
    """
    kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
    dist = kitten.shape[1] - kitten.shape[0]
    kitten_cropped = kitten[:, dist/2:-dist/2, :]

    img_size = 200
    input_x = np.zeros((2, 3, img_size, img_size))
    input_x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
    input_x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))

    w_shape = (2, 3, 3, 3)
    param_w = np.zeros(w_shape)
    param_w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
    param_w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
    param_w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]

    param_w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
    param_w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
    param_w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]
    # Second filter detects horizontal edges in the blue channel.
    param_w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    conv = ConvolutionLayer(w_shape, Activation('same'))
    conv.setup(input_x.shape)
    conv.conv_w = param_w

    out = conv.forward(input_x)

    plt.subplot(2, 3, 1)
    imshow_noax(puppy, normalize=False)
    plt.title('Original image')
    plt.subplot(2, 3, 2)
    imshow_noax(out[0, 0])
    plt.title('Grayscale')
    plt.subplot(2, 3, 3)
    imshow_noax(out[0, 1])
    plt.title('Edges')
    plt.subplot(2, 3, 4)
    imshow_noax(kitten_cropped, normalize=False)
    plt.subplot(2, 3, 5)
    imshow_noax(out[1, 0])
    plt.subplot(2, 3, 6)
    imshow_noax(out[1, 1])
    plt.show()

if __name__ == '__main__':
    main()
