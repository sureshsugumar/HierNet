from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def preprocess_images(imgs):
    """
    pre-condition the images for consumption by network
    arg:
        images
    return:
        images
    """
    img = imgs if len(imgs.shape) == 2 else imgs[0]
    assert img.shape in [(28, 28, 1), (28, 28)], img.shape
    return imgs / 255.0


def squash(cap_input):
    """
    squash function for keeping the length of capsules between 0 - 1
    arg:
        cap_input: total input of capsules, with shape: [None, h, w, c] or [None, n, d]
    return:
        cap_output: output of each capsules, which has the shape as cap_input
    """

    with tf.name_scope('squash'):
        # compute norm square of inputs with the last axis, keep dims for broadcasting
        # ||s_j||^2
        input_norm_square = tf.reduce_sum(tf.square(cap_input), axis=-1, keepdims=True)

        # ||s_j||^2 / (1. + ||s_j||^2) * (s_j / ||s_j||)
        scale = input_norm_square / (1. + input_norm_square) / tf.sqrt(input_norm_square)

    return cap_input * scale


def imshow_noax(img, nomalize=True):
    """
    show image by plt with axis off
    arg:
        image
    return:
        none
    """

    if nomalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255. * (img - img_min) / (img_max - img_min)

    plt.imshow(img.astype('uint8'), cmap='gray')
    plt.gca().axis('off')


