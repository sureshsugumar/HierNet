from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np
from six.moves import xrange
from config import cfg
from scipy import ndimage
from utils import preprocess_images
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

def main():
    
    # Import dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    test_images = preprocess_images(test_images)
    
    # Generate images for test
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        for i in range(0, 20):
            n = np.random.randint(low=1, high=100)
            x1 = test_images[n]
            print(test_labels[n])
            img_max, img_min = np.max(x1), np.min(x1)
            x1 = 255. * (x1 - img_min) / (img_max - img_min)
            filename = '../images/img'
            filename += str(i)
            filename += '.png'
            cv2.imwrite(filename, x1) 
            
        for i in range(20, 40):
            n = np.random.randint(low=1, high=100)
            m = np.random.randint(low=1, high=100)
            x1 = test_images[n]
            x2 = test_images[m]
            x3 = cv2.addWeighted(x1, 0.5, x2, 0.5, 0)
            print(test_labels[n], test_labels[m])
            img_max, img_min = np.max(x3), np.min(x3)
            x3 = 255. * (x3 - img_min) / (img_max - img_min)
            filename = '../images/img'
            filename += str(i)
            filename += '.png'
            cv2.imwrite(filename, x3) 

        for i in range(40, 60):
            n = np.random.randint(low=1, high=100)
            m = np.random.randint(low=1, high=100)
            l = np.random.randint(low=1, high=100)           
            x1 = test_images[n]
            x2 = test_images[m]
            x3 = test_images[l]
            x4 = cv2.addWeighted(x1, 0.5, x2, 0.5, 0)
            x5 = cv2.addWeighted(x4, 0.5, x3, 0.5, 0)           
            print(test_labels[n], test_labels[m], test_labels[l])
            img_max, img_min = np.max(x5), np.min(x5)
            x5 = 255. * (x5 - img_min) / (img_max - img_min)
            filename = '../images/img'
            filename += str(i)
            filename += '.png'
            cv2.imwrite(filename, x5)
            
        for i in range(60, 80):
            n = np.random.randint(low=1, high=100)
            angle = np.random.randint(low=-180, high=180)
            x6 = test_images[n]
            print(test_labels[n])
            x6 = np.reshape(x6, [28, 28])
            x6 = ndimage.rotate(x6, angle, reshape=False)
            img_max, img_min = np.max(x6), np.min(x6)
            x6 = 255. * (x6 - img_min) / (img_max - img_min)
            filename = '../images/img'
            filename += str(i)
            filename += '.png'
            cv2.imwrite(filename, x6)     
            
if __name__ == '__main__':
    main()
