from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np
import matplotlib.pyplot as plt

from six.moves import xrange
from CapsNet import CapsNet
from config import cfg
from scipy import ndimage
from utils import preprocess_images
plt.rcParams.update({'figure.max_open_warning': 0})


def main():
    
    # Import data
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    test_images = preprocess_images(test_images)
 
    # Create the model
    caps_net = CapsNet()
    
    # Build up architecture
    caps_net.eval_architecture('test', cfg.FIG_DIR)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Read check point file
    ckpt = tf.train.get_checkpoint_state(cfg.TRAIN_DIR)
   
    # Test the network
    with tf.Session(config=config) as sess:
        caps_net.saver.restore(sess, ckpt.model_checkpoint_path) 
 
        for i in range(60):
            # Read images
            filename = '../images/img'
            filename += str(i)
            filename += '.png'
            img = cv2.imread(filename, 0).astype('float32')
            height, width = img.shape
            if (height != 720) | (width != 720):
                img = cv2.resize(img, (720, 720))

            img = cv2.resize(img, (28, 28))
            img = np.reshape(img, [28, 28])
            img = np.reshape(img, [1, 784])
            
            print('prediction for image ', filename)
            preds, perc = caps_net.test_eval(sess, img, 1)
            #print("predicted: {}".format(preds[0]))
            
if __name__ == '__main__':
    main()
