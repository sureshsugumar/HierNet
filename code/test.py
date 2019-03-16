from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TensorFlow and keras
import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import plot_model
from keras.models import load_model
from keras.models import Model

# Commonly used modules
import numpy as np
import os
import sys
from utils import preprocess_images, imshow_noax

# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import IPython

from six.moves import urllib, xrange
from CapsNet import CapsNet
from config import cfg
from scipy import ndimage
from utils import preprocess_images
plt.rcParams.update({'figure.max_open_warning': 0})

def main():

    # Import data
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # reshape images to specify that it's a single channel
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    train_images = preprocess_images(train_images)
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

        mnist_in_path = '../videos/mnist_in.mp4'
        mnist_out_path = '../videos/mnist_caps_out.mp4'

        cap = cv2.VideoCapture(mnist_in_path)
        vw = None
        frame = -1  # counter for debugging (mostly), 0-indexed

        # go through all the frames and run our classifier on the high res MNIST images as they morph from number to number
        while True:  # should 481 frames
            frame += 1
            ret, img = cap.read()
            if not ret: break

            img = cv2.resize(img, (720, 720))

            # preprocess the image for prediction
            img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_proc = cv2.resize(img_proc, (28, 28))
            img_proc = preprocess_images(img_proc)
            img_proc = 1 - img_proc  # inverse since training dataset is white text with black background

            net_in = np.expand_dims(img_proc, axis=0)  # expand dimension to specify batch size of 1
            net_in = np.expand_dims(net_in, axis=3)  # expand dimension to specify number of channels

            img_t = img_proc.copy()
            img_t = np.reshape(img_t, [28, 28])
            img_t = np.reshape(img_t, [1, 784])
            preds, perc = caps_net.test_eval(sess, img_t, 1)

            img = 255 - img
            pad_color = 0
            img = np.pad(img, ((0, 0), (0, 1280 - 720), (0, 0)), mode='constant', constant_values=(pad_color))

            line_type = cv2.LINE_AA
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.3
            thickness = 2
            x, y = 740, 60
            color = (255, 255, 255)

            text = "HNN Output:"
            cv2.putText(img, text=text, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                    color=color, lineType=line_type)
            text = "Input:"
            cv2.putText(img, text=text, org=(30, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                    color=color, lineType=line_type)

            guess = perc[1][0][0]
            
            # sort the lists
            p1 = perc[1][0]
            p2 = perc[0][0]
            ziplist = sorted(zip(p1, p2))
            p3, p4 = zip(*ziplist) 
                                 
            y = 130
            #for i, p in enumerate(perc[0]):
            for i in range(0, 10):
                pred = p3[i]
                prob = p4[i]
                              
                p = np.rint(prob * 100).astype(int)
                if pred == guess:
                    #color = (255, 218, 158)
                    color = (25, 200, 25)
                else:
                    color = (100, 100, 100)

                rect_width = 0
                if p > 0: rect_width = int(p * 3.3)

                rect_start = 180
                cv2.rectangle(img, (x + rect_start, y - 5), (x + rect_start + rect_width, y - 20), color, -1)

                text = '{}: {:>3}%'.format(i, int(p))
                cv2.putText(img, text=text, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness,
                        color=color, lineType=line_type)
                y += 60
    
            # if you don't want to save the output as a video, set this to False
            save_video = True

            if save_video:
                if vw is None:
                    codec = cv2.VideoWriter_fourcc(*'mp4v')
                    vid_width_height = img.shape[1], img.shape[0]
                    vw = cv2.VideoWriter(mnist_out_path, codec, 30, vid_width_height)
                # 15 fps above doesn't work robustly so we write frame twice at 30 fps
                vw.write(img)
                vw.write(img)


    cap.release()
    if vw is not None:
        vw.release()
        
if __name__ == '__main__':
    main()