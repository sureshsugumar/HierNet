from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data
from CapsNet import CapsNet
from config import cfg
from utils import preprocess_images


def main():

    # Import data
    mnist = input_data.read_data_sets(cfg.DATA_DIR, one_hot=True)

    tf.reset_default_graph()

    # Create the model
    caps_net = CapsNet()
    
    # build up architecture
    caps_net.train_architecture()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # read check point file
    ckpt = tf.train.get_checkpoint_state(cfg.TRAIN_DIR)

    with tf.Session(config=config) as sess:
        if ckpt:
            print("Reading parameters from %s" % ckpt.model_checkpoint_path)
            caps_net.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created model with fresh paramters.')
            sess.run(tf.global_variables_initializer())
            print('Num params: %d' % sum(v.get_shape().num_elements()
                                         for v in tf.trainable_variables()))

        caps_net.train_writer.add_graph(sess.graph)

        tic = time.time()
        for iters in range(cfg.MAX_ITERS):
            sys.stdout.write('>>> %d / %d \r' % (iters % cfg.PRINT_EVERY, cfg.PRINT_EVERY))
            sys.stdout.flush()

            data_in = mnist.train.next_batch(cfg.BATCH_SIZE)
            loss = caps_net.train(sess, data_in)

            if iters % cfg.PRINT_EVERY == 0 and iters > 0:
                train_acc, train_summary = caps_net.get_summary(sess, data_in)
                caps_net.train_writer.add_summary(train_summary, iters)
                caps_net.train_writer.flush()

                print("iters: %d / %d, loss ==> %.4f " % (iters, cfg.MAX_ITERS, loss))
                print('train accuracy: %.4f' % train_acc)

                val_batch = mnist.validation.next_batch(cfg.BATCH_SIZE)
                test_acc, test_summary = caps_net.get_summary(sess, val_batch)
                print('val accuracy: %.4f' % test_acc)
                caps_net.val_writer.add_summary(test_summary, iters)
                caps_net.val_writer.flush()
                toc = time.time()
                print('average time: %.2f secs' % (toc - tic))
                tic = time.time()

            if iters % cfg.SAVE_EVERY == 0 and iters > 0:
                caps_net.snapshot(sess, iters=iters)
                caps_net.test(sess, mnist, set='validation')

        caps_net.snapshot(sess, iters)
        caps_net.test(sess, mnist, 'test')

if __name__ == '__main__':
    main()
    