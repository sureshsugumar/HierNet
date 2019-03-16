from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from six.moves import xrange
from tensorflow.contrib import slim
from tqdm import tqdm
from utils import squash, imshow_noax
from config import cfg


class CapsNet(object):

    def __init__(self):
        """
        initialize the capsule, the basic structure of a hierarchical network
      
        """
        # keep tracking of the dimension of feature maps
        self._dim = 28

        # store number of capsules of each capsule layer
        # the conv1-layer has 0 capsules
        self._num_caps = [0]
        
        # set for counting
        self._count = 0
        
        # set up placeholder of input data and labels
        self._x = tf.placeholder(tf.float32, [None, 784])
        self._y_ = tf.placeholder(tf.float32, [None, 10])
        
        # set up initializer for weights and bias
        self._w_initializer = tf.truncated_normal_initializer(stddev=0.1)
        self._b_initializer = tf.zeros_initializer()
        
    def _capsule(self, input, i_c, o_c, idx):
        """
        compute a capsule,
        conv op with kernel: 9x9, stride: 2,
        padding: VALID, output channels: 8 per capsule.
        :arg
            input: input for computing capsule, shape: [None, w, h, c]
            i_c: input channels
            o_c: output channels
            idx: index of the capsule about to create

        :return
            capsule: computed capsule
        """
        with tf.variable_scope('cap_' + str(idx)):
            w = tf.get_variable('w', shape=[9, 9, i_c, o_c], dtype=tf.float32)
            cap = tf.nn.conv2d(input, w, [1, 2, 2, 1],
                               padding='VALID', name='cap_conv')
            if cfg.USE_BIAS:
                b = tf.get_variable('b', shape=[o_c, ], dtype=tf.float32,
                                    initializer=self._b_initializer)
                cap = cap + b
            # cap with shape [None, 6, 6, 8] for mnist dataset
            # use "squash" as its non-linearity.
            capsule = squash(cap)
            
            # capsule with shape: [None, 6, 6, 8]
            # expand the dimensions to [None, 1, 6, 6, 8] for following concat
            capsule = tf.expand_dims(capsule, axis=1)

            # return capsule with shape [None, 1, 6, 6, 8]
            return capsule

    def _dynamic_routing(self, primary_caps, layer_index):
        """"
        dynamic routing between capsules
        :arg
            primary_caps: primary capsules with shape [None, 1, 32 x 6 x 6, 1, 8]
            layer_index: index of the current capsule layer, i.e. the input layer for routing
            
        :return
            digit_caps: the output of digit capsule layer output, with shape: [None, 10, 16]
        """
        # number of the capsules in current layer
        num_caps = self._num_caps[layer_index]
        
        # weight matrix for capsules in "layer_index" layer
        # W_ij
        cap_ws = tf.get_variable('cap_w', shape=[10, num_caps, 8, 16],
                                 dtype=tf.float32,
                                 )
        
        # initial value for "tf.scan", see official doc for details
        fn_init = tf.zeros([10, num_caps, 1, 16])

        # x after tiled with shape: [10, num_caps, 1, 8]
        # cap_ws with shape: [10, num_caps, 8, 16],
        # [8 x 16] for each pair of capsules between two layers
        # u_hat_j|i = W_ij * u_i
        cap_predicts = tf.scan(lambda ac, x: tf.matmul(x, cap_ws),
                               tf.tile(primary_caps, [1, 10, 1, 1, 1]),
                               initializer=fn_init, name='cap_predicts')
        
        # cap_predicts with shape: [None, 10, num_caps, 1, 16]
        cap_predictions = tf.squeeze(cap_predicts, axis=[3])
        
        # after squeeze with shape: [None, 10, num_caps, 16]
        # log prior probabilities
        log_prior = tf.get_variable('log_prior', shape=[10, num_caps], dtype=tf.float32,
                                    initializer=tf.zeros_initializer(),
                                    trainable=cfg.PRIOR_TRAINING)
        # log_prior with shape: [10, num_caps]
        
        digit_caps = self._routing(log_prior, cap_predictions, num_caps)

        # return digit capsule layer with shape: [None, 10, 16]
        return digit_caps

    def _routing(self, prior, cap_predictions, num_caps):
        """
        doing dynamic routing with tf.while_loop
        :arg
            prior: log prior for scaling with shape [10, num_caps]
            cap_prediction: predictions from layer below with shape [None, 10, num_caps, 16]
            num_caps: num_caps
        :return
            digit_caps: digit capsules with shape [None, 10, 16]
        """
        init_cap = tf.reduce_sum(cap_predictions, -2)
        iters = tf.constant(cfg.ROUTING_ITERS)
        prior = tf.expand_dims(prior, 0)

        def body(i, prior, cap_out):
            c = tf.nn.softmax(prior, axis=1)
            c_expand = tf.expand_dims(c, axis=-1)
            s_t = tf.multiply(cap_predictions, c_expand)
            s = tf.reduce_sum(s_t, axis=[2])
            cap_out = squash(s)
            delta_prior = tf.reduce_sum(tf.multiply(tf.expand_dims(cap_out, axis=2),
                                                    cap_predictions),
                                        axis=[-1])
            prior = prior + delta_prior

            return [i - 1, prior, cap_out]

        condition = lambda i, proir, cap_out: i > 0
        _, prior, digit_caps = tf.while_loop(condition, body, [iters, prior, init_cap],
                                             shape_invariants=[iters.get_shape(),
                                                               tf.TensorShape([None, 10, num_caps]),
                                                               init_cap.get_shape()])

        # return digit capsules with shape [None, 10, 16]
        return digit_caps


    def _reconstruct(self, target_cap):
        """
        reconstruct from digit capsules with 3 fully connected layer
        :param
            digit_caps: digit capsules with shape [None, 16]
        :return:
            out: out of reconstruction
        """
        print('Inside reconstruct') 
        with tf.name_scope('reconstruct'):
            fc = slim.fully_connected(target_cap, 512,
                                      weights_initializer=self._w_initializer)
            fc = slim.fully_connected(fc, 1024,
                                      weights_initializer=self._w_initializer)
            fc = slim.fully_connected(fc, 784,
                                      weights_initializer=self._w_initializer,
                                      activation_fn=None)
            # the last layer with sigmoid activation
            out = tf.sigmoid(fc)
            # out with shape [None, 784]

            self._recons_img = out
            return out

    def _add_loss(self):
        """
        add the margin loss and reconstruction loss
        :arg
            digit_caps: output of digit capsule layer, shape [None, 10, 16]
        :return
            total_loss:
        """
        with tf.name_scope('loss'):
            # loss of positive classes
            # max(0, m+ - ||v_c||) ^ 2
            with tf.name_scope('pos_loss'):
                pos_loss = tf.maximum(0., cfg.M_POS - tf.reduce_sum(self._digit_caps_norm * self._y_,
                                                                    axis=1), name='pos_max')
                pos_loss = tf.square(pos_loss, name='pos_square')
                pos_loss = tf.reduce_mean(pos_loss)
            tf.summary.scalar('pos_loss', pos_loss)
            # pos_loss shape: [None, ]

            # get index of negative classes
            y_negs = 1. - self._y_
            # max(0, ||v_c|| - m-) ^ 2
            with tf.name_scope('neg_loss'):
                neg_loss = tf.maximum(0., self._digit_caps_norm * y_negs - cfg.M_NEG)
                neg_loss = tf.reduce_sum(tf.square(neg_loss), axis=-1) * cfg.LAMBDA
                neg_loss = tf.reduce_mean(neg_loss)
            tf.summary.scalar('neg_loss', neg_loss)
            # neg_loss shape: [None, ]

            # Use the target capsule with dimension [None, 16] or [16,] (use it for default)
            y_ = tf.expand_dims(self._y_, axis=2)
            # y_ shape: [None, 10, 1]

            target_cap = y_ * self._digit_caps
            # target_cap shape: [None, 10, 16]
            target_cap = tf.reduce_sum(target_cap, axis=1)
            # target_cap: [None, 16]

            reconstruct = self._reconstruct(target_cap)

            # loss of reconstruction
            with tf.name_scope('l2_loss'):
                reconstruct_loss = tf.reduce_sum(tf.square(self._x - reconstruct), axis=-1)
                reconstruct_loss = tf.reduce_mean(reconstruct_loss)
            tf.summary.scalar('reconstruct_loss', reconstruct_loss)

            total_loss = pos_loss + neg_loss + \
                         cfg.RECONSTRUCT_W * reconstruct_loss

            tf.summary.scalar('loss', total_loss)
            
        # return total_loss
        return total_loss

    def train_architecture(self):
        """
        create architecture of the whole network
        arg:
            self
        return:
            none
        """
        print('Inside train_architecture') 
        with tf.variable_scope('CapsNet', initializer=self._w_initializer):
            # build net
            self._build_net()

            # set up losses
            self._loss = self._add_loss()

            # set up exponentially decay learning rate
            self._global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(cfg.LR, self._global_step,
                                                       cfg.STEP_SIZE, cfg.DECAY_RATIO,
                                                       staircase=True)
            tf.summary.scalar('learning rate', learning_rate)

            # set up adam optimizer with default setting
            self._optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients = self._optimizer.compute_gradients(self._loss)
            tf.summary.scalar('grad_norm', tf.global_norm(gradients))

            self._train_op = self._optimizer.apply_gradients(gradients,
                                                             global_step=self._global_step)
            # set up accuracy ops
            self._accuracy()
            # set up summary op
            self._summary_op = tf.summary.merge_all()
            # create a saver
            self.saver = tf.train.Saver()

            # set up summary writer
            self.train_writer = tf.summary.FileWriter(cfg.TB_DIR + '/train')
            self.val_writer = tf.summary.FileWriter(cfg.TB_DIR + '/val')

    def _build_net(self):
        """
        build the graph of the network
        arg:
            self
        return:
            none        
        """
        # reshape for conv ops
        with tf.name_scope('x_reshape'):
            x_image = tf.reshape(self._x, [-1, 28, 28, 1])

        # initial conv1 op
        # 1). conv1 with kernel 9x9, stride 1, output channels 256
        with tf.variable_scope('conv1'):
            # specially initialize it with xavier initializer with no good reason.
            w = tf.get_variable('w', shape=[9, 9, 1, 256], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer()
                                )
            # conv op
            conv1 = tf.nn.conv2d(x_image, w, [1, 1, 1, 1],
                                 padding='VALID', name='conv1')
            if cfg.USE_BIAS:
                b = tf.get_variable('b', shape=[256, ], dtype=tf.float32,
                                    initializer=self._b_initializer)
                conv1 = tf.nn.relu(conv1 + b)
            else:
                conv1 = tf.nn.relu(conv1)

            # update dimensions of feature map
            self._dim = (self._dim - 9) // 1 + 1
            assert self._dim == 20, "after conv1, dimensions of feature map" \
                                    "should be 20x20"

            # conv1 with shape [None, 20, 20, 256]

        # build up primary capsules
        with tf.variable_scope('PrimaryCaps'):

            # update dim of capsule grid
            self._dim = (self._dim - 9) // 2 + 1
            # number of primary caps: 6x6x32 = 1152
            self._num_caps.append(self._dim ** 2 * cfg.PRIMARY_CAPS_CHANNELS)
            assert self._dim == 6, "dims for primary caps grid should be 6x6."

            # build up PrimaryCaps with 32 channels and 8-D vector
            primary_caps = slim.conv2d(conv1, 32 * 8, 9, 2, padding='VALID', activation_fn=None)
            primary_caps = tf.reshape(primary_caps, [-1, 1, self._num_caps[1], 1, 8])
            primary_caps = squash(primary_caps)

        # dynamic routing
        with tf.variable_scope("digit_caps"):
            self._digit_caps = self._dynamic_routing(primary_caps, 1)

            self._digit_caps_norm = tf.norm(self._digit_caps, ord=2, axis=2,
                                            name='digit_caps_norm')
            # digit_caps_norm shape: [None, 10]
            
    def _accuracy(self):
        """
        set up accuracy
        arg:
            self
        return:
            none        
        """
        with tf.name_scope('accuracy'):
            self._py = tf.argmax(self._digit_caps_norm, 1)
            correct_prediction = tf.equal(tf.argmax(self._y_, 1),
                                          self._py)
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)
            tf.summary.scalar('accuracy', self.accuracy)

    def get_summary(self, sess, data):
        """
        get training summary
        arg:
            self, data
        return:
            accuracy, summary        
        """
        acc, summary = sess.run([self.accuracy, self._summary_op],
                                feed_dict={self._x: data[0],
                                           self._y_: data[1]})

        return acc, summary

    def train(self, sess, data):
        """training process
        arg:
            data: (images, labels)
        return:
            loss
        """
        loss, _ = sess.run([self._loss, self._train_op],
                           feed_dict={self._x: data[0],
                                      self._y_: data[1]})

        return loss

    def snapshot(self, sess, iters):
        """
        save checkpoint
        arg:
            self
        return:
            none        
        """
        save_path = cfg.TRAIN_DIR + '/capsnet'
        self.saver.save(sess, save_path, iters)

    def test(self, sess, mnist, set='validation'):
        """
        test trained model on specific dataset split
        arg:
            self, mnist
        return:
            none        
        """
        tic = time.time()
        if set == 'test':
            x = mnist.test.images
            y_ = mnist.test.labels
        elif set == 'validation':
            x = mnist.validation.images
            y_ = mnist.validation.labels
        elif set == 'train':
            x = mnist.train.images
            y_ = mnist.train.labels
        else:
            raise ValueError

        acc = []
        for i in tqdm(xrange(len(x) // 100), desc="calculating %s accuracy" % set):
            x_i = x[i * 100: (i + 1) * 100]
            y_i = y_[i * 100: (i + 1) * 100]
            ac = sess.run(self.accuracy,
                          feed_dict={self._x: x_i, self._y_: y_i})
            acc.append(ac)

        all_ac = np.mean(np.array(acc))
        t = time.time() - tic
        print("{} set accuracy: {}, with time: {:.2f} secs".format(set, all_ac, t))

    def reconstruct_eval(self, sess, x, y, batch_size):
        """
        do image reconstruction and representations
        arg:
            self, image, label, batch size
        return:
            none        
        """
        ori_img = x
        label = np.argmax(y, axis=1)
        res_img, res_label, acc, norms = sess.run([self._recons_img, self._py, self.accuracy,
                                                   self._digit_caps_norm],
                                                   feed_dict={self._x: x,
                                                             self._y_: y})
        if acc < 1:
            ori_img = np.reshape(ori_img, [batch_size, 28, 28])
            res_img = np.reshape(res_img, [batch_size, 28, 28])
            num_rows = int(np.ceil(batch_size / 10))
            fig, _ = plt.subplots(nrows=2 * num_rows, ncols=10, figsize=(10, 7))
            for r in range(num_rows):
                for i in range(10):
                    idx = i + r * 10
                    if idx == batch_size:
                        break
                    plt.subplot(2 * num_rows, 10, idx + 1 + 10 * r)
                    imshow_noax(ori_img[idx])
                    plt.title(label[idx])
                    plt.subplot(2 * num_rows, 10, idx + 11 + 10 * r)
                    imshow_noax(res_img[idx])
                    plt.title("%s_%.3f" % (res_label[idx], norms[idx][res_label[idx]]))

            self._count += 1
            plt.savefig(self._fig_dir + '/%s.png' % self._count, dpi=200)

    def eval_architecture(self, mode, fig_dir):
        """
        evaluate architecture
        arg:
            self, mode
        return:
            none        
        """
        with tf.variable_scope('CapsNet', initializer=self._w_initializer):
            self._build_net()
            self._adver_loss = tf.reduce_sum(self._digit_caps_norm * self._y_)
            grads = tf.gradients(self._adver_loss, self._x)
            self._adver_gradients = grads / tf.norm(grads)
            self._py = tf.argmax(self._digit_caps_norm, 1)
        
            self._fig_dir = fig_dir + '/%s' % mode
            if not os.path.exists(self._fig_dir):
                os.makedirs(self._fig_dir)
                
            self.saver = tf.train.Saver()
      
    def test_eval(self, sess, img, lr=1):
        """
        evaluation test
        arg:
            self, image
        return:
            prediction, probablity lists
        """
        ori_img = np.reshape(img, [28, 28])
        x_copy = img.copy()
        py = [None]
        tar_oh = np.zeros(10, dtype=np.float32)
        tar_oh[9] = 1
        self._count = 0
        grads, py = sess.run([self._adver_gradients, self._py],
                                    feed_dict={self._x: x_copy,
                                    self._y_: tar_oh[None, :]})
        x_copy += lr * grads[0]
        self._count += 1
        print("predicted: {}".format(py[0]))
        
        
        self._py1 = tf.math.top_k(self._digit_caps_norm, 10)
        grads, py1, acc1 = sess.run([self._adver_gradients, self._py1, self._y_],
                                    feed_dict={self._x: x_copy,
                                    self._y_: tar_oh[None, :]})

        print("probability: {0:.0%}".format(py1[0][0][0]))
        print("all predictions: {}".format(py1[1][0]))
        print("all probabilities: {}".format(py1[0][0]))
        
        x_copy = np.reshape(x_copy, [28, 28])
        fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(6, 2))
        plt.subplot(1, 3, 1)
        imshow_noax(ori_img)
        plt.title('predicted: %s' % format(py))
        plt.savefig(self._fig_dir + '/%s' % format(py))
        plt.show()

        return py, py1
        
