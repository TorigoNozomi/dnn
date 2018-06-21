#Simple multi classifier

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data

import argparse
import sys
import scipy.io
import numpy as np
import tensorflow as tf
import pickle
import random
import os
import math
import csv

NDATA = 784
NCLASS = 10
NLAYERS = 5
NUNITS = [NDATA, 90, 80, 50, 30, NCLASS]
EPOCH = 100
MBATCH = 100
FLAGS = None
DATASET = 'MNIST'

def simplereturn(y):
    return y

def sigmoid(y):
    # Sigmoid function
    return 1/(1+tf.exp(-y))

def ReLU(y):
    # ReLU
    return tf.maximum(0.0, y)

ACTIVATION_FUNC = {'SimpleReturn':simplereturn, 'ReLU':ReLU,'sigmoid':sigmoid}
USE_FUNC = ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'SimpleReturn']

def main(self):
    # Import data
    file_in = '../datasets/MNIST_data/'
    mnist = input_data.read_data_sets(file_in, one_hot=True)
    W_b_in = '../datasets/W_b_init'
    with open(W_b_in, mode='rb') as f:
        W_b_init = pickle.load(f)
        for i in range(NLAYERS):
            exec('W' + str(i+1) + '_init = W_b_init[\'W' + str(i+1) + '_init\']')
            exec('b' + str(i+1) + '_init = W_b_init[\'b' + str(i+1) + '_init\']')

    print('%s loaded' % file_in)
    print('%s loaded' % W_b_in)

    # Create the model
    x = tf.placeholder(tf.float32, [None, NDATA])
    h_0 = x
    y = tf.placeholder(tf.float32, [None, NCLASS])

    for i in range(NLAYERS):
            exec('W_' + str(i+1) + ' = tf.Variable(np.resize(W' + str(i+1) + '_init, (NUNITS[' + str(i) + '], NUNITS[' + str(i+1) + '])))')
            exec('b_' + str(i+1) + ' = tf.Variable(np.resize(b' + str(i+1) + '_init, (NUNITS[' + str(i+1) + '])))')
            if i == (NLAYERS-1):
                exec('y = tf.matmul(h_' + str(
                    i) + ', W_' + str(i + 1) + ') + b_' + str(i + 1))
            else:
                exec('h_' + str(i+1) + ' = ACTIVATION_FUNC[USE_FUNC[' + str(i) + ']](tf.matmul(h_' + str(i) + ', W_' + str(i+1) + ') + b_' + str(i+1) + ')')


    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, NCLASS])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Train
    iter = int(55000/MBATCH*EPOCH)
    for i in range(iter):
        batch_xs, batch_ys = mnist.train.next_batch(MBATCH,shuffle=False)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # Display progress
        sys.stdout.write("\r%.3f%%" % (i / iter * 100))
        sys.stdout.flush()

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)