# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

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
import itertools

NLAYERS = 5
NCLASS = 10
NDATA = 784
NUNITS = [NDATA, 300, 150, 40, NCLASS]
EPOCH = 1000
MBATCH = 100
LEARNING_RATE = 0.05
DATASET = 'MNIST'
FLAGS = None
INIT = 'glorot3'
r = 1.0


def discretization(x, i):
    y = np.where(x <= r - 1, -np.sqrt(2 / (r * (NUNITS[i - 1] + NUNITS[i]))), x)
    z = np.where((y > r - 1) & (y <= (1 - r)), -np.sqrt(2 / (r * (NUNITS[i - 1] + NUNITS[i]))), y)
    return np.where(z <= 1 - r, -np.sqrt(2 / (r * (NUNITS[i - 1] + NUNITS[i]))), z) * 0.01



def simplereturn(y):
    return y

def step(y):
    # Step Function
    if tf.maximum(0.0, y) == 0:
        return 0.0
    else:
        return 1.0

def softsign(y):
    # softsign
    return tf.nn.softsign(y)

#def SeLU(y):
    # Self-Normalizing Neural Networks

def sigmoid(y):
    # Sigmoid function
    return 1 / (1 + tf.exp(-y))

def tanh(y):
    # Hyperbolic tangent
    return tf.nn.tanh(y)


def ReLU(y):
    # ReLU
    return tf.maximum(0.0, y)

def Approximated_ReLU(y):
    #Approximated_ReLU
    u = tf.exp(y)
    return tf.log1p(tf.clip_by_value(u, 1e-16, 1e16))

def Parametric_ReLU(y):
    # Parametric ReLU
    alpha = 0.01
    return tf.where(y >= 0.0, y, alpha * y)

def ELU(y):
    #Exponential Linear Units
    alpha = 1.0
    return tf.where(y >= 0.0, y, alpha * tf.exp(y) - alpha)

def weight_variable(shape):
    # Define initial weight by normal distribution
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # Define initial weight by normal distribution
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def tanhplus(y):
    # Hyperbolic tangent
    alpha = 0.01
    return tf.nn.softsign(y) + alpha * y

def softsignplus(y):
    # Hyperbolic tangent
    alpha = 0.01
    return tf.nn.tanh(y) + alpha * y


ACTIVATION_FUNC = {'Step':step, 'SimpleReturn':simplereturn, 'ReLU':ReLU, 'Approximated_ReLU':Approximated_ReLU, 'Parametric_ReLU':Parametric_ReLU, 'ELU':ELU, 'sigmoid':sigmoid, 'softsign':softsign, 'tanh':tanh, 'tanhplus':tanhplus, 'softsignplus':softsignplus}
USE_FUNC = ['tanh', 'tanh', 'tanh', 'tanh', 'SimpleReturn']
#USE_FUNC = ['softsign', 'softsign', 'softsign', 'softsign', 'SimpleReturn']
#USE_FUNC = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'SimpleReturn']
#USE_FUNC = ['tanhplus', 'tanhplus', 'tanhplus', 'tanhplus', 'SimpleReturn']
#USE_FUNC = ['softsignplus', 'softsignplus', 'softsignplus', 'softsignplus', 'SimpleReturnplus']


def main(self):
    # Import data
    file_in = '../datasets/MNIST_data/'
    mnist = input_data.read_data_sets(file_in, one_hot=True)
    W_b_in = '../datasets/W_vs_xavier_init'
    with open(W_b_in, mode='rb') as f:
        W_vs_xavier_init = pickle.load(f)
        W1_glorot3_init = discretization(W_vs_xavier_init['W1_vs_xavier_init'], 1)
        W2_glorot3_init = discretization(W_vs_xavier_init['W2_vs_xavier_init'], 2)
        W3_glorot3_init = discretization(W_vs_xavier_init['W3_vs_xavier_init'], 3)
        W4_glorot3_init = discretization(W_vs_xavier_init['W4_vs_xavier_init'], 4)
        W_glorot3_init = {'W1_glorot3_init': W1_glorot3_init, 'W2_glorot3_init': W2_glorot3_init,
                             'W3_glorot3_init': W3_glorot3_init, 'W4_glorot3_init': W4_glorot3_init}

    print('%s loaded' % file_in)
    print('%s loaded' % W_b_in)

    # Create the model
    x = tf.placeholder(tf.float32, [None, NDATA])
    h = x

    for i in range(NLAYERS - 1):
        W = tf.Variable(W_glorot3_init['W' + str(i + 1) + '_glorot3_init'], name='W' + str(i + 1))
        b = tf.Variable(np.zeros(NUNITS[i + 1]).astype(np.float32), name='b' + str(i + 1))
        h = ACTIVATION_FUNC[USE_FUNC[i]](tf.matmul(h, W) + b)
    y = h
    y_hist = tf.summary.histogram("y" + USE_FUNC[0], y)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Define save data
    model = {'x': x, 'y': y, 'y_': y_}

    # Model
    x, y, y_ = model['x'], model['y'], model['y_']

    # Load Data
    train_data = mnist.train.images
    train_label = mnist.train.labels
    test_data = mnist.test.images
    test_label = mnist.test.labels
    validation_data = mnist.validation.images
    validation_label = mnist.validation.labels

    # Define lists
    list_epoch = ['epoch']
    list_iter = ['iter']
    list_train_loss = ['train_loss']
    list_validation_loss = ['validation_loss']
    list_train_accuracy = ['train_accuracy']
    list_validation_accuracy = ['validation_accuracy']
    list_gradients_norm = ['gradients_norm']

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.

    # logit = tf.nn.softmax(y)
    # cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(logit, 1e-10, 1.0)), 1)

    # cross_entropy = tf.reduce_sum(tf.reduce_logsumexp(y - y_ * y, 1))

    # loss = tf.reduce_mean(cross_entropy)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    loss_summary = tf.summary.scalar(
        "loss_" + USE_FUNC[0] + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(NLAYERS), loss)
    loss2_summary = tf.summary.scalar(
        "loss_" + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(NLAYERS), loss)
    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    # gradients_summary = tf.summary.scalar("gradients", gradients)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).apply_gradients(zip(gradients, params))

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary = tf.summary.scalar(
        "accuracy_" + USE_FUNC[0] + '_' + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(NLAYERS), accuracy)
    accuracy_summary = tf.summary.scalar(
        "accuracy_" + str(EPOCH) + '_' + str(LEARNING_RATE) + '_' + str(NLAYERS), accuracy)

    # Setting
    saver = tf.train.Saver(max_to_keep=0)
    sess = tf.InteractiveSession()

    # 全ての要約をマージしてそれらを /tmp/mnist_logs に書き出します。
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("../tensorboard/mnist_logs/glorot3_" + USE_FUNC[0] + str(r), sess.graph_def)
    tf.initialize_all_variables().run()

    sess.run(tf.global_variables_initializer())

    # Train
    iter = int(55000 / MBATCH * EPOCH)
    for i in range(iter):
        batch_xs, batch_ys = mnist.train.next_batch(MBATCH, shuffle=False)
        gradient, _, summary_str = sess.run([gradients, train_step, merged], feed_dict={x: batch_xs, y_: batch_ys})

        writer.add_summary(summary_str, i)

        # Display progress
        sys.stdout.write("\r%.3f%%" % (i / iter * 100))
        sys.stdout.flush()

        # Testing
        if (i + 1) % (55000 / MBATCH) == 0:
            list_epoch.append(int((i + 1) / (55000 / MBATCH)))
            list_iter.append(i + 1)
            list_train_loss.append(sess.run(loss, feed_dict={x: train_data, y_: train_label}))
            list_train_accuracy.append(sess.run(accuracy, feed_dict={x: train_data, y_: train_label}))
            list_validation_loss.append(sess.run(loss, feed_dict={x: validation_data, y_: validation_label}))
            list_validation_accuracy.append(sess.run(accuracy, feed_dict={x: validation_data, y_: validation_label}))
            # list_gradients_norm.append(
            #   np.linalg.norm(list(itertools.chain.from_iterable(na.flatten().tolist() for na in gradient))))
            list_gradients_norm.append(gradient)

            # Display accuracy
            print(list_train_accuracy[-1])

            if (i + 1) / (55000 / MBATCH) % 10 == 0:
                # Save a model
                saver.save(sess,
                           '../models/' + DATASET + '/' + USE_FUNC[0] + str(r) + '/' + str(NLAYERS) + 'layers_glorot3_' + str(
                               int((i + 1) / (55000 / MBATCH))) + '.ckpt')
                # print('Saved a models.')

    print()

    # Make CSV
    with open(
            '../models/%s/%s%s/%slayers_glorot3_%f_%s.csv' % (DATASET, USE_FUNC[0], str(r), NLAYERS, LEARNING_RATE, USE_FUNC[0]),
            'w') as f:
        list_score = [list_epoch, list_iter, list_train_loss, list_train_accuracy, list_validation_loss,
                      list_validation_accuracy, list_gradients_norm]
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(list_score)
        print('%s.csv written' % USE_FUNC[0])

    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

    # Accuracy
    std_output = 'Train Loss: %s, \t Train Accuracy: %s, \t Test Accuracy: %s'
    print(std_output % (list_train_loss[-1], list_train_accuracy[-1], list_validation_accuracy[-1]))

    # Restore Model
    for i in range(iter):
        if (i + 1) / (55000 / MBATCH) % 10 == 0:
            saver.restore(sess,
                          '../models/' + DATASET + '/' + USE_FUNC[0] + str(r) + '/' + str(NLAYERS) + 'layers_glorot3_' + str(
                              int((i + 1) / (55000 / MBATCH))) + '.ckpt')

            # print('Restored a model')

    test_accuracy = sess.run(accuracy, feed_dict={x: test_data, y_: test_label})
    print('Test Accuracy(restore): %s' % test_accuracy)

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
