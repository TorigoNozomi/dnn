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

#5 layer perceptron

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

NLAYERS = 3
NCLASS = 10
NDATA = 784
NUNITS = [NDATA, 90, NCLASS]
EPOCH = 100
MBATCH = 100
DATASET = 'MNIST'
FLAGS = None

def sigmoid(y):
    # Sigmoid function
    return 1 / (1 + tf.exp(-y))


def ReLU(y):
    # ReLU
    return tf.maximum(0.0, y)

def weight_variable(shape):
    # Define initial weight by normal distribution
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # Define initial weight by normal distribution
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

ACTIVATION_FUNC = {'ReLU':ReLU,'sigmoid':sigmoid}
USE_FUNC = 'sigmoid'

def main(self):
  # Import data
  file_in = '../datasets/MNIST_data/'
  mnist = input_data.read_data_sets(file_in, one_hot=True)
  W_b_in = '../datasets/W_b_init'
  with open(W_b_in, mode='rb') as f:
      W_b_init = pickle.load(f)
      W1_init = W_b_init['W1_init']
      W2_init = W_b_init['W2_init']
      W3_init = W_b_init['W3_init']
      b1_init = W_b_init['b1_init']
      b2_init = W_b_init['b2_init']
      b3_init = W_b_init['b3_init']


  print('%s loaded' % file_in)
  print('%s loaded' % W_b_in)

  # Create the model
  x = tf.placeholder(tf.float32, [None, NDATA])

  with tf.variable_scope('fc1') as scope:
      W_1 = tf.Variable(np.resize(W1_init, (NUNITS[0], NUNITS[1])))
      b_1 = tf.Variable(np.resize(b1_init, (NUNITS[1])))
      h_1 = ACTIVATION_FUNC[USE_FUNC](tf.matmul(x, W_1) + b_1)

  with tf.variable_scope('fc2') as scope:
      W_2 = tf.Variable(np.resize(W2_init, (NUNITS[1], NUNITS[2])))
      b_2 = tf.Variable(np.resize(b2_init, (NUNITS[2])))
      y = tf.matmul(h_1, W_2) + b_2

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  tf.add_to_collection('vars', W_1)
  tf.add_to_collection('vars', b_1)
  tf.add_to_collection('vars', W_2)
  tf.add_to_collection('vars', b_2)

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
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  cross_entropy = tf.reduce_mean(loss)
  params = tf.trainable_variables()
  gradients = tf.gradients(loss, params)
  train_step = tf.train.GradientDescentOptimizer(0.5).apply_gradients(zip(gradients, params))

  #Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Setting
  saver = tf.train.Saver(max_to_keep=0)
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  # Train
  iter = int(55000 / MBATCH * EPOCH)
  for i in range(iter):
    batch_xs, batch_ys = mnist.train.next_batch(MBATCH, shuffle=False)
    _, gradient = sess.run([train_step, gradients], feed_dict={x: batch_xs, y_: batch_ys})

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
        list_gradients_norm.append(
            np.linalg.norm(list(itertools.chain.from_iterable(na.flatten().tolist() for na in gradient))))

        #Display accuracy
        print(list_train_accuracy[-1])

        if (i + 1) / (55000 / MBATCH) % 10 == 0:
            # Save a model
            saver.save(sess,
                       '../models/' + DATASET + '/' + USE_FUNC + '/'+ str(NLAYERS) + 'layers_' + str(int((i + 1) / (55000 / MBATCH))) + '.ckpt')
            #print('Saved a models.')

  print()

  # Make CSV
  with open('../models/%s/%s/%slayers_%s.csv' % (DATASET, USE_FUNC, NLAYERS, USE_FUNC), 'w') as f:
      list_score = [list_epoch, list_iter, list_train_loss, list_train_accuracy, list_validation_loss,
                    list_validation_accuracy, list_gradients_norm]
      writer = csv.writer(f, lineterminator='\n')
      writer.writerows(list_score)
      print('%s.csv written' % USE_FUNC)

  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

  # Accuracy
  std_output = 'Train Loss: %s, \t Train Accuracy: %s, \t Test Accuracy: %s'
  print(std_output % (list_train_loss[-1], list_train_accuracy[-1], list_validation_accuracy[-1]))

 #Restore Model
  for i in range(iter):
      if (i + 1) / (55000 / MBATCH) % 10 == 0:
          saver.restore(sess,
                        '../models/' + DATASET + '/' + USE_FUNC + '/' + str(NLAYERS) + 'layers_' + str(int((i + 1) / (55000 / MBATCH))) + '.ckpt')

          #print('Restored a model')

  test_accuracy = sess.run(accuracy, feed_dict={x: test_data, y_: test_label})
  print('Test Accuracy(restore): %s' % test_accuracy)

  sess.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)