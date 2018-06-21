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
from tensorflow.examples.tutorials.mnist import input_data

import argparse
import sys
import scipy.io
import numpy as np
import tensorflow as tf
import pickle

NUNITS = 90
EPOCH = 100
MBATCH = 100
FLAGS = None

def sigmoid(y):
    # Sigmoid function
    return tf.exp(y)/(1+tf.exp(-y))

def ReLU(y):
    # ReLU
    return tf.maximum(0.0,y)


def main(_):
  # Import data
  file_in = '../datasets/MNIST_data/'
  mnist = input_data.read_data_sets(file_in, one_hot=True)
  W_b_in = '../datasets/W_b_init'
  with open(W_b_in, mode='rb') as f:
      W_b_init = pickle.load(f)
      W1_init = W_b_init['W1_init']
      W2_init = W_b_init['W2_init']
      b1_init = W_b_init['b1_init']
      b2_init = W_b_init['b2_init']

  print('%s loaded' % file_in)
  print('%s loaded' % W_b_in)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  with tf.variable_scope('fc1'):
     W_1 = tf.Variable(np.resize(W1_init, (784, NUNITS)))
     b_1 = tf.Variable(np.resize(b1_init, (NUNITS)))
     h_1 = tf.matmul(x, W_1) + b_1

  with tf.variable_scope('fc2'):
    W_2 = tf.Variable(np.resize(W2_init, (NUNITS, 10)))
    b_2 = tf.Variable(np.resize(b2_init, (10)))
    y = ReLU(tf.matmul(h_1, W_2) + b_2)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

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
  tf.global_variables_initializer().run()
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