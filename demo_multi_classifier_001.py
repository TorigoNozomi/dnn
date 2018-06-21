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

NUNITS = [90, 80, 50, 30, 10]
EPOCH = 1000
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
      W3_init = W_b_init['W3_init']
      W4_init = W_b_init['W4_init']
      W5_init = W_b_init['W5_init']
      b1_init = W_b_init['b1_init']
      b2_init = W_b_init['b2_init']
      b3_init = W_b_init['b3_init']
      b4_init = W_b_init['b4_init']
      b5_init = W_b_init['b5_init']



  print('%s loaded' % file_in)
  print('%s loaded' % W_b_in)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  with tf.variable_scope('fc1'):
     W_1 = tf.Variable(np.resize(W1_init, (784, NUNITS[0])))
     b_1 = tf.Variable(np.resize(b1_init, (NUNITS[0])))
     h_1 = ReLU(tf.matmul(x, W_1) + b_1)

  with tf.variable_scope('fc2'):
    W_2 = tf.Variable(np.resize(W2_init, (NUNITS[0], NUNITS[1])))
    b_2 = tf.Variable(np.resize(b2_init, (NUNITS[1])))
    h_2 = ReLU(tf.matmul(h_1, W_2) + b_2)

  with tf.variable_scope('fc3'):
    W_3 = tf.Variable(np.resize(W3_init, (NUNITS[1], NUNITS[2])))
    b_3 = tf.Variable(np.resize(b3_init, (NUNITS[2])))
    h_3 = ReLU(tf.matmul(h_2, W_3) + b_3)

  with tf.variable_scope('fc4'):
    W_4 = tf.Variable(np.resize(W4_init, (NUNITS[2], NUNITS[3])))
    b_4 = tf.Variable(np.resize(b4_init, (NUNITS[3])))
    h_4 = ReLU(tf.matmul(h_3, W_4) + b_4)

  with tf.variable_scope('fc5'):
    W_5 = tf.Variable(np.resize(W5_init, (NUNITS[3], NUNITS[4])))
    b_5 = tf.Variable(np.resize(b5_init, (NUNITS[4])))
    y = ReLU(tf.matmul(h_4, W_5) + b_5)

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