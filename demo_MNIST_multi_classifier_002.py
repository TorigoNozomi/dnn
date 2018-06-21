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

NUNITS = 90
EPOCH = 100
MBATCH = 100
FLAGS = None


def sigmoid(y):
    # Sigmoid function
    return tf.exp(y) / (1 + tf.exp(-y))


def ReLU(y):
    # ReLU
    return tf.maximum(0.0, y)

class Test:
    def __init__(self):
      pass

    def main(self):
        # Import data
        data = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        model = self.design_model(data)
        self.save_model(data, model)
        self.restore_model(data, model)

    def design_model(self, data):
        W_b_in = '../datasets/W_b_init'
        with open(W_b_in, mode='rb') as f:
            W_b_init = pickle.load(f)
            W1_init = W_b_init['W1_init']
            W2_init = W_b_init['W2_init']
            b1_init = W_b_init['b1_init']
            b2_init = W_b_init['b2_init']

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

        tf.add_to_collection('vars', W_1)
        tf.add_to_collection('vars', b_1)
        tf.add_to_collection('vars', W_2)
        tf.add_to_collection('vars', b_2)

        model = {'x': x, 'y': y, 'y_': y_}

        return model

    def save_model(self, data, model):
        # Load Data
        train_data = data.train.images
        train_label = data.train.labels
        test_data = data.test.images
        test_label = data.test.labels

        # Model
        x, y, y_ = model['x'], model['y'], model['y_']

        # Functions
        loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Setting
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        # Training
        iter = int(55000 / MBATCH * EPOCH)
        for _ in range(iter):
            batch_xs, batch_ys = data.train.next_batch(MBATCH, shuffle=False)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            # Testing
            global train_loss
            global train_accuracy
            global test_accuracy
            train_loss = sess.run(loss, feed_dict={x: train_data, y_: train_label})
            train_accuracy = sess.run(accuracy, feed_dict={x: train_data, y_: train_label})
            test_accuracy = sess.run(accuracy, feed_dict={x: test_data, y_: test_label})

        # Accuracy
        std_output = 'Train Loss: %s, \t Train Accuracy: %s, \t Test Accuracy: %s'
        print(std_output % (train_loss, train_accuracy, test_accuracy))

        # Save a model
        for f in os.listdir('../models/'):
            os.remove('../models/'+f)
        saver.save(sess, '../models/test_model')
        print('Saved a models.')

        sess.close()

    def restore_model(self, data, model):
        # Data
        train_data = data.train.images
        train_label = data.train.labels
        test_data = data.test.images
        test_label = data.test.labels

        # Model
        x, y, y_ = model['x'], model['y'], model['y_']

        # Function
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))

        # Setting
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, '../model/test_model')
        print('Restored a model')

        test_accuracy = sess.run(accuracy, feed_dict={x: test_data, y_: test_label})
        print('Test Accuracy: %s' % test_accuracy)