#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import pickle

NCLASS = 10
NSIZE = 32
NDATA = NSIZE * NSIZE * 3
NUNITS = [NDATA, 500, 300, 200, 150, 100, 60, 40, 30, NCLASS]
EPOCH = 100
MBATCH = 100
LEARNING_RATE = 0.05
DATASET = 'CIFAR-10'
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', '/tmp/data', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data

# ラベル名をロード
label_names = unpickle("../datasets/data/cifar10/cifar-10-batches-py/batches.meta")["label_names"]
d = unpickle("../datasets/data/cifar10/cifar-10-batches-py/data_batch_1")
data = d["data"]
labels = np.array(d["labels"])
nsamples = len(data)

print(label_names)