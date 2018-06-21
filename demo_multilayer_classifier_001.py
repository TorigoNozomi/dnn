Python

import numpy as np
import tensorflow as tf
import time

# --------------------------------------
# パラメータ設定
# --------------------------------------
MAX_STEPS = 1000  # 学習回数
BATCH_SIZE = 100  # バッチサイズ

IMAGE_WIDTH = 28  # 画像サイズ：幅
IMAGE_HEIGHT = 28  # 画像サイズ：高さ
IMAGE_CHANNE = 1  # 画像チャネル数
TARGET_SIZE = 10  # 教師画像の種類数

CONV1_FILTER_SIZE = 3  # フィルターサイズ（幅、高さ）
CONV1_FEATURES = 32  # 特徴マップ数
CONV1_STRIDE = 1  # ストライドの設定
MAX_POOL_SIZE1 = 2  # マップサイズ（幅、高さ）
POOL1_STRIDE = 2  # ストライドの設定
AFFINE1_OUTPUT_SIZE = 100  # 全結合層（１）の出力数

NUM_THREADS = 4  # スレッド

INPUT_TFRECORD_TRAIN = "training.tfrecords"  # TFRecordファイル名（学習用）
INPUT_TFRECORD_TEST = "test.tfrecords"  # TFRecordファイル名（評価用）


# --------------------------------------
# 入力データの解析処理
# --------------------------------------
# データ解析（１）
def _parse_function(example_proto):
    features = {
        'label': tf.FixedLenFeature((), tf.int64, default_value=0),
        'image': tf.FixedLenFeature((), tf.string, default_value="")
    }
    parsed_features = tf.parse_single_example(example_proto, features)  # データ構造を解析

    return parsed_features["image"], parsed_features["label"]


# データ解析（２）
def read_image(images, labels):
    label = tf.cast(labels, tf.int32)
    label = tf.one_hot(label, TARGET_SIZE)

    image = tf.decode_raw(images, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = image / 255  # 画像データを、0～1の範囲に変換する
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNE])

    return image, label


# --------------------------------------
# モデルの作成
# --------------------------------------
# 入力層
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function, NUM_THREADS)  # レコードを解析し、テンソルに変換
dataset = dataset.map(read_image, NUM_THREADS)  # データの形式、形状を変更
dataset = dataset.batch(BATCH_SIZE)  # 連続するレコードをバッチに結合
dataset = dataset.repeat(-1)  # 無限に繰り返す
iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)  # イテレータを作成
x, y_ = iterator.get_next()  # イテレータの次の要素を取得

init_op = iterator.make_initializer(dataset)  # イテレータを初期化

# CONV層
conv1_weight = tf.Variable(tf.truncated_normal([CONV1_FILTER_SIZE, CONV1_FILTER_SIZE, 1, CONV1_FEATURES], stddev=0.1))
conv1_bias = tf.Variable(tf.zeros([CONV1_FEATURES]))
conv1 = tf.nn.bias_add(tf.nn.conv2d(x, conv1_weight, strides=[1, CONV1_STRIDE, CONV1_STRIDE, 1], padding='SAME'),
                       conv1_bias)
relu1 = tf.nn.relu(conv1)
# POOL層
pool1 = tf.nn.max_pool(relu1, ksize=[1, MAX_POOL_SIZE1, MAX_POOL_SIZE1, 1], strides=[1, POOL1_STRIDE, POOL1_STRIDE, 1],
                       padding='SAME')
pool1_shape = pool1.get_shape().as_list()
pool1_flat_shape = pool1_shape[1] * pool1_shape[2] * pool1_shape[3]
pool1_flat = tf.reshape(pool1, [-1, pool1_flat_shape])  # 2次元に変換

# 全結合層1
W1 = tf.Variable(tf.truncated_normal([pool1_flat_shape, AFFINE1_OUTPUT_SIZE]))
b1 = tf.Variable(tf.zeros([AFFINE1_OUTPUT_SIZE]))
affine1 = tf.matmul(pool1_flat, W1) + b1
sigmoid1 = tf.sigmoid(affine1)
# 全結合層2
W2 = tf.Variable(tf.truncated_normal([AFFINE1_OUTPUT_SIZE, TARGET_SIZE]))
b2 = tf.Variable(tf.zeros([TARGET_SIZE]))
affine2 = tf.matmul(sigmoid1, W2) + b2

# 出力層
y = tf.nn.softmax(affine2)

# 誤差関数(loss)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=affine2))

# 最適化手段(最急降下法)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# 正答率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------
# 学習と評価
# --------------------------------------
with tf.Session() as sess:
    try:
        # 学習

        init = tf.global_variables_initializer()
        sess.run(init)  # 変数の初期化処理

        sess.run(init_op, feed_dict={filenames: [INPUT_TFRECORD_TRAIN]})  # データの初期化
        for step in range(MAX_STEPS):
            start_time = time.time()

            _, l, acr = sess.run([train_step, loss, accuracy])  # 最急勾配法でパラメータ更新

            duration = time.time() - start_time

            if (step + 1) % 100 == 0:
                print("step={:4d}, loss={:5.2f}, Accuracy={:5.2f} ({:.3f} sec)".format(step + 1, l, acr, duration))

        # 評価
        sess.run(init_op, feed_dict={filenames: [INPUT_TFRECORD_TEST]})  # データの初期化
        est_accuracy, est_y, new_y_ = sess.run([accuracy, y, y_])
        print("Accuracy (for test data): {:5.2f}".format(est_accuracy))
        print("True Label:", np.argmax(new_y_[0:15, ], 1))
        print("Est Label:", np.argmax(est_y[0:15, ], 1))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        print("step={:4d}".format(step + 1))
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
import numpy as np
import tensorflow as tf
import time

# --------------------------------------
# パラメータ設定
# --------------------------------------
MAX_STEPS = 1000  # 学習回数
BATCH_SIZE = 100  # バッチサイズ

IMAGE_WIDTH = 28  # 画像サイズ：幅
IMAGE_HEIGHT = 28  # 画像サイズ：高さ
IMAGE_CHANNE = 1  # 画像チャネル数
TARGET_SIZE = 10  # 教師画像の種類数

CONV1_FILTER_SIZE = 3  # フィルターサイズ（幅、高さ）
CONV1_FEATURES = 32  # 特徴マップ数
CONV1_STRIDE = 1  # ストライドの設定
MAX_POOL_SIZE1 = 2  # マップサイズ（幅、高さ）
POOL1_STRIDE = 2  # ストライドの設定
AFFINE1_OUTPUT_SIZE = 100  # 全結合層（１）の出力数

NUM_THREADS = 4  # スレッド

INPUT_TFRECORD_TRAIN = "training.tfrecords"  # TFRecordファイル名（学習用）
INPUT_TFRECORD_TEST = "test.tfrecords"  # TFRecordファイル名（評価用）


# --------------------------------------
# 入力データの解析処理
# --------------------------------------
# データ解析（１）
def _parse_function(example_proto):
    features = {
        'label': tf.FixedLenFeature((), tf.int64, default_value=0),
        'image': tf.FixedLenFeature((), tf.string, default_value="")
    }
    parsed_features = tf.parse_single_example(example_proto, features)  # データ構造を解析

    return parsed_features["image"], parsed_features["label"]


# データ解析（２）
def read_image(images, labels):
    label = tf.cast(labels, tf.int32)
    label = tf.one_hot(label, TARGET_SIZE)

    image = tf.decode_raw(images, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = image / 255  # 画像データを、0～1の範囲に変換する
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNE])

    return image, label


# --------------------------------------
# モデルの作成
# --------------------------------------
# 入力層
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function, NUM_THREADS)  # レコードを解析し、テンソルに変換
dataset = dataset.map(read_image, NUM_THREADS)  # データの形式、形状を変更
dataset = dataset.batch(BATCH_SIZE)  # 連続するレコードをバッチに結合
dataset = dataset.repeat(-1)  # 無限に繰り返す
iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)  # イテレータを作成
x, y_ = iterator.get_next()  # イテレータの次の要素を取得

init_op = iterator.make_initializer(dataset)  # イテレータを初期化

# CONV層
conv1_weight = tf.Variable(tf.truncated_normal([CONV1_FILTER_SIZE, CONV1_FILTER_SIZE, 1, CONV1_FEATURES], stddev=0.1))
conv1_bias = tf.Variable(tf.zeros([CONV1_FEATURES]))
conv1 = tf.nn.bias_add(tf.nn.conv2d(x, conv1_weight, strides=[1, CONV1_STRIDE, CONV1_STRIDE, 1], padding='SAME'),
                       conv1_bias)
relu1 = tf.nn.relu(conv1)
# POOL層
pool1 = tf.nn.max_pool(relu1, ksize=[1, MAX_POOL_SIZE1, MAX_POOL_SIZE1, 1], strides=[1, POOL1_STRIDE, POOL1_STRIDE, 1],
                       padding='SAME')
pool1_shape = pool1.get_shape().as_list()
pool1_flat_shape = pool1_shape[1] * pool1_shape[2] * pool1_shape[3]
pool1_flat = tf.reshape(pool1, [-1, pool1_flat_shape])  # 2次元に変換

# 全結合層1
W1 = tf.Variable(tf.truncated_normal([pool1_flat_shape, AFFINE1_OUTPUT_SIZE]))
b1 = tf.Variable(tf.zeros([AFFINE1_OUTPUT_SIZE]))
affine1 = tf.matmul(pool1_flat, W1) + b1
sigmoid1 = tf.sigmoid(affine1)
# 全結合層2
W2 = tf.Variable(tf.truncated_normal([AFFINE1_OUTPUT_SIZE, TARGET_SIZE]))
b2 = tf.Variable(tf.zeros([TARGET_SIZE]))
affine2 = tf.matmul(sigmoid1, W2) + b2

# 出力層
y = tf.nn.softmax(affine2)

# 誤差関数(loss)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=affine2))

# 最適化手段(最急降下法)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# 正答率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------
# 学習と評価
# --------------------------------------
with tf.Session() as sess:
    try:
        # 学習

        init = tf.global_variables_initializer()
        sess.run(init)  # 変数の初期化処理

        sess.run(init_op, feed_dict={filenames: [INPUT_TFRECORD_TRAIN]})  # データの初期化
        for step in range(MAX_STEPS):
            start_time = time.time()

            _, l, acr = sess.run([train_step, loss, accuracy])  # 最急勾配法でパラメータ更新

            duration = time.time() - start_time

            if (step + 1) % 100 == 0:
                print("step={:4d}, loss={:5.2f}, Accuracy={:5.2f} ({:.3f} sec)".format(step + 1, l, acr, duration))

        # 評価
        sess.run(init_op, feed_dict={filenames: [INPUT_TFRECORD_TEST]})  # データの初期化
        est_accuracy, est_y, new_y_ = sess.run([accuracy, y, y_])
        print("Accuracy (for test data): {:5.2f}".format(est_accuracy))
        print("True Label:", np.argmax(new_y_[0:15, ], 1))
        print("Est Label:", np.argmax(est_y[0:15, ], 1))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        print("step={:4d}".format(step + 1))