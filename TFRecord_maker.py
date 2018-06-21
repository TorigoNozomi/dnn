from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import glob

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


def matrix_to_image(imageMatrix, imageShape, dirName, labal):
    imageMatrix = imageMatrix * 255  # 画像データの値を0～255の範囲に変更する
    for i in range(0, imageMatrix.shape[0]):
        imageArray = imageMatrix[i].reshape(imageShape)
        outImg = Image.fromarray(imageArray)
        outImg = outImg.convert("L")  # グレースケール
        outImg.save(dirName + os.sep + str(i) + "-" + str(np.argmax(labal[i])) + ".jpg", format="JPEG")


matrix_to_image(mnist.test.images, imageShape=(28, 28), dirName="mnistVisualize", labal=mnist.test.labels)

def matrix_to_image(imageMatrix, imageShape, dirName, labal):
    imageMatrix = imageMatrix * 255  # 画像データの値を0～255の範囲に変更する
    for i in range(0, imageMatrix.shape[0]):
        imageArray = imageMatrix[i].reshape(imageShape)
        outImg = Image.fromarray(imageArray)
        outImg = outImg.convert("L")  # グレースケール
        outImg.save(dirName + os.sep + str(i) + "-" + str(np.argmax(labal[i])) + ".jpg", format="JPEG")


matrix_to_image(mnist.test.images, imageShape=(28, 28), dirName="mnistVisualize", labal=mnist.test.labels)

OUTPUT_TFRECORD_NAME = "test_tf_file.tfrecords"  # アウトプットするTFRecordファイル名


def CreateTensorflowReadFile(img_files, out_file):
    with tf.python_io.TFRecordWriter(out_file) as writer:
        for f in img_files:
            # ファイルを開く
            with Image.open(f).convert("L") as image_object:  # グレースケール
                image = np.array(image_object)

                height = image.shape[0]
                width = image.shape[1]
                image_raw = image.tostring()
                label = int(f[f.rfind("-") + 1: -4])  # ファイル名からラベルを取得

                example = tf.train.Example(features=tf.train.Features(feature={
                    "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                    "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_object.tobytes()]))
                }))

            # レコード書込
            writer.write(example.SerializeToString())


# 書き込み
files = glob.glob("mnistVisualize" + os.sep + "*.jpg")
CreateTensorflowReadFile(files, OUTPUT_TFRECORD_NAME)

cnt = len(list(tf.python_io.tf_record_iterator(OUTPUT_TFRECORD_NAME)))
print("データ件数：{}".format(cnt))

example = next(tf.python_io.tf_record_iterator(OUTPUT_TFRECORD_NAME))
tf.train.Example.FromString(example)

Python

count = 0
for record in tf.python_io.tf_record_iterator(OUTPUT_TFRECORD_NAME):
    example = tf.train.Example()
    example.ParseFromString(record)  # バイナリデータからの読み込み

    height = example.features.feature["height"].int64_list.value[0]
    width = example.features.feature["width"].int64_list.value[0]
    label = example.features.feature["label"].int64_list.value[0]
    image = example.features.feature["image"].bytes_list.value[0]

    image = np.fromstring(image, dtype=np.uint8)
    image = image.reshape([height, width])
    img = Image.fromarray(image, "L")  # グレースケール
    img.save(os.path.join("check_tfRecords", "tfrecords_{0}-{1}.jpg".format(str(count), label)))
    count += 1

count = 0
for record in tf.python_io.tf_record_iterator(OUTPUT_TFRECORD_NAME):
    example = tf.train.Example()
    example.ParseFromString(record)  # バイナリデータからの読み込み

    height = example.features.feature["height"].int64_list.value[0]
    width = example.features.feature["width"].int64_list.value[0]
    label = example.features.feature["label"].int64_list.value[0]
    image = example.features.feature["image"].bytes_list.value[0]

    image = np.fromstring(image, dtype=np.uint8)
    image = image.reshape([height, width])
    img = Image.fromarray(image, "L")  # グレースケール
    img.save(os.path.join("check_tfRecords", "tfrecords_{0}-{1}.jpg".format(str(count), label)))
    count += 1