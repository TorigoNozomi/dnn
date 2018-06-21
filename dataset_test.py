import tensorflow as tf
import os

hello = tf.constant("Hello World!")
sess = tf.Session()
print(sess.run(hello))
print(os.getcwd())
files = os.path.dirname(os.path.abspath(__file__))
print(files)
