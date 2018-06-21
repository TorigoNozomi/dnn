import numpy as np
import tensorflow as tf

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

def tanhplus(y):
    # Hyperbolic tangent
    alpha = 0.01
    return tf.nn.softsign(y) + alpha * y

def softsignplus(y):
    # Hyperbolic tangent
    alpha = 0.01
    return tf.nn.tanh(y) + alpha * y