import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import parameter_initialization as pi
import activation_func as af

NLAYER = 5
NUNIT = [100, 80, 60, 40, 10]
EPOCH = 10
LEARNING_RATE = 0.05
ACTIVATION_FUNC = {'Step': af.step, 'SimpleReturn': af.simplereturn, 'ReLU': af.ReLU, 'Approximated_ReLU': af.Approximated_ReLU, 'Parametric_ReLU': af.Parametric_ReLU, 'ELU': af.ELU, 'sigmoid': af.sigmoid, 'softsign': af.softsign, 'tanh': af.tanh, 'tanhplus': af.tanhplus, 'softsignplus': af.softsignplus}
USE_FUNC = 'ReLU'
PARAMETER_INIT = {'he': pi.he, 'randn': pi.randn, 'xavier': pi.xavier, 'normalized_initialization': pi.ni}
USE_INIT = 'he'
y_lim = (0, 7000)

x = np.random.randn(1000, 100)
x_label = np.random.randint(0,9,(1000))
y_ = tf.one_hot(x_label,10)

with tf.Session() as sess:
