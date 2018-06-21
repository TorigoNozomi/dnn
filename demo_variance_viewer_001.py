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

x = np.random.randn(1000,100)
activations = {}
for i in range(NLAYER):
    if i != 0:
        x = activations[i - 1]

    w = pi.he(NUNIT[i], NUNIT[i+1])
    z = tf.cast(tf.matmul(x, w), tf.float32)
    a = ACTIVATION_FUNC[USE_FUNC](z)
    activations[i] = a

plt.figure(figsize=(18, 4))
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))
    plt.ylim(y_lim)
plt.show()
plt.savefig('../image/' + USE_FUNC + '_' + USE_INIT + '_figure.png')
