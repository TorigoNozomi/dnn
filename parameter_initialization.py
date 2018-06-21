import numpy as np

#Gaussian distribution initialization
def randn(n_1,n_2,s_dev):
    return np.random.randn(n_1, n_2) * s_dev

#Xavier initialization
def xavier(n_1,n_2):
    return np.random.randn(n_1, n_2) * np.sqrt(1.0 / n_1)

#normalized initialization
def ni(n_1,n_2):
    return np.random.randn(n_1, n_2) * np.sqrt(2.0 / n_1 + n_2)

#He initialization
def he(n_1,n_2):
   return np.random.randn(n_1, n_2) * np.sqrt(2.0 / n_1)