import scipy.io
import tensorflow as tf
import numpy as np
import sys
import pickle


NLAYERS = 5
NCLASS = 10
NDATA = 784
NUNITS = [NDATA, 300, 150, 40, NCLASS]
np.random.seed(0)

W1_vs_he_init = np.random.normal(0, 1, [NUNITS[0], NUNITS[1]]).astype(np.float32)
W2_vs_he_init = np.random.normal(0, 1, [NUNITS[1], NUNITS[2]]).astype(np.float32)
W3_vs_he_init = np.random.normal(0, 1, [NUNITS[2], NUNITS[3]]).astype(np.float32)
W4_vs_he_init = np.random.normal(0, 1, [NUNITS[3], NUNITS[4]]).astype(np.float32)

W1_he_init = (W1_vs_he_init / np.sqrt(NUNITS[0] / 2.0)).astype(np.float32)
W2_he_init = (W2_vs_he_init / np.sqrt(NUNITS[1] / 2.0)).astype(np.float32)
W3_he_init = (W3_vs_he_init / np.sqrt(NUNITS[2] / 2.0)).astype(np.float32)
W4_he_init = (W4_vs_he_init / np.sqrt(NUNITS[3] / 2.0)).astype(np.float32)


W_he_init = {'W1_he_init': W1_he_init, 'W2_he_init': W2_he_init, 'W3_he_init': W3_he_init, 'W4_he_init': W4_he_init}
W_vs_he_init = {'W1_vs_he_init': W1_vs_he_init, 'W2_vs_he_init': W2_vs_he_init, 'W3_vs_he_init': W3_vs_he_init, 'W4_vs_he_init': W4_vs_he_init}

with open('W_he_init', mode= 'wb') as f:
        pickle.dump(W_he_init,f)
        print(W1_he_init)

with open('W_vs_he_init', mode='wb') as f:
        pickle.dump(W_vs_he_init, f)
        print(W1_vs_he_init)