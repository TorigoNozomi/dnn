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

W1_vs_xavier2_init = np.random.normal(0, 1, (NUNITS[0], NUNITS[1])).astype(np.float32)
W2_vs_xavier2_init = np.random.normal(0, 1, (NUNITS[1], NUNITS[2])).astype(np.float32)
W3_vs_xavier2_init = np.random.normal(0, 1, (NUNITS[2], NUNITS[3])).astype(np.float32)
W4_vs_xavier2_init = np.random.normal(0, 1, (NUNITS[3], NUNITS[4])).astype(np.float32)

W1_xavier2_init = (W1_vs_xavier2_init / np.sqrt((NUNITS[0] + NUNITS[1]) / 2)).astype(np.float32)
W2_xavier2_init = (W2_vs_xavier2_init / np.sqrt((NUNITS[1] + NUNITS[2]) / 2)).astype(np.float32)
W3_xavier2_init = (W3_vs_xavier2_init / np.sqrt((NUNITS[2] + NUNITS[3]) / 2)).astype(np.float32)
W4_xavier2_init = (W4_vs_xavier2_init / np.sqrt((NUNITS[3] + NUNITS[4]) / 2)).astype(np.float32)

W_xavier2_init = {'W1_xavier2_init': W1_xavier2_init, 'W2_xavier2_init': W2_xavier2_init, 'W3_xavier2_init': W3_xavier2_init, 'W4_xavier2_init': W4_xavier2_init}
W_vs_xavier2_init = {'W1_vs_xavier2_init': W1_vs_xavier2_init, 'W2_vs_xavier2_init': W2_vs_xavier2_init, 'W3_vs_xavier2_init': W3_vs_xavier2_init, 'W4_vs_xavier2_init': W4_vs_xavier2_init}

with open('W_xavier2_init', mode= 'wb') as f:
        pickle.dump(W_xavier2_init,f)
        print(W1_xavier2_init)

with open('W_vs_xavier2_init', mode='wb') as f:
        pickle.dump(W_vs_xavier2_init, f)
        print(W1_vs_xavier2_init)