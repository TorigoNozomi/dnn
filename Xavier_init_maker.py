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

W1_vs_xavier_init = np.random.uniform(-1, 1, [NUNITS[0], NUNITS[1]]).astype(np.float32)
W2_vs_xavier_init = np.random.uniform(-1, 1, [NUNITS[1], NUNITS[2]]).astype(np.float32)
W3_vs_xavier_init = np.random.uniform(-1, 1, [NUNITS[2], NUNITS[3]]).astype(np.float32)
W4_vs_xavier_init = np.random.uniform(-1, 1, [NUNITS[3], NUNITS[4]]).astype(np.float32)

W1_xavier_init = (W1_vs_xavier_init / np.sqrt(NUNITS[0])).astype(np.float32)
W2_xavier_init = (W2_vs_xavier_init / np.sqrt(NUNITS[1])).astype(np.float32)
W3_xavier_init = (W3_vs_xavier_init / np.sqrt(NUNITS[2])).astype(np.float32)
W4_xavier_init = (W4_vs_xavier_init / np.sqrt(NUNITS[3])).astype(np.float32)

W1_glorot_init = (W1_vs_xavier_init / np.sqrt((NUNITS[0] + NUNITS[1]) / 6)).astype(np.float32)
W2_glorot_init = (W2_vs_xavier_init / np.sqrt((NUNITS[1] + NUNITS[2]) / 6)).astype(np.float32)
W3_glorot_init = (W3_vs_xavier_init / np.sqrt((NUNITS[2] + NUNITS[3]) / 6)).astype(np.float32)
W4_glorot_init = (W4_vs_xavier_init / np.sqrt((NUNITS[3] + NUNITS[4]) / 6)).astype(np.float32)

W_xavier_init = {'W1_xavier_init': W1_xavier_init, 'W2_xavier_init': W2_xavier_init, 'W3_xavier_init': W3_xavier_init, 'W4_xavier_init': W4_xavier_init}
W_vs_xavier_init = {'W1_vs_xavier_init': W1_vs_xavier_init, 'W2_vs_xavier_init': W2_vs_xavier_init, 'W3_vs_xavier_init': W3_vs_xavier_init, 'W4_vs_xavier_init': W4_vs_xavier_init}
W_glorot_init = {'W1_glorot_init': W1_glorot_init, 'W2_glorot_init': W2_glorot_init, 'W3_glorot_init': W3_glorot_init, 'W4_glorot_init': W4_glorot_init}

with open('W_xavier_init', mode= 'wb') as f:
        pickle.dump(W_xavier_init,f)
        print(W1_xavier_init)

with open('W_vs_xavier_init', mode='wb') as f:
        pickle.dump(W_vs_xavier_init, f)
        print(W1_vs_xavier_init)

with open('W_glorot_init', mode='wb') as f:
    pickle.dump(W_glorot_init, f)
    print(W1_glorot_init)