import scipy.io
import tensorflow as tf
import numpy as np
import sys
import pickle

NDATA = 1000
NUNITS = 1000
NLAYERS = 5

np.random.seed(0)

W1_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
b1_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
W2_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
b2_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
W3_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
b3_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
W4_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
b4_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
W5_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
b5_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
W6_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
b6_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
W7_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
b7_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
W8_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
b8_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
W9_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
b9_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
W10_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)
b10_init = np.random.normal(0, 0.1, (NDATA, NUNITS)).astype(np.float32)

W_b_init = {'W1_init': W1_init, 'W2_init': W2_init, 'W3_init': W3_init, 'W4_init': W4_init, 'W5_init': W5_init, 'W6_init': W6_init, 'W7_init': W7_init, 'W8_init': W8_init, 'W9_init': W9_init, 'W10_init': W10_init, 'b1_init': b1_init, 'b2_init': b2_init, 'b3_init': b3_init, 'b4_init': b4_init, 'b5_init': b5_init, 'b6_init': b6_init, 'b7_init': b7_init, 'b8_init': b8_init, 'b9_init': b9_init, 'b10_init': b10_init}

with open('W_b_init', mode= 'wb') as f:
        pickle.dump(W_b_init,f)
        print(W1_init)


