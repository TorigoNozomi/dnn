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

W1_vs_he2_init = np.random.uniform(0, 1, [NUNITS[0], NUNITS[1]]).astype(np.float32)
W2_vs_he2_init = np.random.uniform(0, 1, [NUNITS[1], NUNITS[2]]).astype(np.float32)
W3_vs_he2_init = np.random.uniform(0, 1, [NUNITS[2], NUNITS[3]]).astype(np.float32)
W4_vs_he2_init = np.random.uniform(0, 1, [NUNITS[3], NUNITS[4]]).astype(np.float32)

W1_he2_init = (W1_vs_he2_init / np.sqrt(NUNITS[0] / 2.0)).astype(np.float32)
W2_he2_init = (W2_vs_he2_init / np.sqrt(NUNITS[1] / 2.0)).astype(np.float32)
W3_he2_init = (W3_vs_he2_init / np.sqrt(NUNITS[2] / 2.0)).astype(np.float32)
W4_he2_init = (W4_vs_he2_init / np.sqrt(NUNITS[3] / 2.0)).astype(np.float32)

b1_he2_init = np.random.uniform(0, 1, NUNITS[1]).astype(np.float32)
b2_he2_init = np.random.uniform(0, 1, NUNITS[1]).astype(np.float32)
b3_he2_init = np.random.uniform(0, 1, NUNITS[1]).astype(np.float32)
b4_he2_init = np.random.uniform(0, 1, NUNITS[1]).astype(np.float32)


Wb_he2_init = {'W1_he2_init': W1_he2_init, 'W2_he2_init': W2_he2_init, 'W3_he2_init': W3_he2_init, 'W4_he2_init': W4_he2_init, 'b1_he2_init': b1_he2_init, 'b2_he2_init': b2_he2_init, 'b3_he2_init': b3_he2_init, 'b4_he2_init': b4_he2_init}
Wb_vs_he2_init = {'W1_vs_he2_init': W1_vs_he2_init, 'W2_vs_he2_init': W2_vs_he2_init, 'W3_vs_he2_init': W3_vs_he2_init, 'W4_vs_he2_init': W4_vs_he2_init, 'b1_he2_init': b1_he2_init, 'b2_he2_init': b2_he2_init, 'b3_he2_init': b3_he2_init, 'b4_he2_init': b4_he2_init}

with open('Wb_he2_init', mode= 'wb') as f:
        pickle.dump(Wb_he2_init,f)
        print(W1_he2_init)

with open('Wb_vs_he2_init', mode='wb') as f:
        pickle.dump(Wb_vs_he2_init, f)
        print(W1_vs_he2_init)