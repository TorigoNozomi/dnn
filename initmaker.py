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

#Xavier
W1_vs_xavier_init = np.random.uniform(-1, 1, [NUNITS[0], NUNITS[1]]).astype(np.float32)
W2_vs_xavier_init = np.random.uniform(-1, 1, [NUNITS[1], NUNITS[2]]).astype(np.float32)
W3_vs_xavier_init = np.random.uniform(-1, 1, [NUNITS[2], NUNITS[3]]).astype(np.float32)
W4_vs_xavier_init = np.random.uniform(-1, 1, [NUNITS[3], NUNITS[4]]).astype(np.float32)

W1_xavier_init = (W1_vs_xavier_init / np.sqrt(NUNITS[0])).astype(np.float32)
W2_xavier_init = (W2_vs_xavier_init / np.sqrt(NUNITS[1])).astype(np.float32)
W3_xavier_init = (W3_vs_xavier_init / np.sqrt(NUNITS[2])).astype(np.float32)
W4_xavier_init = (W4_vs_xavier_init / np.sqrt(NUNITS[3])).astype(np.float32)

W_xavier_init = {'W1_xavier_init': W1_xavier_init, 'W2_xavier_init': W2_xavier_init, 'W3_xavier_init': W3_xavier_init, 'W4_xavier_init': W4_xavier_init}
W_vs_xavier_init = {'W1_vs_xavier_init': W1_vs_xavier_init, 'W2_vs_xavier_init': W2_vs_xavier_init, 'W3_vs_xavier_init': W3_vs_xavier_init, 'W4_vs_xavier_init': W4_vs_xavier_init}

with open('W_xavier_init', mode='wb') as f:
    pickle.dump(W_xavier_init, f)
    print(W1_xavier_init)

with open('W_vs_xavier_init', mode='wb') as f:
    pickle.dump(W_vs_xavier_init, f)
    print(W1_vs_xavier_init)

#Glorot
W1_glorot_init = (W1_vs_xavier_init / np.sqrt((NUNITS[0] + NUNITS[1]) / 6)).astype(np.float32)
W2_glorot_init = (W2_vs_xavier_init / np.sqrt((NUNITS[1] + NUNITS[2]) / 6)).astype(np.float32)
W3_glorot_init = (W3_vs_xavier_init / np.sqrt((NUNITS[2] + NUNITS[3]) / 6)).astype(np.float32)
W4_glorot_init = (W4_vs_xavier_init / np.sqrt((NUNITS[3] + NUNITS[4]) / 6)).astype(np.float32)

W1_vs_glorot2_init = np.random.normal(0, 1, (NUNITS[0], NUNITS[1])).astype(np.float32)
W2_vs_glorot2_init = np.random.normal(0, 1, (NUNITS[1], NUNITS[2])).astype(np.float32)
W3_vs_glorot2_init = np.random.normal(0, 1, (NUNITS[2], NUNITS[3])).astype(np.float32)
W4_vs_glorot2_init = np.random.normal(0, 1, (NUNITS[3], NUNITS[4])).astype(np.float32)

W1_glorot2_init = (W1_vs_glorot2_init / np.sqrt((NUNITS[0] + NUNITS[1]) / 2)).astype(np.float32)
W2_glorot2_init = (W2_vs_glorot2_init / np.sqrt((NUNITS[1] + NUNITS[2]) / 2)).astype(np.float32)
W3_glorot2_init = (W3_vs_glorot2_init / np.sqrt((NUNITS[2] + NUNITS[3]) / 2)).astype(np.float32)
W4_glorot2_init = (W4_vs_glorot2_init / np.sqrt((NUNITS[3] + NUNITS[4]) / 2)).astype(np.float32)


W_glorot_init = {'W1_glorot_init': W1_glorot_init, 'W2_glorot_init': W2_glorot_init, 'W3_glorot_init': W3_glorot_init, 'W4_glorot_init': W4_glorot_init}
W_glorot2_init = {'W1_glorot2_init': W1_glorot2_init, 'W2_glorot2_init': W2_glorot2_init, 'W3_glorot2_init': W3_glorot2_init, 'W4_glorot2_init': W4_glorot2_init}
W_vs_glorot2_init = {'W1_vs_glorot2_init': W1_vs_glorot2_init, 'W2_vs_glorot2_init': W2_vs_glorot2_init, 'W3_vs_glorot2_init': W3_vs_glorot2_init, 'W4_vs_glorot2_init': W4_vs_glorot2_init}




with open('W_glorot_init', mode='wb') as f:
    pickle.dump(W_glorot_init, f)
    print(W1_glorot_init)

with open('W_glorot2_init', mode= 'wb') as f:
        pickle.dump(W_glorot2_init,f)
        print(W1_glorot2_init)

with open('W_vs_glorot2_init', mode='wb') as f:
        pickle.dump(W_vs_glorot2_init, f)
        print(W1_vs_glorot2_init)

#Xavier2
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

#He
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

with open('W_he_init', mode='wb') as f:
    pickle.dump(W_he_init, f)
    print(W1_he_init)

with open('W_vs_he_init', mode='wb') as f:
    pickle.dump(W_vs_he_init, f)
    print(W1_vs_he_init)

#He2
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