import numpy as np
import scipy.io
import sys
import os
import tensorflow as tf

def main():
    file_in = '../datasets/k.demo111_010.mat'
    mat_content = scipy.io.loadmat(file_in)
    print('%s loaded' % file_in)
    X_tra = mat_content['X_tra']
    y_tra = mat_content['y_tra']
    eta = 0.1
    nfeas = X_tra.shape[0]
    X_smbl = tf.placeholder(tf.float32, [None, nfeas])
    w_smbl = tf.Variable(tf.zeros([nfeas, 1]))
    b_smbl = tf.Variable(tf.zeros([1]))
    yhat_smbl = tf.matmul(X_smbl, w_smbl) + b_smbl
    ytru_smbl = tf.placeholder(tf.float32, [None, 1])
    loss_smbl = tf.reduce_mean(tf.square(yhat_smbl - ytru_smbl)) * 0.5
    optimizer = tf.train.GradientDescentOptimizer(eta)
    train = optimizer.minimize(loss_smbl)
    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)
    w0, b0, loss0 = sess.run([w_smbl, b_smbl, loss_smbl],
                             {X_smbl: X_tra.T, ytru_smbl: y_tra})
    sess.run(train, {X_smbl: X_tra.T, ytru_smbl: y_tra})
    w1, b1, loss1 = sess.run([w_smbl, b_smbl, loss_smbl],
                             {X_smbl: X_tra.T, ytru_smbl: y_tra})
    print('After initialization')
    print('(w,b,loss)=(%5.3f,%5.3f,%5.3f)' % (w0, b0, loss0))
    print('After first iterate')
    print('(w,b,loss)=(%5.3f,%5.3f,%5.3f)' % (w1, b1, loss1))


if __name__ == '__main__':
    main()

