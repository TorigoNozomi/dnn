import numpy as np
import scipy.io
import sys
import tensorflow as tf


def main():
    file_in = '../datasets/k.demo111_010.mat'
    mat_content = scipy.io.loadmat(file_in)
    print('%s loaded' % file_in)
    X_tra = mat_content['X_tra']
    y_tra = mat_content['y_tra']
    ntras = X_tra.shape[1]
    eta = 0.1
    lam = 1.0 / ntras
    nfeas = X_tra.shape[0]
    X_smbl = tf.placeholder(tf.float32, [None, nfeas])
    w_smbl = tf.Variable(tf.zeros([nfeas, 1]))
    b_smbl = tf.Variable(tf.zeros([1]))
    u_smbl = tf.matmul(X_smbl, w_smbl) + b_smbl
    ytru_smbl = tf.placeholder(tf.float32, [None, 1])
    loss_smbl = tf.nn.sigmoid_cross_entropy_with_logits(logits=u_smbl, labels=(ytru_smbl + 1) / 2)
    reg_smbl = tf.nn.l2_loss(w_smbl) + tf.nn.l2_loss(b_smbl)
    obj_smbl = tf.reduce_mean(loss_smbl) + lam * reg_smbl
    obj_smbl_summary = tf.summary.scalar('obj_smbl',obj_smbl)
    optimizer = tf.train.GradientDescentOptimizer(eta)
    train = optimizer.minimize(obj_smbl)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    train_writer = tf.summary.FileWriter('../output/train', sess.graph)
    test_writer = tf.summary.FileWriter('../output/test', sess.graph)

    correct_prediction = tf.equal(tf.sign(loss_smbl - 0.5), tf.sign(ytru_smbl - 0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    test_accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    niters = 10000
    Wb_rec = np.zeros([nfeas + 1, niters])
    obj_rec = np.zeros([niters])
    for itera in range(niters):
        w_cur, b_cur, obj_cur = sess.run([w_smbl, b_smbl, obj_smbl],
                                         {X_smbl: X_tra.T, ytru_smbl: y_tra})
        sess.run(train, {X_smbl: X_tra.T, ytru_smbl: y_tra})
        Wb_rec[0, itera] = w_cur
        Wb_rec[-1, itera] = b_cur
        obj_rec[itera] = obj_cur

        if itera % 100 == 0:
            print('After %d iterate' % itera)
            print('(w,b,loss)=(%5.3f,%5.3f,%5.3f)' % (w_cur, b_cur, obj_cur))
            print('step %d' % itera)

    #Output W,b,obj as a mat file
    #file_out = 'k.demo111lr_080.mat'
    #scipy.io.savemat(file_out, {'Wb_tf': Wb_rec, 'obj_tf': obj_rec})
    #print('%s written.' % file_out)


if __name__ == '__main__':
    main()
