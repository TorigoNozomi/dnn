# coding: utf-8

import _pickle as cPickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

DIR = '../data/cifar10/cifar-10-batches-py/'
FILE_TRAIN = 'data_batch_%s'
FILE_TEST = 'test_batch'


class DataCifar10:
    def __init__(self):
        pass

    def main(self):
        # Training
        train_data = np.empty((0, 3072))
        train_label = np.empty((0, 10))
        for i in range(1, 6):
            print('read %s' % FILE_TRAIN % i)
            # data
            train_data_1 = self.unpickle(DIR + FILE_TRAIN % i)['data']
            train_data = np.concatenate((train_data, train_data_1), axis=0)
            # labels
            train_label_1 = self.unpickle(DIR + FILE_TRAIN % i)['labels']
            train_label_1 = self.onehot(train_label_1)
            train_label = np.concatenate((train_label, train_label_1), axis=0)

        # Testing
        print('read %s' % FILE_TEST)
        # data
        test_data = self.unpickle(DIR + FILE_TEST)['data']
        # labels
        test_label = self.unpickle(DIR + FILE_TEST)['labels']
        test_label = self.onehot(test_label)

        # Collect
        data = [train_data, train_label, test_data, test_label]

        return data

    def unpickle(self, f):
        fo = open(f, 'rb')
        d = cPickle.load(fo)
        fo.close()

        return d

    def onehot(self, X):
        X = np.array(X).reshape(1, -1)
        X = X.transpose()
        encoder = OneHotEncoder(n_values=max(X) + 1)
        X = encoder.fit_transform(X).toarray()

        return X

    def test(self):
        data = self.main()

        print('')
        print('Training Data: %s columns, %s records' % (data[0].shape[1], data[0].shape[0]))
        print(data[0])
        print('')
        print('Training Labels: %s columns, %s records' % (data[1].shape[1], data[1].shape[0]))
        print(data[1])
        print('')

        print('Test Data: %s columns, %s records' % (data[2].shape[1], data[2].shape[0]))
        print(data[2])
        print('')
        print('Test Labels: %s columns, %s records' % (data[3].shape[1], data[3].shape[0]))
        print(data[3])
        print('')


if __name__ == "__main__":
    DataCifar10().test()