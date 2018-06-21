import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import copy
# Visualize objective for neural network with single unit on each layers.

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def deriv_tanh(u):
    return 1-np.tanh(u)**2

def relu(u):
    return np.array(list(map(lambda e: max(0,e),u)))

def deriv_relu(u):
    return relu(np.sign(u))

def deriv_sigmoid(u):
    # vector -> vector
    return (1 - sigmoid(u)) * sigmoid(u)


def create_data_norm(shape, mean_range=(0, 0), std_range=(1, 1)):
    # 人工データ生成
    ndata = shape[0]
    dim = shape[1:]

    def create_data1(sh):
        if not len(sh):
            mean = np.random.uniform(mean_range[0], mean_range[1])
            std = np.random.uniform(std_range[0], std_range[0])
            return np.random.normal(mean, std, ndata)
        else:
            return [create_data1(sh[1:]) for _ in range(sh[0])]

    return np.asarray(create_data1(dim)).transpose((len(shape) - 1,) + tuple(range(len(shape) - 1)))


class NetSigmoid():
    def __init__(self, w_ini):
        self.w_cur = w_ini
        self.loss_cur = 0
        self.hist = {'loss':[], 'weight':[]}

    @staticmethod
    def loss(X, Y, w):
        u1 = X * w[0]
        z1 = sigmoid(u1)
        u2 = z1 * w[1]
        z2 = sigmoid(u2)

        val_loss = -np.mean(Y * np.log(z2) + (1 - Y) * np.log(1 - z2))
        return val_loss

    @staticmethod
    def gradient(X, Y, w):
        u1 = X * w[0]
        z1 = sigmoid(u1)
        u2 = z1 * w[1]
        z2 = sigmoid(u2)

        dldw1 = np.mean((z2 - Y) * w[1] * deriv_sigmoid(u1) * X)
        dldw2 = np.mean((z2 - Y) * z1)

        return np.array([dldw1, dldw2])

    def sgd_update(self, X, Y, rho=1):
        self.w_cur -= rho * self.gradient(X, Y, self.w_cur)
        self.loss_cur = self.loss(X, Y, self.w_cur)
        self.hist['loss'].append(copy.copy(self.loss_cur))
        self.hist['weight'].append(copy.copy(self.w_cur))

class NetRelu(NetSigmoid):
    @staticmethod
    def loss(X, Y, w):
        u1 = X * w[0]
        z1 = relu(u1)
        u2 = z1 * w[1]
        z2 = sigmoid(u2)

        val_loss = -np.mean(Y * np.log(z2) + (1 - Y) * np.log(1 - z2))
        return val_loss

    @staticmethod
    def gradient(X, Y, w):
        u1 = X * w[0]
        z1 = relu(u1)
        u2 = z1 * w[1]
        z2 = sigmoid(u2)

        dldw1 = np.mean((z2 - Y) * w[1] * deriv_relu(u1) * X)
        dldw2 = np.mean((z2 - Y) * z1)

        return np.array([dldw1, dldw2])

class NetTanh(NetSigmoid):
    @staticmethod
    def loss(X, Y, w):
        u1 = X * w[0]
        z1 = np.tanh(u1)
        u2 = z1 * w[1]
        z2 = sigmoid(u2)

        val_loss = -np.mean(Y * np.log(z2) + (1 - Y) * np.log(1 - z2))
        return val_loss

    @staticmethod
    def gradient(X, Y, w):
        u1 = X * w[0]
        z1 = np.tanh(u1)
        u2 = z1 * w[1]
        z2 = sigmoid(u2)

        dldw1 = np.mean((z2 - Y) * w[1] * deriv_tanh(u1) * X)
        dldw2 = np.mean((z2 - Y) * z1)

        return np.array([dldw1, dldw2])

def sgd():
    ### 設定
    # ウェイトの平面を決定するパラメータ
    np.random.seed(1)
    n_p = 6
    n_n = 3

    # 人工データの生成
    X_p = create_data_norm([n_p], mean_range=[-2, 2], std_range=[0, 0.1])
    X_n = create_data_norm([n_n], mean_range=[-2, 2], std_range=[0, 0.1])
    X = np.concatenate((X_p, X_n))
    Y = np.concatenate((np.zeros(len(X_p)), np.ones(len(X_n))))

    # ウェイトの平面を定義
    xlim = [-7, 3]
    ylim = [-7, 3]
    xarange = np.arange(xlim[0], xlim[1], 0.25)
    yarange = np.arange(ylim[0], ylim[1] , 0.25)
    w_1, w_2 = np.meshgrid(xarange, yarange)
    W = np.asarray([w_1, w_2]).T

    # NNのインスタンス生成
    nets = [
        NetSigmoid(w_ini=[0, 2]),
        NetSigmoid(w_ini=[-1.5, 2]),
        NetSigmoid(w_ini=[-2.165, 2]),
        NetSigmoid(w_ini=[-4, 2])
    ]
    colors = ['r', 'y', 'c', 'k']

    loss_surf = np.asarray([[NetSigmoid.loss(X, Y, w) for w in h] for h in W])

    ### 図の作成
    fig = plt.figure(figsize=(8, 3))
    plt.subplots_adjust(wspace=0.4, hspace=0.6, bottom=0.2)

    ax3d = fig.add_subplot(121, projection='3d', azim=120, elev=40)
    ax3d.set_xlim(xlim[0], xlim[1])
    ax3d.set_ylim(ylim[0], ylim[1])
    ax3d.set_xlabel('w2')
    ax3d.set_ylabel('w1')
    ax3d.set_zlabel('loss')
    ax3d.axes.plot_surface(w_1, w_2, loss_surf)
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.set_xlabel('iter')
    ax2d.set_ylabel('loss')
    ax2d.get_ymajorticklabels()

    niter = 200
    flame = []
    for i in range(niter):
        item_plot = []
        for net, color in zip(nets, colors):
            net.sgd_update(X, Y)
            hist_w1, hist_w2= np.asarray(net.hist['weight']).T
            item_plot += ax3d.plot(hist_w2, hist_w1, net.hist['loss'], color + '-')
            item_plot += ax3d.plot([hist_w2[-1]], [hist_w1[-1]], [net.hist['loss'][-1]], color + 'o')
            item_plot += ax2d.plot(range(i+1), net.hist['loss'], color + '-')
            item_plot += ax2d.plot([i], [net.hist['loss'][-1]], color + 'o')

        flame.append(item_plot)
    anime = animation.ArtistAnimation(fig, flame)
    anime.save('sgd_sigmoid.gif', writer='imagemagick', fps=30)


if __name__ == '__main__':
    sgd()