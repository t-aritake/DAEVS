# -*- coding: utf-8 -*-
import numpy
import scipy.stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb


def plot2D(ax, X, y, markers=['x', 'v'], colors=['C0', 'C1'],
           classifier=None, levels=[-1, 0, 1]):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], marker=markers[0], alpha=0.5,
               color=colors[0])
    ax.scatter(X[y == 1, 0], X[y == 1, 1], marker=markers[1], alpha=0.5,
               color=colors[1])
    # ax.scatter(Xt[yt == 0, 0], Xt[yt == 0, 1])
    # ax.scatter(Xt[yt == 1, 0], Xt[yt == 1, 1])
    ax.set_xlabel('xc', fontsize=15)
    ax.set_ylabel('xe', fontsize=15)

    if classifier is None:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = numpy.linspace(xlim[0], xlim[1], 30)
    yy = numpy.linspace(ylim[0], ylim[1], 30)
    YY, XX = numpy.meshgrid(yy, xx)
    xy = numpy.vstack([XX.ravel(), YY.ravel()]).T
    Z = classifier.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=levels, alpha=0.5,
               linestyles=['--', '-', 'dotted'])


def plot_data(ax, X, y):
    Xneg = X[y == 0]
    Xpos = X[y == 1]

    ax.scatter(Xneg[:, 0], Xneg[:, 1], marker='x', alpha=0.5, color='C0')
    ax.scatter(Xpos[:, 0], Xpos[:, 1], marker='v', alpha=0.5, color='C1')

    divider = make_axes_locatable(ax)

    axx = divider.append_axes("top", size="10%", pad=0.1, sharex=ax)
    axy = divider.append_axes("right", size="10%", pad=0.1, sharey=ax)

    xc_neg_kde = scipy.stats.gaussian_kde(Xneg[:, 0])
    xc_pos_kde = scipy.stats.gaussian_kde(Xpos[:, 0])
    xe_neg_kde = scipy.stats.gaussian_kde(Xneg[:, 1])
    xe_pos_kde = scipy.stats.gaussian_kde(Xpos[:, 1])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = numpy.linspace(xlim[0], xlim[1], 150)
    yy = numpy.linspace(ylim[0], ylim[1], 150)

    axx.xaxis.set_tick_params(labelbottom=False)
    axy.yaxis.set_tick_params(labelleft=False)

    axx.plot(xx, xc_neg_kde(xx))
    axx.plot(xx, xc_pos_kde(xx))
    axy.plot(xe_neg_kde(yy), yy)
    axy.plot(xe_pos_kde(yy), yy)

    return ax, axx, axy


def plot3D(ax, Xs, ys, Xt, yt, classifier):
    ax.scatter3D(Xs[ys == 0, 0], Xs[ys == 0, 1], Xs[ys == 0, 2])
    ax.scatter3D(Xs[ys == 1, 0], Xs[ys == 1, 1], Xs[ys == 1, 2])
    ax.scatter3D(Xt[yt == 0, 0], Xt[yt == 0, 1], Xt[yt == 0, 2])
    ax.scatter3D(Xt[yt == 1, 0], Xt[yt == 1, 1], Xt[yt == 1, 2])

    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z', fontsize=15)


class LabelConverter(object):
    def __init__(self):
        self._unique = None

    def encode(self, labels, update_unique_elements=False):
        if self._unique is None or update_unique_elements:
            unique, unique_inverse = numpy.unique(labels, return_inverse=True)
            self._unique = unique
        unique_inverse = numpy.where(labels[:, None] == self._unique)[1]
        one_hot_labels = numpy.eye(len(self._unique))[unique_inverse]

        return one_hot_labels.astype(int)

    def get_index(self, label):
        if type(label) != numpy.ndarray and type(label) != list:
            label = numpy.array([label, ])
        elif type(label) == list:
            label = numpy.array(label)

        return numpy.where(label[:, None] == self._unique)[1]

    def decode(self, one_hot_vectors):
        inverse_index = numpy.where(one_hot_vectors == 1)[1]
        return self._unique[inverse_index]

    def hard_decode(self, soft_vectors, random=False):
        if random:
            inverse_index = self._get_random_inverse(soft_vectors)
        else:
            inverse_index = numpy.argmax(soft_vectors, axis=1)
        return self._unique[inverse_index]

    def _get_random_inverse(self, soft_vectors):
        one_hot_vectors = numpy.zeros_like(soft_vectors).astype(int)
        for i in range(len(soft_vectors)):
            idx = numpy.argwhere(
                soft_vectors[i] == soft_vectors[i].max()).squeeze()
            if len(idx) == 1:
                one_hot_vectors[i, idx] = 1
            elif len(idx) > 1:
                idx = numpy.random.choice(idx)
                one_hot_vectors[i, idx] = 1
        inverse_index = numpy.argmax(one_hot_vectors, axis=1)
        return inverse_index


if __name__ == '__main__':
    labels = [1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1]
    converter = LabelConverter()
    one_hot = converter.encode(labels)

    one_hot = numpy.abs(one_hot - 0.5)
    decoded = converter.decode(one_hot)
