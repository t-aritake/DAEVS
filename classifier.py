import numpy
import pdb
import torch


class Model(torch.nn.Module):
    def __init__(self, hidden_features_list, activations_list):
        super().__init__()
        layers = []
        for i in range(len(hidden_features_list)-1):
            layers.append(
                torch.nn.Linear(hidden_features_list[i], hidden_features_list[i+1]))
            layers.append(self._activation(activations_list[i]))

        self.module_list = torch.nn.ModuleList(layers)

    def _activation(self, type_of_activation):
        if type_of_activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif type_of_activation == 'relu':
            return torch.nn.ReLU()
        elif type_of_activation == 'softmax':
            return torch.nn.Softmax(dim=1)
        return

    def forward(self, x):
        for i in range(len(self.module_list)-1):
            f = self.module_list[i]
            if f is None:
                continue
            x = f(x)
        x[(x == 0).any(1)] += 1e-9
        x = self.module_list[-1](x)

        return x


class NeuralNetwork(object):
    def __init__(self, hidden_features_list, activations_list):
        self._model = None
        self._hidden_features_list = hidden_features_list
        self._activations_list = activations_list

    def fit(self, X, y, num_epochs=10):
        if self._model is None:
            self._model = Model(
                [X.shape[1]] + self._hidden_features_list + [1],
                self._activations_list)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-2)
        # self._optimizer = torch.optim.SGD(self._model.parameters(), lr=1e-2)

        # loss_func = torch.nn.CrossEntropyLoss()
        loss_func = torch.nn.BCELoss()

        y = numpy.copy(y)
        # y[y > 0] = 1
        # y[y < 0] = 0

        if type(X) == numpy.ndarray:
            X = torch.from_numpy(X.astype(numpy.float32))
        if type(y) == numpy.ndarray:
            y = torch.from_numpy(y.astype(numpy.float32))
        for epoch in range(num_epochs):
            self._optimizer.zero_grad()
            output = self._model(X).squeeze()
            loss = loss_func(output, y)
            loss.backward()
            self._optimizer.step()
            print(loss.item())

    def predict(self, X):
        self._model.eval()
        if type(X) == numpy.ndarray:
            X = torch.from_numpy(X.astype(numpy.float32))

        y = self._model(X).detach().numpy()

        if y.shape[1] == 1:
            y = y.squeeze()
            y[y >= 0.5] = 1
            y[y < 0.5] = 0

        return y

    def decision_function(self, X):
        self._model.eval()
        if type(X) == numpy.ndarray:
            X = torch.from_numpy(X.astype(numpy.float32))

        y = self._model(X).detach().numpy()
        if y.shape[1] == 1:
            y = y.squeeze()

        return y


class SGDSVC(object):

    def __init__(self,
                 kernel="rbf", lmd=1e-1, gamma=0.1, bias=1.0, max_iter=100):
        if kernel not in self.__kernel_dict:
            print(kernel + " kernel does not exist!\nUse rbf kernel.")
            kernel = "rbf"
        if kernel == "rbf":
            def kernel_func(x, y):
                return self.__kernel_dict[kernel](x, y, gamma=gamma)
        else:
            kernel_func = self.__kernel_dict[kernel]
        self.kernel = kernel_func
        self.lmd = lmd
        self.bias = bias
        self.max_iter = max_iter

    def __linear_kernel(x, y):
        return numpy.dot(x, y)

    def __gaussian_kernel(x, y, gamma):
        diff = x - y
        return numpy.exp(-gamma * numpy.dot(diff, diff))

    __kernel_dict = {"linear": __linear_kernel, "rbf": __gaussian_kernel}

    def fit(self, X, y):
        def update_alpha(alpha, t):
            data_size, feature_size = numpy.shape(self.X_with_bias)
            new_alpha = numpy.copy(alpha)
            it = numpy.random.randint(low=0, high=data_size)
            x_it = self.X_with_bias[it]
            y_it = self.y[it]
            if (y_it *
                (1. / (self.lmd * t)) *
                sum([alpha_j * y_it * self.kernel(x_it, x_j)
                     for x_j, alpha_j in zip(self.X_with_bias, alpha)])) < 1.:
                new_alpha[it] += 1
            return new_alpha

        self.X_with_bias = numpy.c_[
                X, numpy.ones((numpy.shape(X)[0])) * self.bias]
        self.y = y
        alpha = numpy.zeros((numpy.shape(self.X_with_bias)[0], 1))
        for t in range(1, self.max_iter + 1):
            alpha = update_alpha(alpha, t)
        self.alpha = alpha

    def decision_function(self, X):
        X_with_bias = numpy.c_[X, numpy.ones((numpy.shape(X)[0])) * self.bias]
        y_score = [(1. / (self.lmd * self.max_iter)) *
                   sum([alpha_j * y_j * self.kernel(x_j, x)
                        for (x_j, y_j, alpha_j) in zip(
                        self.X_with_bias, self.y, self.alpha)])
                   for x in X_with_bias]
        return numpy.array(y_score)

    def predict(self, X):
        y_score = self.decision_function(X)
        y_predict = map(lambda s: 1 if s >= 0. else -1, y_score)
        return y_predict


class OnlineLinearSVC(object):
    def __init__(self, C=0.01, fit_intercept=True,
                 warm_start=False):
        self.coef_ = None
        self.intercept_ = 0
        self._C = C
        self._fit_intercept = fit_intercept
        self._warm_start = warm_start

    def fit(self, X, y, num_epochs=1):
        if self.coef_ is None or not self._warm_start:
            self.coef_ = numpy.zeros(X.shape[1])

        indices = numpy.arange(0, X.shape[0])
        numpy.random.shuffle(indices)

        for count, idx in enumerate(indices):
            tmp_loss = self.loss(X, y)
            eta = self._C / (count+1)
            if tmp_loss[idx] == 0:
                self.coef_ -= eta * self.coef_
                if self._fit_intercept:
                    self.intercept_ -= eta * self.intercept_
                continue

            self.coef_ -= eta * (-self._C * y[idx] * X[idx] + self.coef_)
            if self._fit_intercept:
                self.intercept_ -= eta * (-y[idx] + self._C * self.intercept_)

        tmp_loss = self.loss(X, y)
        return tmp_loss

    def decision_function(self, X):
        return X.dot(self.coef_) + self.intercept_

    def loss(self, X, y):
        pred = X.dot(self.coef_) + self.intercept_
        return numpy.maximum(0, 1 - y * pred)

    def predict(self, X):
        return numpy.sign(self.decision_function(X))


class OnlineSVC(object):

    def __init__(self, kernel='rbf', C=1, fit_intercept=True,
                 max_iteration=10, degree=3, coef0=0.0, gamma='scale',
                 warm_start=False):
        self.alpha_ = None
        self.intercept_ = 0

        self._max_iteration = max_iteration

        self._degree = degree
        self._coef0 = coef0
        self._gamma = gamma
        self._d = degree

        self._C = C
        self._fit_intercept = fit_intercept

        self._warm_start = warm_start
        self._T = 0

        if kernel == 'linear':
            self._kernel = self._linear_kernel
        elif kernel == 'polynomial':
            self._kernel = self._polynomial_kernel
        elif kernel == 'rbf':
            self._kernel = self._rbf_kernel

    def fit(self, X, y):
        if self.alpha_ is None or not self._warm_start:
            self.alpha_ = numpy.zeros(X.shape[0])
            self._T = 0

        if self._gamma == 'scale':
            self._gamma = 1 / (X.shape[1] * X.var())
        if self._gamma == 'auto':
            self._gamma = 1 / X.shape[1]

        if self._fit_intercept:
            X = numpy.column_stack((X, numpy.ones(X.shape[0])))

        self._X = X
        self._y = y
        # calculate gram matrix
        G = self._calc_gram_matrix(X, X)

        for epoch in range(self._max_iteration):
            indices = numpy.arange(X.shape[0])
            numpy.random.shuffle(indices)

            for t, i in enumerate(indices):
                tmp_decision = self._C / (self._T + t + 1)\
                    * (y * self.alpha_).dot(G[:, i])

                if y[i] * tmp_decision < 1:
                    self.alpha_[i] += 1

            self._T += len(indices)

    def _calc_gram_matrix(self, X1, X2):
        X1 = self._check_2D_mat(X1)
        X2 = self._check_2D_mat(X2)

        return self._kernel(X1, X2)

        # G = numpy.zeros((X1.shape[0], X2.shape[0]))
        # for i in range(X1.shape[0]):
        #     for j in range(X2.shape[0]):
        #         G[i, j] = self._kernel(X1[i], X2[j])

        # return G

    def _check_2D_mat(self, X):
        if numpy.ndim(X) == 1:
            return X[None, :]
        # 3次元以上の配列になってしまっている場合は別のところで処理してもらう感じで
        return X

    def _linear_kernel(self, x, y):
        return x.dot(y.T)

    def _polynomial_kernel(self, x, y):
        return (self._gamma * x.dot(y.T) + self._coef0)**self._d

    def _rbf_kernel(self, x, y):
        xx, yy = numpy.indices((x.shape[0], y.shape[0]))
        dist = numpy.sum((x[xx] - y[yy])**2, 2)

        return numpy.exp(-self._gamma * dist)

    def decision_function(self, X):
        if X.ndim == 1:
            X = X[:, None]

        if self._fit_intercept:
            X = numpy.column_stack((X, numpy.ones(X.shape[0])))

        G = self._calc_gram_matrix(self._X, X)
        return self._C / self._T * (self.alpha_ * self._y).dot(G)

    def predict(self, X):
        return numpy.sign(self.decision_function(X))


if __name__ == '__main__':
    N1_s = 30
    N2_s = 70

    # cs = numpy.array([3, 8])

    # theta1 = numpy.random.uniform(0, 2 * numpy.pi, size=N1_s)
    # theta2 = numpy.random.uniform(0, 2 * numpy.pi, size=N2_s)
    # r1 = numpy.random.normal(3, 0.1, size=N1_s)
    # r2 = numpy.random.normal(1, 0.1, size=N2_s)
    # X1_s = numpy.column_stack((
    #     r1 * numpy.cos(theta1) + cs[0],
    #     r1 * numpy.sin(theta1) + cs[1]))
    # X2_s = numpy.column_stack((
    #     r2 * numpy.cos(theta2) + cs[0],
    #     r2 * numpy.sin(theta2) + cs[1]))
    # ys = numpy.concatenate((-numpy.ones(N1_s), numpy.ones(N2_s)))
    # Xs = numpy.concatenate((X1_s, X2_s))
    # import sklearn.datasets
    # iris = sklearn.datasets.load_iris()
    # Xs = iris.data[:100, :2]
    # ys = iris.target[:100]
    # ys[ys == 0] = -1
    # ys = 1 if ys == 1 else -1
    m1_s = numpy.array([-2, -1])
    m2_s = numpy.array([1, 3])
    S1_s = numpy.array([[0.8, -0.7], [-0.7, 1]])
    S2_s = numpy.array([[1.2, 0.4], [0.4, 1.3]])
    ys = numpy.concatenate((-numpy.ones(N1_s), numpy.ones(N2_s)))

    X1_s = numpy.random.multivariate_normal(m1_s, S1_s, size=N1_s)
    X2_s = numpy.random.multivariate_normal(m2_s, S2_s, size=N2_s)

    Xs = numpy.concatenate((X1_s, X2_s))

    svc = OnlineSVC(kernel='rbf', C=100, fit_intercept=True, max_iteration=100)
    # svc = SGDSVC(gamma = 1 / Xs.shape[0] / Xs.var())
    svc.fit(Xs, ys)
    print(svc.decision_function(Xs))

    import matplotlib.pyplot as plt

    plt.scatter(Xs[ys == -1, 0], Xs[ys == -1, 1])
    plt.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1])
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = numpy.linspace(xlim[0], xlim[1], 30)
    yy = numpy.linspace(ylim[0], ylim[1], 30)
    YY, XX = numpy.meshgrid(yy, xx)
    xy = numpy.vstack([XX.ravel(), YY.ravel()]).T
    Z = svc.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    plt.show()
