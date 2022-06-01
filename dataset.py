# -*- coding: utf-8 -*-
import numpy
import pickle
import pandas


class LinearDataset(object):
    def __init__(
            self, source_mu_list, source_Sigma_list,
            target_mu_list, target_Sigma_list,
            Ns=1000, Nt=100,
            class_ratio=[0.5, 0.5], common_idx=None):
        self._source_mu_list = source_mu_list
        self._source_Sigma_list = source_Sigma_list
        self._target_mu_list = target_mu_list
        self._target_Sigma_list = target_Sigma_list
        self._Ns = Ns
        self._Nt = Nt
        self._class_ratio = class_ratio
        self._common_idx = common_idx

    def gen_data(self):
        Xs, ys = self._gen_linear_data(
            self._Ns, self._source_mu_list, self._source_Sigma_list)
        Xt, yt = self._gen_linear_data(
            self._Nt, self._target_mu_list, self._target_Sigma_list)

        if self._common_idx is None:
            self._common_idx = numpy.arange(Xs.shape[1])

        return Xs, ys, Xt, yt, self._common_idx

    def _gen_linear_data(self, N, mu_list, Sigma_list):
        # クラスタごとの平均値と分散共分散行列を入力する形
        N_list = (N * numpy.array(self._class_ratio)).astype(int)
        X_list = []
        y_list = []

        for i in range(len(self._class_ratio)):
            X = numpy.random.multivariate_normal(
                mu_list[i], Sigma_list[i], size=N_list[i])
            y = numpy.ones(N_list[i]) * i
            X_list.append(X)
            y_list.append(y)

        return numpy.row_stack(X_list), numpy.concatenate(y_list)


class AffineDataset(object):
    def __init__(self, Xs, ys,A, b, common_idx=None):
        self._Xs = Xs
        self._ys = ys
        self._A = A
        self._b = b

        self._common_idx = common_idx

    def gen_data(self):
        Xt = self._Xs.dot(self._A.T) + self._b

        if self._common_idx is None:
            self._common_idx = numpy.arange(self._Xs.shape[1])

        return self._Xs, self._ys, Xt, self._ys, self._common_idx


class TwoBallsDataset(object):
    def __init__(self, Ns=1000, Nt=1000, class_ratio=[0.5, 0.5],
                 radius_list=[2, 3], radius_std_list=[0.2, 0.1],
                 source_mu_list=numpy.array([0, 0]),
                 source_scales_list=numpy.array([1, 1]),
                 target_mu_list=numpy.array([3, 1]),
                 target_scales_list=numpy.array([1, 1]),
                 common_idx=None, only_common=False):
        # 2クラスであると考える．radius_listやradius_std_listは2クラス分数があればよい
        # source_mu_listなどは次元に応じた要素数で与えるものとする
        self._Ns = Ns
        self._Nt = Nt
        self._class_ratio = class_ratio
        # radius_list等は要素数2ならばsource, targetで同じ半径を利用
        # そうでなければ要素数4にして，source, targetで異なる半径とする
        if len(radius_list) == 2:
            self._radius_list = numpy.append(radius_list, radius_list)
        else:
            self._radius_list = radius_list
        if len(radius_std_list) == 2:
            self._radius_std_list =\
                numpy.append(radius_std_list, radius_std_list)
        else:
            self._radius_std_list = radius_std_list
        self._source_mu_list = source_mu_list
        self._source_scales_list = source_scales_list
        self._target_mu_list = target_mu_list
        self._target_scales_list = target_scales_list
        self._common_idx = common_idx
        self._only_common = only_common

    def gen_data(self):
        Xs, ys = self._gen_two_balls(
            self._Ns, self._class_ratio,
            self._radius_list[:2], self._radius_std_list[:2],
            self._source_mu_list, self._source_scales_list)
        Xt, yt = self._gen_two_balls(
            self._Nt, self._class_ratio,
            self._radius_list[2:], self._radius_std_list[2:],
            self._target_mu_list, self._target_scales_list)

        return Xs, ys, Xt, yt, self._common_idx

    def _gen_two_balls(self, N, class_ratio, radius_list, radius_std_list,
                       mu_list, scales_list):
        assert len(mu_list[0]) == len(scales_list[0]),\
            "length of averages and standard deviation should be the same"

        dim = len(mu_list[0])
        num_samples = (N * numpy.array(class_ratio)).astype(int)

        # n次元の場合はn-1個の角度パラメータが必要
        theta1 = numpy.random.uniform(
            0, 2 * numpy.pi, size=(num_samples[0], dim-1))
        theta2 = numpy.random.uniform(
            0, 2 * numpy.pi, size=(num_samples[1], dim-1))

        cos1 = numpy.column_stack(
            (numpy.cos(theta1), numpy.ones(num_samples[0])))
        sin1 = numpy.column_stack(
            (numpy.ones(num_samples[0]), numpy.sin(theta1)))
        cos2 = numpy.column_stack(
            (numpy.cos(theta2), numpy.ones(num_samples[1])))
        sin2 = numpy.column_stack(
            (numpy.ones(num_samples[1]), numpy.sin(theta2)))

        X1 = numpy.cumprod(sin1, axis=1) * cos1
        X2 = numpy.cumprod(sin2, axis=1) * cos2

        # 半径はスカラーでOK
        r1 = numpy.random.normal(
            self._radius_list[0], self._radius_std_list[0],
            size=(num_samples[0], 1))
        r2 = numpy.random.normal(
            self._radius_list[1], self._radius_std_list[1],
            size=(num_samples[1], 1))

        X1 = r1 * scales_list[0] * X1 + mu_list[0]
        X2 = r2 * scales_list[1] * X2 + mu_list[1]

        X = numpy.concatenate((X1, X2), axis=0)
        y = numpy.concatenate(
            (numpy.zeros(num_samples[0]), numpy.ones(num_samples[1])))

        return X, y


class TwoCircleDataset(object):
    def __init__(self, Ns=1000, Nt=1000, class_ratio=[0.5, 0.5],
                 radius_list=[2, 3], radius_std_list=[0.2, 0.1],
                 source_mu_list=numpy.array([0, 0]),
                 source_scales_list=numpy.array([1, 1]),
                 target_mu_list=numpy.array([3, 1]),
                 target_scales_list=numpy.array([1, 1]),
                 common_idx=None):
        self._Ns = Ns
        self._Nt = Nt
        self._class_ratio = class_ratio
        if len(radius_list) == 2:
            self._radius_list = numpy.append(radius_list, radius_list)
        else:
            self._radius_list = radius_list
        if len(radius_std_list) == 2:
            self._radius_std_list =\
                numpy.append(radius_std_list, radius_std_list)
        else:
            self._radius_std_list = radius_std_list
        self._source_mu_list = source_mu_list
        self._source_scales_list = source_scales_list
        self._target_mu_list = target_mu_list
        self._target_scales_list = target_scales_list
        self._common_idx = common_idx

    def gen_data(self):
        Xs, ys = self._gen_two_circles(
            self._Ns, self._class_ratio,
            self._radius_list[:2], self._radius_std_list[:2],
            self._source_mu_list, self._source_scales_list)
        Xt, yt = self._gen_two_circles(
            self._Nt, self._class_ratio,
            self._radius_list[2:], self._radius_std_list[2:],
            self._target_mu_list, self._target_scales_list)

        return Xs, ys, Xt, yt, self._common_idx

    def _gen_two_circles(self, N, class_ratio, radius_list, radius_std_list,
                         mu_list, scales_list):
        num_samples = (N * numpy.array(class_ratio)).astype(int)
        theta1 = numpy.random.uniform(0, 2 * numpy.pi, size=num_samples[0])
        theta2 = numpy.random.uniform(0, 2 * numpy.pi, size=num_samples[1])
        r1 = numpy.random.normal(
            self._radius_list[0], self._radius_std_list[0], size=num_samples[0])
        r2 = numpy.random.normal(
            self._radius_list[1], self._radius_std_list[1], size=num_samples[1])

        X1 = numpy.column_stack(
            (r1 * numpy.cos(theta1), r1 * numpy.sin(theta1)))
        X2 = numpy.column_stack(
            (r2 * numpy.cos(theta2), r2 * numpy.sin(theta2)))

        X1 = X1 * scales_list[0] + mu_list[1]
        X2 = X2 * scales_list[1] + mu_list[1]

        X = numpy.concatenate((X1, X2), axis=0)
        y = numpy.concatenate((numpy.zeros(num_samples[0]), numpy.ones(num_samples[1])))

        return X, y


class TwoSpiralsDataset(object):
    def __init__(self, Ns=1000, Nt=1000, class_ratio=[0.5, 0.5],
                 radius=2.4*numpy.pi, start=numpy.pi/2, noise=0.2,
                 Xs_center=[0, 0], Xt_center=[0, 0],
                 Xs_scale=[1, 1], Xt_scale=[1, 1],
                 common_idx=None, only_common=False):
        self._Ns = Ns
        self._Nt = Nt
        self._class_ratio = class_ratio
        self._radius = radius
        self._start = start
        self._noise = noise

        self._Xs_center = Xs_center
        self._Xt_center = Xt_center
        self._Xs_scale = Xs_scale
        self._Xt_scale = Xt_scale
        self._common_idx = common_idx
        self._only_common = only_common

    def gen_data(self):
        # ここで生成したデータを複数のモデルで使いたいタイミングがあるかもしれないがとりあえず考えない
        Xs, ys = self._gen_two_spirals(self._Ns, self._class_ratio)
        Xt, yt = self._gen_two_spirals(self._Nt, self._class_ratio)

        Xs = Xs * self._Xs_scale + self._Xs_center
        Xt = Xt * self._Xt_scale + self._Xt_center

        return Xs, ys, Xt, yt, self._common_idx

    def _gen_two_spirals(self, N, class_ratio):
        num_samples = (N * numpy.array(self._class_ratio)).astype(int)
        # 円の半径が小さい方が密になるようにsqrtを一様分布からのサンプルにかけている
        theta = self._start + numpy.sqrt(numpy.random.random(size=num_samples[0]))\
            * self._radius
        d1 = numpy.array([
            -numpy.cos(theta) * theta +
            numpy.random.random(size=num_samples[0]) * self._noise,
            numpy.sin(theta) * theta +
            numpy.random.random(size=num_samples[0]) * self._noise]).T
        y1 = numpy.zeros(shape=(d1.shape[0]))

        theta = self._start + numpy.sqrt(numpy.random.random(size=num_samples[1]))\
            * self._radius
        d2 = numpy.array([
            numpy.cos(theta) * theta +
            numpy.random.random(size=num_samples[1]) * self._noise,
            -numpy.sin(theta) * theta +
            numpy.random.random(size=num_samples[1]) * self._noise]).T
        y2 = numpy.ones(shape=(d2.shape[0]))

        X = numpy.concatenate((d1, d2), axis=0)
        y = numpy.concatenate((y1, y2), axis=0)

        return X, y


class TwoMoonsDataset(object):
    def __init__(self, Ns=1000, Nt=1000, class_ratio=[0.5, 0.5],
                 radius=0.5, noise=0.1,
                 Xs_shift=[0, 0], Xt_shift=[0, 0],
                 Xs_scale=[1, 1], Xt_scale=[1, 1],
                 rotation=None, common_idx=None, only_common=False):
        self._Ns = Ns
        self._Nt = Nt
        self._class_ratio = class_ratio
        self._radius = radius
        self._noise = noise

        self._Xs_shift = Xs_shift
        self._Xt_shift = Xt_shift
        self._Xs_scale = Xs_scale
        self._Xt_scale = Xt_scale
        self._rotation = rotation
        self._common_idx = common_idx
        self._only_common = only_common

    def gen_data(self):
        # ここで生成したデータを複数のモデルで使いたいタイミングがあるかもしれないがとりあえず考えない
        Xs, ys = self._gen_two_moons(self._Ns)
        Xt, yt = self._gen_two_moons(self._Nt)

        Xs = Xs * self._Xs_scale + self._Xs_shift
        Xt = Xt * self._Xt_scale + self._Xt_shift

        return Xs, ys, Xt, yt, self._common_idx

    def _gen_two_moons(self, N):
        num_samples = (N * numpy.array(self._class_ratio)).astype(int)
        # 円の半径が小さい方が密になるようにsqrtを一様分布からのサンプルにかけている
        theta = numpy.linspace(0, numpy.pi, num=num_samples[0])
        d1 = numpy.array([
            self._radius * numpy.cos(theta) +
            numpy.random.normal(size=num_samples[0]) * self._noise,
            self._radius * numpy.sin(theta) +
            numpy.random.normal(size=num_samples[0]) * self._noise]).T
        y1 = numpy.zeros(shape=(d1.shape[0]))

        theta = numpy.linspace(0, numpy.pi, num=num_samples[1])
        d2 = numpy.array([
            self._radius * numpy.cos(theta) +
            numpy.random.normal(size=num_samples[1]) * self._noise,
            -self._radius * numpy.sin(theta) +
            numpy.random.normal(size=num_samples[1]) * self._noise]).T
        y2 = numpy.ones(shape=(d2.shape[0]))

        d2 += [self._radius, self._radius / 2]

        X = numpy.concatenate((d1, d2), axis=0)
        y = numpy.concatenate((y1, y2), axis=0)

        if self._rotation is not None:
            rot = numpy.array([
                [numpy.cos(self._rotation), -numpy.sin(self._rotation)],
                [numpy.sin(self._rotation), numpy.cos(self._rotation)]])
            X = rot.dot(X.T).T

        return X, y


class DatasetFromBoundary(object):
    def __init__(
            self, boundary, source_mu_list, source_Sigma_list,
            target_mu_list, target_Sigma_list, shift_list,
            noise=0.1, target_transpose=0.0,
            Ns=1000, Nt=1000, class_ratio=[0.5, 0.5],
            common_idx=None):
        self._boundary = boundary
        self._source_mu_list = source_mu_list
        self._source_Sigma_list = source_Sigma_list
        self._target_mu_list = target_mu_list
        self._target_Sigma_list = target_Sigma_list
        self._shift_list = shift_list
        self._noise = noise
        self._target_transpose = target_transpose
        self._Ns = Ns
        self._Nt = Nt
        self._class_ratio = class_ratio
        self._common_idx = common_idx

    def gen_data(self):
        # ここで生成したデータを複数のモデルで使いたいタイミングがあるかもしれないがとりあえず考えない
        Xs, ys = self._gen_data(
            self._Ns, self._source_mu_list, self._source_Sigma_list)
        Xt, yt = self._gen_data(
            self._Nt, self._target_mu_list, self._target_Sigma_list)

        Xt[:, 0] += self._target_transpose

        return Xs, ys, Xt, yt, self._common_idx

    def _gen_data(self, N, mu_list, Sigma_list):
        # クラスタごとの平均値と分散共分散行列を入力する形
        N_list = (N * numpy.array(self._class_ratio)).astype(int)
        X_list = []
        y_list = []

        for i in range(len(self._class_ratio)):
            x1 = numpy.random.normal(
                mu_list[i], Sigma_list[i], size=N_list[i])
            x2 = self._boundary(x1) + self._shift_list[i]

            X = numpy.column_stack((x1, x2))
            X += numpy.random.normal(size=X.shape) * self._noise
            y = numpy.ones(N_list[i]) * i
            X_list.append(X)
            y_list.append(y)

        return numpy.row_stack(X_list), numpy.concatenate(y_list)


class GasDataset(object):
    def __init__(self, target_labels, source_id, target_id, common_sensors, only_common=True):
        self._source_id = source_id
        self._target_id = target_id

        self._target_labels = target_labels

        # ここで使うfeaturesを選択
        transient_features = [i for i in range(8 * 16) if i % 8 > 1]
        self._common_idx = [
            x for x in transient_features if (x//8) in common_sensors]
        self._specific_idx =\
            list(set(transient_features).difference(self._common_idx))
        self._only_common = only_common

    def gen_data(self):
        with open('./datasets/GasSensorArrayDriftDataset/batch' +
                  str(self._source_id) + '.pkl', 'rb') as source:
            source_dataset = pickle.load(source)
        with open('./datasets/GasSensorArrayDriftDataset/batch' +
                  str(self._target_id) + '.pkl', 'rb') as source:
            target_dataset = pickle.load(source)
        Xs = source_dataset['features']
        Xt = target_dataset['features']
        ys = source_dataset['labels']
        yt = target_dataset['labels']
        Xs /= Xs.sum(0)
        Xt /= Xt.sum(0)
        idx = (ys == self._target_labels[0]) + (ys == self._target_labels[1])
        Xs = Xs[idx]
        ys = ys[idx]
        idx = (yt == self._target_labels[0]) + (yt == self._target_labels[1])
        Xt = Xt[idx]
        yt = yt[idx]
        ys[ys == self._target_labels[0]] = 0
        ys[ys == self._target_labels[1]] = 1
        yt[yt == self._target_labels[0]] = 0
        yt[yt == self._target_labels[1]] = 1

        # とりあえずここまででデータ読み込み（ラベルも指定したものに限定）
        Xs = Xs[:, self._common_idx]
        if self._only_common:
            Xt = Xt[:, self._common_idx]
        else:
            Xt = Xt[:, self._common_idx + self._specific_idx]

        return Xs, ys, Xt, yt, numpy.arange(len(self._common_idx))


class GenActivityData(object):
    def __init__(self, subject_num, common_sensors, target_labels=[11, 13],
                 only_common=True):

        # とりあえず加速度だけ使うことを考えている
        # ここで使うfeaturesを選択（各センサーの中の使う変数の選択）
        # 加速度，ジャイロ，磁気
        features_list = []
        # 0, 加速度, 1ジャイロ, 2磁気, 3方向
        feature_type = 0
        # センサ位置は9個
        for i in range(9):
            features_list += [
                j + feature_type * 15 + i * 65 for j in range(15)]
            # 方向の場合はこっち
            # transient_features += [
            #     j + feature_type * 15 + i * 65 for j in range(20)]
        self._common_idx = [
            x for x in features_list if (x//65) in common_sensors]
        self._specific_idx =\
            list(set(features_list).difference(self._common_idx))
        self._subject_num = subject_num
        self._only_common = only_common
        self._target_labels = target_labels

    def gen_data(self):
        with open('./datasets/realistic_sensor_displacement/subject_all_ideal.pkl', 'rb')\
                as reader:
            source_dataset = pickle.load(reader)
        with open('./datasets/realistic_sensor_displacement/subject' + str(self._subject_num)
                  + '_self.pkl', 'rb') as reader:
            target_dataset = pickle.load(reader)
        Xs = source_dataset['features']
        Xt = target_dataset['features']
        ys = source_dataset['labels']
        yt = target_dataset['labels']
        Xs /= Xs.sum(0)
        Xt /= Xt.sum(0)
        idx = (ys == self._target_labels[0]) + (ys == self._target_labels[1])
        Xs = Xs[idx]
        ys = ys[idx]
        idx = (yt == self._target_labels[0]) + (yt == self._target_labels[1])
        Xt = Xt[idx]
        yt = yt[idx]
        ys[ys == self._target_labels[0]] = 0
        ys[ys == self._target_labels[1]] = 1
        yt[yt == self._target_labels[0]] = 0
        yt[yt == self._target_labels[1]] = 1
        # とりあえずここまででデータ読み込み（ラベルも指定したものに限定）
        Xs = Xs[:, self._common_idx]
        if self._only_common:
            Xt = Xt[:, self._common_idx]
        else:
            Xt = Xt[:, self._common_idx + self._specific_idx]

        return Xs, ys, Xt, yt, numpy.arange(len(self._common_idx))


if __name__ == '__main__':
    source_mu_list = [[-2, -1], [1, 3]]
    source_Sigma_list = []
    source_Sigma_list.append(numpy.array([[0.8, -0.7], [-0.7, 1]]))
    source_Sigma_list.append(numpy.array([[1.2, 0.4], [0.4, 1.3]]))

    target_mu_list = [[1.5, -1], [-3, 3]]
    target_Sigma_list = []
    target_Sigma_list.append(numpy.array([[2.0, 0.7], [0.7, 1]]))
    target_Sigma_list.append(numpy.array([[1.2, 0.8], [0.8, 1.3]]))

    ds = TwoCircleDataset(Ns=1000, Nt=1000)
    Xs, ys, Xt, yt, common_idx = ds.gen_data()

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1])
    ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1])
    ax.scatter(Xt[yt == 0, 0], Xt[yt == 0, 1])
    ax.scatter(Xt[yt == 1, 0], Xt[yt == 1, 1])
    plt.savefig('scatter.png')
    plt.close()
