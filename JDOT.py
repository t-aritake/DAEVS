# -*- coding: utf-8 -*-
import ot
import numpy
import copy
import warnings
import sklearn.metrics
# import matplotlib.pyplot as plt
import utils
import pdb


class JointDistributionOptimalTransport(object):
    def __init__(self, classifier, loss_type='hinge', metric='sqeuclidean'):
        self._classifier = classifier
        self._transport_map_list = []
        self._classifier_list = []
        self._loss_type = loss_type
        self._metric = metric

        self._target_samples_log = []
        self._target_samples_log2 = []
        self._confidence_log = []

    def fit(self, source_variables, target_variables,
            source_label, alpha=1, num_iterations=100,
            confidence_threshold=None, sinkhorn_reg=0.0):
        ''' source_variablesとtarget_variablesの次元は一致すると仮定'''
        # 特徴量は sample x n_featureの形で持つようにする
        if source_variables.ndim == 1:
            source_variables = source_variables[:, None]
        if target_variables.ndim == 1:
            target_variables = target_variables[:, None]

        # if alpha is not list use same alpha for all iteration step
        if type(alpha) == int or type(alpha) == float:
            alpha = [alpha] * num_iterations

        self._source_variables = source_variables
        self._target_variables = target_variables

        self._label_converter = utils.LabelConverter()

        self._source_label = source_label

        # NOTE:一応ここを変えればやりたいことはできる
        dist = sklearn.metrics.pairwise_distances(
            source_variables, target_variables, metric=self._metric)
        # dist = dist**2
        # use only distance of variables for the first iteration
        cost = dist

        for iteration in range(num_iterations):
            # probability of empirical distributions
            hs = numpy.ones(len(source_variables)) / len(source_variables)
            ht = numpy.ones(len(target_variables)) / len(target_variables)

            # calculate transport map
            if sinkhorn_reg > 0:
                transport_map = ot.sinkhorn(
                    hs, ht, cost, sinkhorn_reg, numItermax=100000)
            elif sinkhorn_reg == 0.0:
                transport_map = ot.emd(hs, ht, cost)
            self._transport_map_list.append(transport_map)

            # generate data to train classifier
            # ニューラルの学習のためにbinary値かつmultilabelでone-hot表現を利用したい場合もあるのでは...？
            # 全部の場合にうまく対応するのは難しいような気もしてきた
            X, labels = self.barycentric_mapping()
            self._target_samples_log2.append((X, labels))
            confidence = self.calc_confidence()

            # 確信度でフィルタリングせずにhard coded labelを利用したい場合はconfidence thresholdを0にして運用
            if confidence_threshold is not None:
                X, labels =\
                    self._get_hard_labels(X, labels, confidence, confidence_threshold)

            self._target_samples_log.append((X, labels))
            self._confidence_log.append(confidence)

            # train classifier
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # labels = self._label_converter.encode(labels)
                # ここですでにsoft labelであることを利用している
                self._classifier.fit(X, labels)
            self._classifier_list.append(copy.deepcopy(self._classifier))

            # calculate predict loss (distと同じサイズのやつ）
            predict_losses = self._calc_predict_losses()

            # cost計算
            cost = dist + alpha[iteration] * predict_losses

    def _get_hard_labels(self, X, soft_labels, confidence, confidence_threshold=0):
        confidence = self.calc_confidence()
        decoded_labels = self.predict_barycentric()
        Xtrain = X[confidence > confidence_threshold]
        ytrain = decoded_labels[confidence > confidence_threshold]

        label_xor = numpy.setxor1d(ytrain, self._source_label)
        if len(label_xor) > 0:
            # labelが足りないなら仕方ないのでランダムにそのラベルを選ぶとか
            additional_data = [Xtrain, ]
            additional_label = [ytrain, ]
            for missed_label in label_xor:
                col_idx = self._label_converter.get_index(missed_label)[0]
                row_idx = soft_labels[:, col_idx] > 0.5
                idx = numpy.argsort(soft_labels[row_idx, col_idx])[::-1]
                missed_data = X[row_idx][idx]
                additional_data.append(missed_data)
                additional_label.append(numpy.full(len(missed_data), missed_label))
                # missed_data = X[decoded_labels == missed_label]
                # label_num = numpy.sum(self._source_label == missed_label)
                # idx = numpy.random.choice(len(missed_data), size=label_num)
                # additional_data.append(missed_data[idx])
                # additional_label.append(numpy.full(len(idx), missed_label))
            Xtrain = numpy.row_stack(additional_data)
            ytrain = numpy.concatenate(additional_label)

        return Xtrain, ytrain

    def _calc_predict_losses(self):
        predict_losses = []
        # join common variables and extra variables
        X = self._target_variables
        # すべての説明変数にある1つの目的変数を割り当てて誤差を計算
        label_est = self._classifier.decision_function(X)
        if label_est.ndim == 1:
            label_est_hard = label_est
        else:
            label_est_hard = self._label_converter.hard_decode(label_est)

        for label in self._source_label:
            if self._loss_type == 'hinge':
                predict_losses.append(numpy.maximum(0, 1 - label * label_est_hard))
            elif self._loss_type == 'zero_one':
                # たぶん下のコードにすればOK
                predict_losses.append((label != label_est_hard).astype(int))
            elif self._loss_type == 'zero_one_soft':
                # label_estがラベル確率を出力していると仮定
                # 不正解となる確率の合計値を出力する(1-正解確率）
                soft_label = self._label_converter.encode(numpy.array([label, ]))
                predict_losses.append(1 - (soft_label * label_est).sum(1))
            elif self._loss_type == 'cross_entropy':
                soft_label = self._label_converter.encode(numpy.array([label, ]))
                predict_losses.append(-(soft_label * numpy.log(label_est)).sum(1))

        return numpy.row_stack(predict_losses)

    # def _calc_predict_losses(self):
    #     predict_losses = []
    #     # すべての説明変数にある1つの目的変数を割り当てて誤差を計算
    #     label_est =\
    #         self._classifier.decision_function(self._target_variables)

    #     for label in self._source_label:
    #         if self._loss_type == 'hinge':
    #             predict_losses.append(numpy.maximum(0, 1 - label * label_est))
    #         elif self._loss_type == 'zero_one':
    #             # たぶん下のコードにすればOK
    #             predict_losses.append((label != label_est).astype(int))

    #     return numpy.row_stack(predict_losses)

    def barycentric_mapping(self, idx=-1):
        one_hot_labels = self._label_converter.encode(self._source_label)
        # source側ラベルの重心を計算し，これをtestの各点のラベルとして利用
        label_barycenter = \
            (self._transport_map_list[idx] /
             self._transport_map_list[idx].sum(0)).T.\
            dot(one_hot_labels)

        # source ,targetで説明変数の次元が同じとき
        return self._target_variables, label_barycenter

    def barycentric_mapping2(self, idx=-1):
        one_hot_labels = self._label_converter.encode(self._source_label)
        # Xの輸送，輸送先の情報の重心を基本的には使い，ラベルはソースドメインのものをそのまま利用
        x_barycenter =\
            (self._transport_map_list[idx] /
             self._transport_map_list[idx].sum(1)[:, None]).\
            dot(self._target_variables)

        return x_barycenter, one_hot_labels

    def calc_confidence(self, idx=-1):
        _, labels = self.barycentric_mapping(idx)
        confidence = numpy.ones(len(labels))
        idx1 = numpy.all(labels < 1e-10, axis=1)
        idx2 = numpy.any(labels > 1-1e-10, axis=1)
        # if all probability is 0
        confidence[idx1] = 0
        # if a class have probability 1
        confidence[idx2] = 1
        # else
        idx = numpy.logical_not(numpy.logical_or(idx1, idx2))
        confidence[idx] -= -(labels[idx] * numpy.log2(labels[idx])).sum(1)\
            / numpy.log2(labels.shape[1])

        return confidence

    def predict_barycentric(self, idx=-1):
        X, labels = self.barycentric_mapping(idx=idx)
        decoded_labels = self._label_converter.hard_decode(labels).astype(int)

        return decoded_labels

    def predict(self, X=None, idx=-1):
        if X is None:
            X = self._target_variables
        est = self._classifier_list[idx].predict(X)
        if est.ndim == 1:
            return est
        return self._label_converter.hard_decode(est).astype(int)


class JDOT_TargetNewObservations(JointDistributionOptimalTransport):
    def __init__(self, classifier, loss_type='zero_one', metric='sqeuclidean'):
        super().__init__(classifier, loss_type=loss_type, metric=metric)

    def fit(self, source_variables, target_variables,
            source_label, alpha=1, num_iterations=100,
            confidence_threshold=None, sinkhorn_reg=0.0):
        # 特徴量は sample x n_featureの形で持つ
        if source_variables.ndim == 1:
            source_variables = source_variables[:, None]
        if target_variables.ndim == 1:
            target_variables = target_variables[:, None]

        # the number of common variables is the number of source variables
        d = source_variables.shape[1]
        target_x = target_variables[:, :d]
        target_xe = target_variables[:, d:]

        self._target_xe = target_xe
        super().fit(source_variables, target_x, source_label,
                    alpha=alpha, num_iterations=num_iterations,
                    confidence_threshold=confidence_threshold,
                    sinkhorn_reg=sinkhorn_reg)

    def _calc_predict_losses(self):
        predict_losses = []
        # join common variables and extra variables
        X = numpy.column_stack((self._target_variables, self._target_xe))
        # すべての説明変数にある1つの目的変数を割り当てて誤差を計算
        label_est = self._classifier.decision_function(X)
        if label_est.ndim == 1:
            label_est_hard = label_est
        else:
            label_est_hard = self._label_converter.hard_decode(label_est)

        for label in self._source_label:
            if self._loss_type == 'hinge':
                if label == 0:
                    label = -1
                predict_losses.append(numpy.maximum(0, 1 - label * label_est_hard))
                # たぶん下のコードにすればOK
            if self._loss_type == 'zero_one':
                label_est_hard[label_est_hard > 0.5] = 1
                label_est_hard[label_est_hard <= 0.5] = 0
                predict_losses.append((label != label_est_hard).astype(int))
            elif self._loss_type == 'zero_one_soft':
                # label_estがラベル確率を出力していると仮定
                # 不正解となる確率の合計値を出力する(1-正解確率）
                soft_label = self._label_converter.encode(numpy.array([label, ]))
                predict_losses.append(1 - (soft_label * label_est).sum(1))
            elif self._loss_type == 'cross_entropy':
                soft_label = self._label_converter.encode(numpy.array([label, ]))
                label_est[label_est == 0] += 1e-9
                label_est /= label_est.sum(1).reshape(-1, 1)
                predict_losses.append(-(soft_label * numpy.log(label_est)).sum(1))

        return numpy.row_stack(predict_losses)

    def barycentric_mapping(self, idx=-1):
        one_hot_labels = self._label_converter.encode(self._source_label)
        # source側ラベルの重心
        label_barycenter = \
            (self._transport_map_list[idx] /
             self._transport_map_list[idx].sum(0)).T.\
            dot(one_hot_labels)

        X = numpy.column_stack((self._target_variables, self._target_xe))

        # source ,targetで説明変数の次元が同じとき
        return X, label_barycenter

    def barycentric_mapping2(self, idx=-1):
        one_hot_labels = self._label_converter.encode(self._source_label)
        # Xの輸送，輸送先の情報の重心を基本的には使う
        X = numpy.column_stack((self._target_variables, self._target_xe))
        x_barycenter =\
            (self._transport_map_list[idx] /
             self._transport_map_list[idx].sum(1)[:, None]).\
            dot(X)

        return x_barycenter, one_hot_labels

    def predict_barycentric(self, idx=-1):
        X, labels = self.barycentric_mapping(idx=idx)
        decoded_labels = self._label_converter.hard_decode(labels).astype(int)

        return decoded_labels

    def predict(self, X=None, idx=-1):
        if X is None:
            X = numpy.column_stack((self._target_variables, self._target_xe))
        est = self._classifier_list[idx].predict(X)
        if est.ndim == 1:
            return est
        return self._label_converter.hard_decode(est).astype(int)


class JDOT_SelfTaught(JointDistributionOptimalTransport):
    '''
    ターゲット側にラベルがあり，ソース側に新規変数がある
    '''

    def __init__(self, classifier, loss_type='hinge'):
        super().__init__(classifier, loss_type)

    def fit(self, source_variables, target_variables,
            target_label, alpha=1, num_iterations=100,
            sinkhorn_reg=0.0):
        # 特徴量は sample x n_featureの形で持つ
        if source_variables.ndim == 1:
            source_variables = source_variables[:, None]
        if target_variables.ndim == 1:
            target_variables = target_variables[:, None]

        # the number of common variables is the number of target variables
        d = target_variables.shape[1]

        source_x = source_variables[:, :d]
        source_xe = source_variables[:, d:]

        self.source_xe = source_xe
        self._target_label = target_label
        super().fit(source_x, target_variables, None,
                    alpha=alpha, num_iterations=num_iterations,
                    sinkhorn_reg=sinkhorn_reg)

    def _calc_predict_losses(self):
        predict_losses = []

        for xe in self.source_xe:
            # すべての説明変数にある1つの目的変数を割り当てて誤差を計算
            target_covariate = numpy.column_stack((
                self._target_variables,
                numpy.tile(xe, (len(self._target_variables), 1))))
            label_est = self._classifier.decision_function(target_covariate)
            predict_losses.append(
                numpy.maximum(0, 1 - self._target_label * label_est))

        return numpy.row_stack(predict_losses)

    def barycentric_mapping(self, idx=-1):
        # source側ラベルの重心
        xe_barycenter = \
            (self._transport_map_list[idx] / self._transport_map_list[idx].sum(0)).T.\
            dot(self.source_xe)

        X = numpy.column_stack((self._target_variables, xe_barycenter))

        # source ,targetで説明変数の次元が同じとき
        return X, self._target_label

    def barycentric_mapping2(self, idx=-1):
        # Xの輸送，輸送先の情報の重心を基本的には使う
        x_barycenter =\
            (self._transport_map_list[idx] / self._transport_map_list[idx].sum(1)[:, None]).\
            dot(self._target_variables)
        label_barycenter =\
            (self._transport_map_list[idx] / self._transport_map_list[idx].sum(1)[:, None]).\
            dot(self._target_label)
        X = numpy.column_stack((x_barycenter, self.source_xe))

        return X, label_barycenter
