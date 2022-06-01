# -*- coding: utf-8 -*-
import numpy


class Runner(object):

    def __init__(self, dataset, model_generator):
        self._dataset = dataset
        self._model_generator = model_generator

    def run(self, num_iterations=10, num_fit_iterations=20,
            alpha=1.0, confidence_threshold=None, sinkhorn_reg=0.0, fill=None):
        accuracy_list = []
        for iteration in range(num_iterations):
            # print('iteration', iteration)
            accuracy_list.append(
                self._run_iteration(alpha, num_fit_iterations,
                                    fill, confidence_threshold, sinkhorn_reg))

        return accuracy_list

    def _run_iteration(self, alpha, fit_iterations,
                       fill=None, confidence_threshold=None, sinkhorn_reg=0.0):
        # データ生成
        data = self._dataset.gen_data()
        Xs0 = data[0]
        ys = data[1]
        Xt = data[2]
        common_idx = data[4]
        if fill is None:
            Xs = Xs0[:, common_idx]
        else:
            Xs = numpy.full_like(Xs0, fill)
            Xs[:, common_idx] = Xs0[:, common_idx]

        # 初期モデルも生成
        model = self._model_generator.gen_model()
        model.fit(Xs, Xt, ys, alpha=alpha, num_iterations=fit_iterations,
                  confidence_threshold=confidence_threshold, sinkhorn_reg=sinkhorn_reg)

        # label_est_barycentric = model.predict_barycentric()
        # confusion_matrix =\
        #     sklearn.metrics.confusion_matrix(yt, numpy.sign(label_est_barycentric))
        # print(confusion_matrix)
        # print("{0:.6f},{1:.6f}".format(acc1, acc2), file=writer)

        return model, data
