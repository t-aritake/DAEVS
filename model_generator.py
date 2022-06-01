# -*- coding: utf-8 -*-
import numpy


class ModelGenerator(object):
    def __init__(self, jdot_class, classifier_class,
                 jdot_losstype='hinge', jdot_metric='sqeuclidean',
                 *args, **kwargs):
        self._classifier_class = classifier_class
        self._jdot_losstype = jdot_losstype
        self._jdot_metric = jdot_metric
        self._jdot_class = jdot_class
        self._args = args
        self._kwargs = kwargs

    def gen_model(self):
        clf = self._classifier_class(*self._args, **self._kwargs)
        jdot = self._jdot_class(
            clf, loss_type=self._jdot_losstype, metric=self._jdot_metric)

        return jdot


if __name__ == '__main__':
    import JDOT
    import sklearn.svm
    mg = ModelGenerator(
        JDOT.JDOT_TargetNewObservations,
        sklearn.svm.SVC,
        jdot_losstype='hinge',
        C=1, kernel='linear')
