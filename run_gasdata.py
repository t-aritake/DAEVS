# -*- coding: utf-8 -*-
import itertools
import dataset
import runner
import numpy
import model_generator
import JDOT
import sklearn.svm
import pickle


def run_estimation(target_labels, id_list, common_sensors):
    tmp_common_accuracy = []
    tmp_extra_accuracy = []
    for (s, t) in itertools.permutations(id_list, 2):
        print(s, t)
        ds1 = dataset.GasDataset(
            target_labels, s, t, common_sensors, only_common=True)
        ds2 = dataset.GasDataset(
            target_labels, s, t, common_sensors, only_common=False)

        runner1 = runner.Runner(ds1, mg)
        runner2 = runner.Runner(ds2, mg)

        res1 = runner1.run(1, alpha=1e-1, sinkhorn_reg=0, confidence_threshold=0.0)
        res2 = runner2.run(1, alpha=1e-1, sinkhorn_reg=0, confidence_threshold=0.0)

        yt = res1[0][1][3]
        acc1 = numpy.mean(yt == res1[0][0].predict())
        acc2 = numpy.mean(yt == res2[0][0].predict())
        tmp_common_accuracy.append(acc1)
        tmp_extra_accuracy.append(acc2)

    return tmp_common_accuracy, tmp_extra_accuracy

if __name__ == '__main__':
    # 対象のバッチのID
    id_list = [1, 2, 3, 4]
    target_labels = [1, 2]

    common_sensors_list = []
    common_accuracy = []
    extra_accuracy = []
    full_accuracy = []

    mg = model_generator.ModelGenerator(
        JDOT.JDOT_TargetNewObservations,
        sklearn.svm.SVC,
        kernel='rbf',
        C=10)

    for (s, t) in itertools.permutations(id_list, 2):
        print(s, t)
        ds = dataset.GasDataset(
            target_labels, s, t, list(range(16)), only_common=False)
        runner1 = runner.Runner(ds, mg)
        res = runner1.run(1, alpha=1e-1, sinkhorn_reg=0, confidence_threshold=0.0)

        yt = res[0][1][3]
        acc = numpy.mean(yt == res[0][0].predict())
        full_accuracy.append(acc)

    for common_sensors in itertools.combinations(range(16), 8):
        print(common_sensors)
        common_sensors = list(common_sensors)
        common_sensors_list.append(common_sensors)

        tmp_common_accuracy, tmp_extra_accuracy =\
            run_estimation(target_labels, id_list, common_sensors)

        common_accuracy.append(tmp_common_accuracy)
        extra_accuracy.append(tmp_extra_accuracy)

    # writer = open('./results/gas/results_gas.pkl', 'wb')
    # results = {
    #     'common_sensors_list': common_sensors_list,
    #     'common_accuracy': common_accuracy,
    #     'extra_accuracy': extra_accuracy,
    #     'full_accuracy': full_accuracy}
    # pickle.dump(results, writer)
    # writer.close()
