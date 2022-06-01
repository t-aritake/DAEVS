# -*- coding: utf-8 -*-
import itertools
import dataset
import runner
import model_generator
import JDOT
import sklearn.svm

if __name__ == '__main__':
    common_sensors_list = []
    common_accuracy = []
    extra_accuracy = []

    mg = model_generator.ModelGenerator(
        JDOT.JDOT_TargetNewObservations,
        sklearn.svm.SVC,
        kernel='rbf',
        C=10, jdot_losstype='zero_one')

    for common_sensors in itertools.combinations(range(9), 5):
        print(common_sensors)
        tmp_common_accuracy = []
        tmp_extra_accuracy = []
        common_sensors = list(common_sensors)
        common_sensors_list.append(common_sensors)

        for subject_id in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]:
            ds1 = dataset.GenActivityData(
                subject_id, common_sensors, target_labels=[11, 13],
                only_common=True)
            ds2 = dataset.GenActivityData(
                subject_id, common_sensors, target_labels=[11, 13],
                only_common=False)

            runner1 = runner.Runner(ds1, mg)
            runner2 = runner.Runner(ds2, mg)

            tmp_common_res = runner1.run(
                1, alpha=1e-1, sinkhorn_reg=0, confidence_threshold=0.5)
            tmp_extra_res = runner2.run(
                1, alpha=1e-1, sinkhorn_reg=0, confidence_threshold=0.5)
            yt = tmp_common_res[-1][1][3]
            acc1 = (yt == tmp_common_res[-1][0].predict_barycentric()).mean()
            acc2 = (yt == tmp_extra_res[-1][0].predict_barycentric()).mean()
            tmp_common_accuracy.append(acc1)
            tmp_extra_accuracy.append(acc2)

        common_accuracy.append(tmp_common_accuracy)
        extra_accuracy.append(tmp_extra_accuracy)

    import pickle
    writer = open('./results/activity/results_activity_confident2.pkl', 'wb')
    results = {
        'common_sensors_list': common_sensors_list,
        'common_accuracy': common_accuracy,
        'extra_accuracy': extra_accuracy}
    pickle.dump(results, writer)
    writer.close()
