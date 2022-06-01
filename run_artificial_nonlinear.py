# -*- coding: utf-8 -*-
import dataset
import numpy
import runner
import classifier
import model_generator
import JDOT
import sklearn.svm
import sklearn.datasets
# numpy.random.seed(0)


def twocircles_dataset(common_idx=[0, ]):
    ds = dataset.TwoBallsDataset(
        Ns=1000, Nt=300, class_ratio=[0.5, 0.5],
        radius_list=[2, 3, 2, 3], radius_std_list=[0.1, 0.1, 0.1, 0.1],
        source_scales_list=[[1, 1], [1.1, 1]],
        source_mu_list=[[0, 0], [0, 0]],
        target_mu_list=[[1, 0], [1, 0]],
        target_scales_list=[[0.8, 1], [1.1, 1]],
        common_idx=common_idx)

    return ds


def twospirals_dataset(common_idx=[0, ]):
    ds = dataset.TwoSpiralsDataset(
        Ns=1000, Nt=300, class_ratio=[0.5, 0.5],
        radius=2.4*numpy.pi, start=numpy.pi/2, noise=0.5,
        Xs_center=[0, 0], Xt_center=[3, -1],
        Xs_scale=[1, 1], Xt_scale=[1, 1],
        common_idx=common_idx)

    return ds


def twoballs_dataset():
    ds = dataset.TwoBallsDataset(
        Ns=1000, Nt=300, class_ratio=[0.5, 0.5],
        radius_list=[2, 3, 2, 3], radius_std_list=[0.1, 0.1, 0.1, 0.1],
        source_scales_list=[[1, 1, 1], [1.1, 1, 1]],
        source_mu_list = [[0, 0, 0], [0, 0, 0]],
        target_mu_list=[[3, -1, 0], [3, -1, 0]],
        target_scales_list=[[0.8, 1, 1.1], [1.1, 1, 1.3]],
        common_idx=[0, 1])

    return ds


def twomoons_dataset(common_idx=[0, ], rotation=0):
    ds = dataset.TwoMoonsDataset(
        Ns=1000, Nt=300, class_ratio=[0.5, 0.5],
        radius=0.5, noise=0.02, Xs_shift=[0, 0], Xt_shift=[3, 0],
        rotation=rotation,
        Xs_scale=[3, 3], Xt_scale=[3, 3], common_idx=common_idx)

    return ds


def dataset_from_boundary(boundary, Ns=100, Nt=100, class_ratio=[0.5, 0.5], common_idx=[0, ]):
    ds = dataset.DatasetFromBoundary(
        boundary, [-2, 2], [2, 2], [-1, 3], [2, 2], [.5, -.5],
        noise=0.2, target_transpose=1.5,
        Ns=Ns, Nt=Nt, class_ratio=class_ratio, common_idx=common_idx)

    return ds


if __name__ == '__main__':
    # numpy.random.seed(0)
    # ds = twocircles_dataset(common_idx=[0, ])
    # ds = twoballs_dataset()
    # ds = twospirals_dataset(common_idx=[0, ])
    # ds = twomoons_dataset([0, ], rotation=90)
    ds = dataset_from_boundary(
        lambda x: numpy.cos(x),
        Ns=2000, Nt=400, class_ratio=[0.5, 0.5],
        common_idx=[0, ])

    import sklearn.linear_model
    # SVM
    mg = model_generator.ModelGenerator(
        JDOT.JDOT_TargetNewObservations,
        sklearn.svm.SVC,
        kernel='rbf',
        jdot_losstype='hinge',
        C=1)
    # mg = model_generator.ModelGenerator(
    #     JDOT.JDOT_TargetNewObservations,
    #     classifier.NeuralNetwork,
    #     jdot_losstype='hinge',
    #     hidden_features_list=[512, 64],
    #     activations_list=['relu', 'relu', 'sigmoid'])

    runner = runner.Runner(ds, mg)
    res_list = runner.run(
        num_iterations=1, num_fit_iterations=20, alpha=.1,
        confidence_threshold=0, sinkhorn_reg=0)

    for res in res_list:
        yt = res[1][3]
        acc1 = numpy.mean(yt == res[0].predict_barycentric())
        acc2 = numpy.mean(yt == res[0].predict())

        print(acc1, acc2)

    Xs = res[1][0]
    ys = res[1][1]
    Xt = res[1][2]
    yt = res[1][3]
    yt_est = res[0].predict_barycentric()
