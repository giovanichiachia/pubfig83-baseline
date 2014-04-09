# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# Slightly based on Nicolas Pinto's
# http://github.com/npinto/sclas/blob/master/svm_ova_fromfilenames.py
#
# License: BSD


import numpy as np
from sklearn.svm import SVC

DEFAULT_REGULARIZATION = 1e5
DEFAULT_TRACE_NORMALIZATION = True


def svm_ova(train_set,
            test_set,
            # --
            train_labels,
            test_labels,
            # --
            C=DEFAULT_REGULARIZATION,
            trace_normalization=DEFAULT_TRACE_NORMALIZATION,
            ):

    categories = np.unique(train_labels)
    n_categories = len(categories)
    n_train = train_set.shape[0]
    n_test = test_set.shape[0]

    ltrain = np.zeros(n_train)
    test_predictions = np.empty((n_test, n_categories))

    assert(test_set.shape[1] == train_set.shape[1])

    print 'kernelizing features...'
    kernel_train = np.dot(train_set, train_set.T)
    kernel_test = np.dot(test_set, train_set.T)

    if trace_normalization:
        kernel_trace = kernel_train.trace()
        kernel_train = kernel_train / kernel_trace
        kernel_test = kernel_test / kernel_trace

    cat_index = {}

    # -- iterates over different categories.
    for icat, cat in enumerate(categories):
        ltrain[train_labels != cat] = -1
        ltrain[train_labels == cat] = +1

        svm = SVC(kernel='precomputed', C=C, tol=1e-5)
        svm.fit(kernel_train, ltrain)
        resps = svm.decision_function(kernel_test)[:, 0]
        test_predictions[:, icat] = resps

        cat_index[cat] = icat

        if n_categories == 2:
            test_predictions[:, 1] = -resps
            break

    gt = np.array([cat_index[e] for e in test_labels]).astype('int')

    pred = test_predictions.argmax(1)
    accuracy = 100. * (pred == gt).sum() / n_test

    return accuracy
