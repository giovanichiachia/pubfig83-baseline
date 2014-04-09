"""Minimal script to execute the 90/10 baseline result on PubFig83 following
the original evaluation protocol
"""

# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

import optparse
import time

import numpy as np
from scipy import misc

import cnnrandom.models as cnn_models
from cnnrandom import BatchExtractor

from data import cvprw11_setup
from svm import svm_ova

DEFAULT_IN_SHAPE = (200, 200)
DEFAULT_MODEL = 'fg11_ht_l3_1_description'


# -- SVM regularization constant
C = 1e5


def load_imgs(fnames, out_shape):

    n_imgs = len(fnames)
    img_set = np.empty((n_imgs,) + out_shape, dtype='float32')

    for i, fname in enumerate(fnames):

        arr = misc.imread(fname, flatten=True)
        arr = misc.imresize(arr, out_shape).astype('float32')

        arr -= arr.min()
        arr /= arr.max()

        img_set[i] = arr

    return img_set


def pf83_protocol(dataset_path):

    t0 = time.time()

    # -- data object implementing the PubFig83 protocol
    data_obj = cvprw11_setup(dataset_path)
    n_splits = data_obj.n_splits

    # -- first, extract features from all samples in the dataset
    fnames = data_obj.classification_task()[0]

    # -- initialize extractor
    baseline_model = cnn_models.fg11_ht_l3_1_description
    extractor = BatchExtractor(in_shape=DEFAULT_IN_SHAPE, model=baseline_model)

    print 'loading all images...'
    imgs = load_imgs(fnames, DEFAULT_IN_SHAPE)

    if len(imgs) > 0:
        print 'extracting features from all images...'
        dataset = extractor.extract(imgs)

        # -- make sure features were properly extracted
        assert(not np.isnan(np.ravel(dataset)).any())
        assert(not np.isinf(np.ravel(dataset)).any())

        # -- reshape dataset
        dataset.shape = dataset.shape[0], -1

        print "processing the dataset splits..."
        accs = np.empty(n_splits, dtype=np.float32)

        for split in xrange(n_splits):

            print 'split:', split

            # -- get split information
            _, train_labels, train_idxs = \
                data_obj.classification_task(split=split, split_role='train')

            _, test_labels, test_idxs = \
                data_obj.classification_task(split=split, split_role='test')

            train_set = dataset[train_idxs]
            test_set = dataset[test_idxs]

            print 'train_set.shape', train_set.shape
            print 'test_set.shape', test_set.shape

            print 'normalizing features...'
            fmean = train_set.mean(axis=0)
            fstd = train_set.std(axis=0)
            fstd[fstd == 0.] = 1.

            train_set -= fmean
            train_set /= fstd
            test_set -= fmean
            test_set /= fstd

            accs[split] = svm_ova(train_set, test_set,
                                  train_labels, test_labels)

            print 'accuracy for split %d: %g' % (split, accs[split])

        print 'mean accuracy (+/- std. error): %g (+/- %g)' % (
            accs.mean(), accs.std(ddof=1) / np.sqrt(n_splits))

        print 'protocol executed in %g seconds...' % (time.time() - t0)

    else:
        print 'no images in the given path'

    print 'done!'


def get_optparser():

    usage = "usage: %prog <DATASET_PATH>"
    parser = optparse.OptionParser(usage=usage)

    return parser


def main():
    parser = get_optparser()
    opts, args = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
    else:
        dataset_path = args[0]

        pf83_protocol(dataset_path)

if __name__ == "__main__":
    main()
