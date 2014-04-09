# Authors: Giovani Chiachia <giovani.chiachia@gmail.com>
#
# Slightly based on James Bergstra's
# http://github.com/jaberg/skdata/blob/master/skdata/pubfig83.py
#
# License: BSD

import os
from glob import glob

import numpy as np
from sklearn import cross_validation


# -- piece of code extracted from scikit-data
def _int_labels(all_labels, labels):
    """['me', 'b', 'b', ...] -> [0, 1, 1, ...]"""
    u = np.unique(all_labels)
    i = np.searchsorted(u, labels)
    return i


# -- return all folders that have files of type "type"
def _get_folders_recursively(path, type):

    names = []

    for root, subFolders, files in os.walk(path):
        for file in files:
            if file[-len(type):] == type:
                names += [os.path.relpath(root, path)]
                break

    return names


class PubFig83(object):

    def __init__(self, path='./', file_type='jpg',
                 seed=42, n_train=90, n_test=10, n_splits=10):

        self.path = path
        self.file_type = file_type
        self.seed = seed
        self.n_train = n_train
        self.n_test = n_test
        self.n_splits = n_splits
        self.meta = self._get_meta()

    def _get_meta(self):
        """
        retrieve dataset metadata
        """
        folders = np.array(sorted(_get_folders_recursively(
                           self.path, self.file_type)))

        meta = {}
        meta['names'] = []
        meta['files'] = []

        for name in folders:
            fnames = sorted(glob(os.path.join(self.path, name,
                                              '*' + self.file_type)))
            for fname in fnames:
                meta['names'] += [name]
                meta['files'] += [fname]

        meta['names'] = np.array(meta['names'])
        meta['files'] = np.array(meta['files'])

        return meta

    @property
    def classification_splits(self):
        """
        generates splits and attaches them in the "splits" attribute
        """
        if not hasattr(self, '_classification_splits'):
            seed = self.seed
            n_train = self.n_train
            n_test = self.n_test
            n_splits = self.n_splits
            self._classification_splits = \
                self._generate_classification_splits(seed, n_train,
                                                     n_test, n_splits)

        return self._classification_splits

    def _generate_classification_splits(self, seed, n_train, n_test, n_splits):
        """
        generates splits according to the PubFig83 protocol
        """
        n_train = self.n_train
        n_test = self.n_test
        rng = np.random.RandomState(seed)

        splits = {}
        for split_id in range(n_splits):
            splits[split_id] = {}
            splits[split_id]['train'] = []
            splits[split_id]['test'] = []

        labels = np.unique(self.meta['names'])
        for label in labels:
            samples_to_consider = (self.meta['names'] == label)
            samples_to_consider = np.where(samples_to_consider)[0]

            L = len(samples_to_consider)
            assert L >= n_train + n_test, 'category %s too small' % label

            ss = cross_validation.ShuffleSplit(L,
                                               n_iterations=n_splits,
                                               test_size=n_test,
                                               train_size=n_train,
                                               random_state=rng)

            for split_id, [train_index, test_index] in enumerate(ss):
                splits[split_id]['train'] += \
                    samples_to_consider[train_index].tolist()
                splits[split_id]['test'] += \
                    samples_to_consider[test_index].tolist()
        return splits

    def classification_task(self, split=None, split_role=None):
        """
        :param split: an integer from 0 to self.n_splits - 1.
        :param split_role: either 'train' or 'test'

        :returns: either all samples (when split_k=None) or the specific
                  train/test split
        """

        if split is not None:
            assert split in range(self.n_splits), ValueError(split)
            assert split_role in ('train', 'test'), ValueError(split_role)
            inds = self.classification_splits[split][split_role]
        else:
            inds = range(len(self.meta['files']))

        paths = self.meta['files'][inds]
        labels = _int_labels(self.meta['names'], self.meta['names'][inds])

        return paths, labels, inds


def cvprw11_setup(path):
    """
    90/10 protocol as described in

    Nicolas Pinto, Zak Stone, Todd Zickler, and David D. Cox, "Scaling-up
    Biologically-Inspired Computer Vision: A Case-Study on Facebook," in IEEE
    Computer Vision and Pattern Recognition, Workshop on Biologically
    Consistent Vision, 2011.
    """

    return PubFig83(path)
