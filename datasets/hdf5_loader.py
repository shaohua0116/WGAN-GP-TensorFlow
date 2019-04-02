from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import h5py
from util import log


class Dataset(object):

    def __init__(self, path, ids, name='default',
                 max_examples=None, is_train=True):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

        filename = 'data.hdf5'

        file = os.path.join(path, filename)
        log.info("Reading %s ...", file)

        self.data = h5py.File(file, 'r')
        log.info("Reading Done: %s", file)

    def get_data(self, id):
        # preprocessing and data augmentation
        img = self.data[id]['image'].value/255. * 2 - 1
        return img

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def create_default_splits(path, h=32, w=32, is_train=True):
    ids = all_ids(path)
    dataset_train = Dataset(path, ids, name='train', is_train=False)
    dataset_test = Dataset(path, ids, name='test', is_train=False)
    return dataset_train, dataset_test


def all_ids(path):
    id_filename = 'id.txt'
    id_txt = os.path.join(path, id_filename)
    with open(id_txt, 'r') as fp:
        _ids = [s.strip() for s in fp.readlines() if s]
    rs = np.random.RandomState(123)
    rs.shuffle(_ids)
    return _ids
