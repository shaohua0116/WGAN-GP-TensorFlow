from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import fnmatch
import glob

import numpy as np
from imageio import imread
from skimage.transform import resize


class Dataset(object):

    def __init__(self, ids, name='default',
                 h=256, w=256, data_augmentation=False,
                 centor_crop=True,
                 max_examples=None, is_train=True):
        self._ids = list(ids)
        self.name = name
        self.is_train = is_train
        self.h = h
        self.w = w
        self.data_augmentation = data_augmentation
        self.center_crop = centor_crop

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

    def get_data(self, id):
        # preprocessing and data augmentation
        img = imread(id)/255.*2-1
        if len(img.shape) == 2:
            # grayscale images
            img = np.tile(np.expand_dims(img, axis=-1), [1, 1, 3])
        if img.shape[-1] == 4:
            # images with alpha channel
            img = img[:, :, :3]
        if not self.data_augmentation:
            min_ratio = min(img.shape[0]/float(self.h),
                            img.shape[1]/float(self.w))
        else:
            min_ratio = min(img.shape[0]/float(self.h*1.1),
                            img.shape[1]/float(self.w*1.1))
        img = resize(img, (np.array(img.shape[:2])/min_ratio).astype(np.int))
        # crop
        if img.shape[0]-self.h > 0:
            y_offset = np.random.randint(img.shape[0]-self.h)
        elif self.center_crop:
            y_offset = int((img.shape[0]-self.h)/2)
        else:
            y_offset = 0
        if img.shape[1]-self.w > 0:
            x_offset = np.random.randint(img.shape[1]-self.w)
        elif self.center_crop:
            x_offset = int((img.shape[1]-self.w)/2)
        else:
            x_offset = 0
        img = img[y_offset:y_offset+self.h, x_offset:x_offset+self.w]
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


def create_default_splits(path, is_train=True, h=256, w=256):
    ids = all_ids(path)

    dataset_train = Dataset(ids, name='train', h=h, w=w, is_train=False)
    dataset_test = Dataset(ids, name='test', h=h, w=w, is_train=False)
    return dataset_train, dataset_test


def all_ids(path):
    _ids = []

    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(('.jpg', '.webp', '.JPEG', '.png', 'jpeg')):
                _ids.append(os.path.join(root, filename))

    rs = np.random.RandomState(123)
    rs.shuffle(_ids)
    return _ids
