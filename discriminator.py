#!/usr/bin/env python
import tensorflow as tf
from ops import conv2d, conv2d_res


class Discriminator(object):
    def __init__(self, name, num_conv, norm_type, num_res_block, is_train):
        self.name = name
        self._num_conv = num_conv
        self._norm_type = norm_type
        self._num_res_block = num_res_block
        self._is_train = is_train
        self._reuse = False

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            if not self._reuse:
                print('\033[93m'+self.name+'\033[0m')
            _ = input
            num_channel = [32, 64, 128, 256, 256, 512, 512, 512, 512]
            assert self._num_conv <= 10 and self._num_conv > 0
            for i in range(self._num_conv):
                _ = conv2d(_, num_channel[i], self._is_train, info=not self._reuse,
                           norm=self._norm_type, name='conv{}'.format(i+1))
                if self._num_conv - i <= self._num_res_block:
                    _ = conv2d_res(
                            _, self._is_train, info=not self._reuse,
                            norm=self._norm_type,
                            name='res_block{}'.format(self._num_res_block - self._num_conv + i + 1))

            _ = conv2d(_, int(num_channel[i]/4), self._is_train, k=1, s=1,
                       info=not self._reuse, norm='none', name='conv{}'.format(i+2))
            _ = conv2d(_, 1, self._is_train, k=1, s=1, info=not self._reuse,
                       activation_fn=None, norm='none',
                       name='conv{}'.format(i+3))

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return _
