import numpy as np
import tensorflow as tf
from util import log
from ops import fc
from ops import conv2d_res_new as conv2d_res


class Generator(object):
    def __init__(self, name, h, w, c, norm_type, deconv_type, 
                 num_res_block, is_train):
        self.name = name
        self._h = h
        self._w = w
        self._c = c
        self._norm_type = norm_type
        self._deconv_type = deconv_type
        self._num_res_block = num_res_block
        self._is_train = is_train
        self._reuse = False
        self.start_dim_x = 4 if w > 32 else 1
        self.start_dim_y = 4 if h > 32 else 1
        self.start_dim_ch = 256

    def __call__(self, input):
        if self._deconv_type == 'bilinear':
            from ops import bilinear_deconv2d as deconv2d
        elif self._deconv_type == 'nn':
            from ops import nn_deconv2d as deconv2d
        elif self._deconv_type == 'transpose':
            from ops import deconv2d
        else:
            raise NotImplementedError
        with tf.variable_scope(self.name, reuse=self._reuse):
            if not self._reuse:
                print('\033[93m'+self.name+'\033[0m')
            _ = fc(input, self.start_dim_x * self.start_dim_y  * self.start_dim_ch, 
                   self._is_train, info=not self._reuse, norm='none', name='fc')
            _ =  tf.reshape(_, [_.shape.as_list()[0], self.start_dim_y, 
                                self.start_dim_x, self.start_dim_ch])
            if not self._reuse:
                log.info('reshape {} '.format(_.shape.as_list()))
            num_deconv_layer = int(np.ceil(np.log2(
                max(float(self._h/self.start_dim_y), float(self._w/self.start_dim_x)))))
            for i in range(num_deconv_layer):
                _ = deconv2d(_, max(self._c, int(_.get_shape().as_list()[-1]/2)), 
                             self._is_train, info=not self._reuse, norm=self._norm_type,
                             name='deconv{}'.format(i+1))
                if num_deconv_layer - i <= self._num_res_block:
                    _ = conv2d_res(
                            _, self._is_train, info=not self._reuse, 
                            name='res_block{}'.format(self._num_res_block - num_deconv_layer + i + 1))
            _ = deconv2d(_, self._c, self._is_train, k=1, s=1, info=not self._reuse,
                         activation_fn=tf.tanh, norm='none',
                         name='deconv{}'.format(i+2))
            _ = tf.image.resize_bilinear(_, [self._h, self._w])

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return _
