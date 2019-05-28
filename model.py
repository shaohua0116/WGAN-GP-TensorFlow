from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = config.batch_size
        self.h = config.h
        self.w = config.w
        self.c_dim = config.c
        self.n_z = config.n_z
        self.num_dis_conv = config.num_dis_conv
        self.num_g_res_block = config.num_g_res_block
        self.num_d_res_block = config.num_d_res_block
        self.g_norm_type = config.g_norm_type
        self.d_norm_type = config.d_norm_type
        self.deconv_type = config.deconv_type
        # gan
        self.gan_type = config.gan_type
        self.gamma = config.gamma

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.h, self.w, self.c_dim],
        )

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk):
        fd = {
            self.image: batch_chunk['image'],  # [bs, h, w, c]
        }

        return fd

    def build(self, is_train=True):

        # Generator {{{
        # =========
        # G takes ramdon noise and generates images [bs, h, w, c]
        G = Generator('Generator', self.h, self.w, self.c_dim,
                      self.g_norm_type, self.deconv_type,
                      self.num_g_res_block, is_train)
        z = tf.random_uniform([self.batch_size, self.n_z], minval=-1, maxval=1, dtype=tf.float32)
        self.fake_image = fake_image = G(z)
        # }}}

        # Discriminator {{{
        # =========
        # D takes images as input and produce real-or-fake maps [bs, n, n]
        D = Discriminator('Discriminator',  self.num_dis_conv,
                          self.d_norm_type, self.num_d_res_block, is_train)
        d_real = D(self.image)
        d_fake = D(fake_image)
        self.fake_images = fake_image
        self.real_images = self.image

        if self.gan_type == 'wgan-gp':
            epsilon = tf.random_uniform(
                shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolated_image = self.image + epsilon * (fake_image - self.image)
            d_interpolated = D(interpolated_image)
        # }}}

        # Build losses {{{
        # =========
        # compute loss and prob
        if self.gan_type == 'lsgan':
            d_real_loss = tf.reduce_mean((d_real - tf.ones_like(d_real))**2)
            d_fake_loss = tf.reduce_mean((d_fake - tf.zeros_like(d_fake))**2)
            g_loss = tf.reduce_mean((d_fake - tf.ones_like(d_fake))**2)
        elif self.gan_type == 'hinge':
            d_real_loss = tf.reduce_mean(tf.nn.relu(tf.ones_like(d_real) - d_real))
            d_fake_loss = tf.reduce_mean(tf.nn.relu(tf.ones_like(d_fake) + d_fake))
            g_loss = -tf.reduce_mean(d_fake)
        elif self.gan_type == 'wgan-gp':
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            g_loss = -tf.reduce_mean(d_fake)
        else:
            raise NotImplementedError

        d_real_prob = tf.reduce_mean(d_real)
        d_fake_prob = tf.reduce_mean(d_fake)

        # compute gradient penalty
        if self.gan_type == 'wgan-gp':
            grad_d_interpolated = tf.gradients(
                d_interpolated, [interpolated_image])[0]
            slopes = tf.sqrt(1e-8 + tf.reduce_sum(
                tf.square(grad_d_interpolated), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        if self.gan_type in ['lsgan', 'hinge']:
            d_loss = d_real_loss + d_fake_loss
        self.d_loss = d_loss
        self.g_loss = g_loss
        if self.gan_type == 'wgan-gp':
            self.d_loss += self.gamma * gradient_penalty
            tf.summary.scalar("loss/gradient_penalty", gradient_penalty)
        # }}}

        # TensorBoard summaries {{{
        # =========
        if self.gan_type == 'lsgan':
            tf.summary.scalar("loss/d_real_loss", d_real_loss)
            tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_real_prob", d_real_prob)
        tf.summary.scalar("loss/d_fake_prob", d_fake_prob)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

        def d_output_vis(d_output):
            d_vis = tf.tile(tf.image.resize_nearest_neighbor(
                tf.clip_by_value(d_output, -1, 1),
                [self.h, self.w]), [1, 1, 1, self.c_dim])
            return d_vis

        tb_d = tf.concat([d_output_vis(d_real), d_output_vis(d_fake)], axis=2)
        if self.gan_type == 'wgan-gp':
            # normlize to [-1, 1]
            tb_d -= tf.reduce_min(tb_d)
            tb_d /= tf.reduce_max(tb_d)
            tb_d = tb_d * 2 - 1
        tb_img = tf.concat([self.image, fake_image], axis=2)
        tb_image = tf.concat([tb_img, tb_d], axis=1)
        tf.summary.image("img", tb_image)

        # visualize generated images
        n = int(np.sqrt(self.batch_size))
        fake_image_vis = tf.reshape(tf.transpose(tf.reshape(
            fake_image[:n*n], [n, n*self.h, self.w, self.c_dim]),
            [1, 0, 2, 3]), [1, n*self.h, n*self.w, self.c_dim])
        fake_image_vis = tf.image.resize_nearest_neighbor(
            tf.clip_by_value(fake_image_vis, -1, 1),
            [self.h*4, self.w*4])

        tf.summary.image("fake_image", fake_image_vis)
        # }}}

        # Output {{{
        # =========
        self.output = {
            'fake_image': fake_image
        }
        # }}}

        print('\033[93mSuccessfully loaded the model.\033[0m')
