from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from six.moves import xrange
from pprint import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim

from input_ops import create_input_ops
from util import log
from config import argparser


class Trainer(object):

    def __init__(self, config, model, dataset, dataset_test):
        self.config = config
        self.model = model
        learning_hyperparameter_str = '{}_{}_bs_{}_lr_g_{}_lr_d_{}_update_G{}D{}'.format(
            config.dataset, config.gan_type, config.batch_size,
            config.learning_rate_g, config.learning_rate_d,
            config.update_g, config.update_d)
        model_hyperparameter_str = 'G_deconv_{}_dis_conv_{}_{}_{}_norm'.format(
            config.deconv_type, config.num_dis_conv, 
            config.g_norm_type, config.d_norm_type)

        self.train_dir = './train_dir/%s-%s-%s' % (
            config.prefix,
            learning_hyperparameter_str + '_' + model_hyperparameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )

        os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        _, self.batch_train = create_input_ops(
            dataset, self.batch_size, is_training=True)
        _, self.batch_test = create_input_ops(
            dataset_test, self.batch_size, is_training=False)

        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)

        # --- checkpoint and monitoring ---
        all_var = tf.trainable_variables()

        d_var = [v for v in all_var if v.name.startswith('Discriminator')]
        log.warn("********* d_var ********** ")
        slim.model_analyzer.analyze_vars(d_var, print_info=True)

        g_var = [v for v in all_var if v.name.startswith(('Generator'))]
        log.warn("********* g_var ********** ")
        slim.model_analyzer.analyze_vars(g_var, print_info=True)

        rem_var = (set(all_var) - set(d_var) - set(g_var))
        print([v.name for v in rem_var])
        assert not rem_var

        self.g_optimizer = tf.train.AdamOptimizer(
            self.config.learning_rate_g, 
            beta1=self.config.adam_beta1, beta2=self.config.adam_beta2
        ).minimize(self.model.g_loss, var_list=g_var,
                   name='g_optimize_loss', global_step=self.global_step)

        self.d_optimizer = tf.train.AdamOptimizer(
            self.config.learning_rate_d, 
            beta1=self.config.adam_beta1, beta2=self.config.adam_beta2
        ).minimize(self.model.d_loss, var_list=d_var,
                   name='d_optimize_loss')

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=1000)
        pretrain_saver = tf.train.Saver(var_list=all_var, max_to_keep=1)
        pretrain_saver_g = tf.train.Saver(var_list=g_var, max_to_keep=1)
        pretrain_saver_d = tf.train.Saver(var_list=d_var, max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=600,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        def load_checkpoint(ckpt_path, saver, name=None):
            if ckpt_path is not None:
                log.info("Checkpoint path for {}: {}".format(name, ckpt_path))
                saver.restore(self.session, ckpt_path)
                log.info("Loaded the pretrain parameters " +
                         "from the provided checkpoint path.")

        load_checkpoint(
            config.checkpoint_g, pretrain_saver_g,  name='Generator')
        load_checkpoint(
            config.checkpoint_d, pretrain_saver_d,  name='Discriminator')
        load_checkpoint(
            config.checkpoint, pretrain_saver, name='All vars')

    def train(self):
        log.infov("Training Starts!")
        pprint(self.batch_train)
        step = self.session.run(self.global_step)

        for s in xrange(self.config.max_training_steps):

            if s % self.config.ckpt_save_step == 0:
                log.infov("Saved checkpoint at %d", step)
                self.saver.save(self.session, os.path.join(
                    self.train_dir, 'model'), global_step=step)

            step, summary, d_loss, g_loss, step_time = \
                self.run_single_step(self.batch_train, step=s, is_train=True)

            if s % self.config.log_step == 0:
                self.log_step_message(step, d_loss, g_loss, step_time)

            if s % self.config.write_summary_step == 0:
                self.summary_writer.add_summary(summary, global_step=step)

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.summary_op,
                 self.model.d_loss, self.model.g_loss,
                 self.g_optimizer, self.d_optimizer]

        fetch_values = self.session.run(
            fetch,
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )
        [step, summary, d_loss, g_loss] = fetch_values[:4]

        for t in range(self.config.update_g - 1):
            self.session.run(self.g_optimizer,
                             feed_dict=self.model.get_feed_dict(batch_chunk))

        for t in range(self.config.update_d - 1):
            batch_chunk = self.session.run(batch)
            self.session.run(self.d_optimizer,
                             feed_dict=self.model.get_feed_dict(batch_chunk))

        _end_time = time.time()

        return step, summary, d_loss, g_loss,  (_end_time - _start_time)

    def log_step_message(self, step, d_loss, g_loss, step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((
            " [{split_mode:5s} step {step:4d}] " +
            "D loss: {d_loss:.5f} G loss: {g_loss:.5f} " +
            "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
            ).format(split_mode=(is_train and 'train' or 'val'),
                     step=step, d_loss=d_loss, g_loss=g_loss,
                     sec_per_batch=step_time,
                     instance_per_sec=self.batch_size / step_time))


def main():

    config, model, dataset_train, dataset_test = argparser(is_train=True)

    trainer = Trainer(config, model, dataset_train, dataset_test)

    log.warning("dataset: %s, learning_rate_g: %f, learning_rate_d: %f",
                config.dataset, config.learning_rate_g, config.learning_rate_d)
    trainer.train()

if __name__ == '__main__':
    main()
