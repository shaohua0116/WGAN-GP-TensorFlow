from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from collections import defaultdict

import numpy as np
from six.moves import xrange
import h5py
from imageio import imwrite
import tensorflow as tf

from input_ops import create_input_ops
from util import log
from config import argparser


class EvalManager(object):

    def __init__(self):
        self._ids = []
        self._output = []

    def add_batch(self, id, output):
        self._ids.append(id)
        self._output.append(output)

    def dump_result(self, filename):
        log.infov("Dumping results into %s ...", filename)
        f = h5py.File(filename, 'w')

        merge_output_list = defaultdict(list)
        for d in tuple(self._output):
            for key in d.keys():
                merge_output_list[key].append(d[key])

        output_list = {}
        for key in merge_output_list.keys():
            stacked_output = np.stack(merge_output_list[key])
            stacked_output = np.reshape(
                    stacked_output,
                    [np.prod(stacked_output.shape[:2])] +
                    list(stacked_output.shape[2:]))
            f[key] = stacked_output
        log.info("Dumping resultsn done.")


class Evaler(object):

    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.train_dir = config.train_dir
        self.output_file = config.output_file
        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        self.dataset = dataset

        _, self.batch = create_input_ops(dataset, self.batch_size,
                                         is_training=False,
                                         shuffle=False)

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        tf.set_random_seed(1234)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=100)

        self.checkpoint = config.checkpoint
        if self.checkpoint is None and self.train_dir:
            self.checkpoint = tf.train.latest_checkpoint(self.train_dir)
        if self.checkpoint is None:
            log.warn("No checkpoint is given. Just random initialization :-)")
            self.session.run(tf.global_variables_initializer())
        else:
            log.info("Checkpoint path : %s", self.checkpoint)

    def eval_run(self):
        # load checkpoint
        if self.checkpoint:
            self.saver.restore(self.session, self.checkpoint)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")
        log.info("# of examples = %d", len(self.dataset))
        log.info("max_steps = %d", self.config.max_evaluation_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(
            self.session, coord=coord, start=True)

        evaler = EvalManager()
        try:
            for s in xrange(self.config.max_evaluation_steps):
                step, step_time, id, d_loss, g_loss, fake_images, \
                    real_images, output = self.run_single_step(self.batch)
                self.log_step_message(s, d_loss, g_loss, step_time)
                evaler.add_batch(id, output)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e))

        if self.config.write_summary_image:
            n = int(np.sqrt(self.batch_size))
            h, w, c = real_images.shape[1:]
            summary_real = np.reshape(np.transpose(
                np.reshape(real_images[:n*n],
                           [n, n*h, w, c]), [1, 0, 2, 3]), [n*h, n*w, c])
            summary_fake = np.reshape(np.transpose(
                np.reshape(fake_images[:n*n],
                           [n, n*h, w, c]), [1, 0, 2, 3]), [n*h, n*w, c])
            summary_image = np.concatenate([summary_real, summary_fake], axis=1)
            log.infov(" Writing a summary image: %s ...",
                      self.config.summary_image_name)
            imwrite(self.config.summary_image_name, summary_image)

        if self.config.output_file:
            evaler.dump_result(self.config.output_file)

    def run_single_step(self, batch):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        [step, d_loss, g_loss, fake_images, real_images, output, _] = self.session.run(
            [self.global_step, self.model.d_loss, self.model.g_loss,
             self.model.fake_images, self.model.real_images,
             self.model.output, self.step_op],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )

        _end_time = time.time()

        return step, (_end_time - _start_time), batch_chunk["id"], \
            d_loss, g_loss, fake_images, real_images, output

    def log_step_message(self, step, d_loss, g_loss, step_time, is_train=False):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn(
            (" [{split_mode:5s} step {step:4d}] " +
             "D loss: {d_loss:.5f} G loss: {g_loss:.5f} " +
             "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
             ).format(split_mode=(is_train and 'train' or 'val'),
                      step=step, d_loss=d_loss, g_loss=g_loss,
                      sec_per_batch=step_time,
                      instance_per_sec=self.batch_size / step_time))


def main():

    config, model, dataset_train, dataset_test = argparser(is_train=False)

    evaler = Evaler(config, model, dataset_test)

    log.warning("dataset: %s", config.dataset)
    evaler.eval_run()

if __name__ == '__main__':
    main()
