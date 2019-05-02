"""
Sentence Pair Classification with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
sentence pair classification, with Gluon NLP Toolkit.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import os
import time
import argparse
import random
#import logging
import warnings
import json
import numpy as np
import pickle as pkl

from .utils import read_args
from .runner import *
from .options import add_default_arguments, add_data_arguments, add_logging_arguments, \
    add_model_arguments, add_training_arguments, \
    check_arguments

logger = logging.getLogger('nli')


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()
    add_default_arguments(parser)
    add_data_arguments(parser)
    add_logging_arguments(parser)
    add_model_arguments(parser)
    add_training_arguments(parser)
    args = parser.parse_args()
    check_arguments(args)
    return args

def get_runner(args, model_args, task, output_dir=None):
    core_runner = {
            'cbow': CBOWNLIRunner,
            'bert': BERTNLIRunner,
            'da': DANLIRunner,
            }
    if not output_dir:
        output_dir = args.output_dir
    if model_args.superficial == 'hypothesis':
        runner = HypothesisNLIRunner(task, output_dir, args.exp_id)
    elif model_args.superficial == 'handcrafted':
        runner = HandcraftedNLIRunner(task, output_dir, args.exp_id)
    elif model_args.additive:
        prev_runners = []
        prev_args = []
        for i, path in enumerate(model_args.additive):
            _prev_args = read_args(path)
            # Change to inference model
            _prev_args.init_from = path
            _prev_args.dropout = 0.0
            _prev_runner = get_runner(args, _prev_args, task, output_dir='/tmp/{}/{}'.format(args.exp_id, i))
            prev_runners.append(_prev_runner)
            prev_args.append(_prev_args)
        runner = get_additive_runner(core_runner[model_args.model_type])(task, output_dir, prev_runners, prev_args, args.exp_id)
    else:
        runner = core_runner[model_args.model_type](task, output_dir, args.exp_id)
    return runner

def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.mode == 'test':
        model_args = read_args(args.init_from)
    else:
        model_args = args

    task = tasks[args.task_name]

    runner = get_runner(args, model_args, task)

    pool_type = os.environ.get('MXNET_GPU_MEM_POOL_TYPE', '')
    if pool_type.lower() == 'round':
        logger.info(
            'Setting MXNET_GPU_MEM_POOL_TYPE="Round" may lead to higher memory '
            'usage and faster speed. If you encounter OOM errors, please unset '
            'this environment variable.')

    try:
        runner.run(args)
        runner.dump_report()
    except KeyboardInterrupt as e:
        print(e)
        logger.info('Terminated. Dumping report.')
        runner.dump_report()


if __name__ == '__main__':
    args = parse_args()
    main(args)
