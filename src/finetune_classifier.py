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
import logging
import warnings
import json
import numpy as np

import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from gluonnlp.model import bert_12_768_12

from .bert import BERTClassifier, BERTRegression
from .tokenizer import FullTokenizer
from .dataset import MRPCDataset, QQPDataset, RTEDataset, \
    STSBDataset, ClassificationTransform, RegressionTransform, SNLISuperficialTransform, \
    QNLIDataset, COLADataset, SNLIDataset, MNLIDataset, WNLIDataset, SSTDataset
from .options import add_default_arguments, add_data_arguments, add_logging_arguments, \
    add_model_arguments, add_training_arguments
from .utils import logging_config, get_dir, metric_to_list


logger = logging.getLogger('nli')

tasks = {
    'MRPC': MRPCDataset,
    'QQP': QQPDataset,
    'QNLI': QNLIDataset,
    'RTE': RTEDataset,
    'STS-B': STSBDataset,
    'CoLA': COLADataset,
    'MNLI': MNLIDataset,
    'SNLI': SNLIDataset,
    'WNLI': WNLIDataset,
    'SST': SSTDataset
}

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
    return parser.parse_args()

def build_model(args, ctx):
    dataset = 'book_corpus_wiki_en_uncased'
    bert, vocabulary = bert_12_768_12(
        dataset_name=dataset,
        pretrained=True,
        ctx=ctx,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False)

    task_name = args.task_name
    num_classes = len(tasks[task_name].get_labels())
    model = BERTClassifier(bert, num_classes=num_classes, dropout=args.dropout)

    do_lower_case = 'uncased' in dataset
    tokenizer = FullTokenizer(vocabulary, do_lower_case=do_lower_case)

    logger.debug(model)
    return model, vocabulary, tokenizer

def dump_vocab(outdir, vocab):
    vocab_path = os.path.join(outdir, 'vocab.jsons')
    with open(vocab_path, 'w') as fout:
        fout.write(vocab.to_json())

def build_data_loader(args, task, split, batch_size, tokenizer, test=False):
    max_len = args.max_len

    if args.superficial:
        trans = SNLISuperficialTransform(
            tokenizer, task.get_labels(), max_len, pad=False)
    else:
        trans = ClassificationTransform(
            tokenizer, task.get_labels(), max_len, pad=False, pair=True)

    dataset = task(split).transform(trans, lazy=False)
    num_samples = len(dataset)
    data_lengths = dataset.transform(
        lambda input_id, length, segment_id, label_id: length)

    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack())
    batch_sampler = nlp.data.FixedBucketSampler(lengths=data_lengths,
                                                batch_size=batch_size,
                                                num_buckets=10,
                                                ratio=0,
                                                shuffle=(not test))
    data_loader = gluon.data.DataLoader(dataset=dataset,
                                       batch_sampler=batch_sampler,
                                       batchify_fn=batchify_fn)
    return data_loader, num_samples


def evaluate(data_loader, model, loss_function, metric, ctx):
    """Evaluate the model on validation dataset.
    """
    loss = 0
    metric.reset()
    for _, seqs in enumerate(data_loader):
        Ls = []
        input_ids, valid_len, type_ids, label = seqs
        out = model(
            input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
            valid_len.astype('float32').as_in_context(ctx))
        loss += loss_function(out, label.as_in_context(ctx)).mean().asscalar()
        metric.update([label], [out])
    loss /= len(data_loader)
    return loss, metric


def train(args, model, train_data, dev_data, num_train_examples, ctx):
    task = tasks[args.task_name]
    model.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    loss_function = gluon.loss.SoftmaxCELoss()
    metric = task.get_metric()

    model.hybridize(static_alloc=True)
    loss_function.hybridize(static_alloc=True)

    lr = args.lr
    optimizer_params_w = {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.01}
    optimizer_params_b = {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.0}
    try:
        trainer_w = gluon.Trainer(
            model.collect_params('.*weight'),
            args.optimizer,
            optimizer_params_w,
            update_on_kvstore=False)
        trainer_b = gluon.Trainer(
            model.collect_params('.*beta|.*gamma|.*bias'),
            args.optimizer,
            optimizer_params_b,
            update_on_kvstore=False)
    except ValueError as e:
        print(e)
        warnings.warn(
            'AdamW optimizer is not found. Please consider upgrading to '
            'mxnet>=1.5.0. Now the original Adam optimizer is used instead.')
        trainer_w = gluon.Trainer(
            model.collect_params('.*weight'),
            'Adam',
            optimizer_params_w,
            update_on_kvstore=False)
        trainer_b = gluon.Trainer(
            model.collect_params('.*beta|.*gamma|.*bias'),
            'Adam',
            optimizer_params_b,
            update_on_kvstore=False)

    num_train_steps = int(num_train_examples / args.batch_size * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_ratio)
    step_num = 0

    # Collect differentiable parameters
    params = [
        p for p in model.collect_params().values() if p.grad_req != 'null'
    ]

    best_dev_loss = float('inf')
    checkpoints_dir = get_dir(os.path.join(args.output_dir, args.exp_id, 'checkpoints'))

    for epoch_id in range(args.epochs):
        metric.reset()
        step_loss = 0
        tic = time.time()
        for batch_id, seqs in enumerate(train_data):
            step_num += 1
            # learning rate schedule
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                offset = (step_num - num_warmup_steps) * lr / (
                    num_train_steps - num_warmup_steps)
                new_lr = lr - offset
            trainer_w.set_learning_rate(new_lr)
            trainer_b.set_learning_rate(new_lr)
            # forward and backward
            with mx.autograd.record():
                input_ids, valid_length, type_ids, label = seqs
                out = model(
                    input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                    valid_length.astype('float32').as_in_context(ctx))
                ls = loss_function(out, label.as_in_context(ctx)).mean()
            ls.backward()
            # update
            trainer_w.allreduce_grads()
            trainer_b.allreduce_grads()
            nlp.utils.clip_grad_global_norm(params, 1)
            trainer_w.update(1)
            trainer_b.update(1)
            step_loss += ls.asscalar()
            metric.update([label], [out])
            if (batch_id + 1) % (args.log_interval) == 0:
                metric_nm, metric_val = metric.get()
                if not isinstance(metric_nm, list):
                    metric_nm = [metric_nm]
                    metric_val = [metric_val]
                eval_str = '[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, metrics=' + \
                    ','.join([i + ':{:.4f}' for i in metric_nm])
                logger.info(eval_str.format(epoch_id + 1, batch_id + 1, len(train_data), \
                    step_loss / args.log_interval, \
                    trainer_w.learning_rate, *metric_val))
                step_loss = 0
        mx.nd.waitall()

        dev_loss, dev_metric = evaluate(dev_data, model, loss_function, metric, ctx)
        metric_names, metric_vals = metric_to_list(dev_metric)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            checkpoint_path = os.path.join(checkpoints_dir, 'valid_best.params')
            model.save_parameters(checkpoint_path)
        logger.info(('[Epoch {}] val_loss={:.4f}, best_val_loss={:.4f}, ' + \
                     'val_metrics=' + \
                     ','.join([i + ':{:.4f}' for i in metric_names]))
                    .format(epoch_id, dev_loss, best_dev_loss, *metric_vals))

        # Save checkpoint of last epoch
        checkpoint_path = os.path.join(checkpoints_dir, 'last.params')
        model.save_parameters(checkpoint_path)

        toc = time.time()
        logger.info('Time cost={:.1f}s'.format(toc - tic))
        tic = toc

def main(args):
    outdir = get_dir(os.path.join(args.output_dir, args.exp_id))
    json.dump(vars(args), open(os.path.join(outdir, 'config.json'), 'w'))

    if args.gpu_id == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    mx.random.seed(args.seed, ctx=ctx)
    np.random.seed(args.seed)
    random.seed(args.seed)

    task = tasks[args.task_name]

    if args.mode == 'train':
        model, vocab, tokenizer = build_model(args, ctx)
        dump_vocab(outdir, vocab)
        train_data, num_train_examples = build_data_loader(args, task, args.train_split, args.batch_size, tokenizer, test=False)
        dev_data, _ = build_data_loader(args, task, args.test_split, args.eval_batch_size, tokenizer, test=True)
        train(args, model, train_data, dev_data, num_train_examples, ctx)
    else:
        model_args = argparse.Namespace(**json.load(
            open(os.path.join(args.init_from, 'config.json'))))
        model, _, tokenizer = build_model(model_args, ctx)
        params_file = 'last.params' if args.use_last else 'valid_best.params'
        model.load_parameters(os.path.join(
            args.init_from, 'checkpoints', params_file), ctx=ctx)
        vocab = nlp.Vocab.from_json(
            open(os.path.join(args.init_from, 'vocab.jsons')).read())
        test_data, _ = build_data_loader(args, task, args.test_split, args.eval_batch_size, tokenizer, test=True)
        loss_function = gluon.loss.SoftmaxCELoss()
        loss_function.hybridize(static_alloc=True)
        loss, metric = evaluate(test_data, model, loss_function, task.get_metric(), ctx)
        metric_names, metric_vals = metric_to_list(metric)
        logger.info(('loss={:.4f}, metrics=' + \
                     ','.join([i + ':{:.4f}' for i in metric_names]))
                    .format(loss, *metric_vals))


if __name__ == '__main__':
    args = parse_args()

    log_path = get_dir(os.path.join(args.output_dir, args.exp_id))
    logging_config(os.path.join(log_path, 'main.log'))

    pool_type = os.environ.get('MXNET_GPU_MEM_POOL_TYPE', '')
    if pool_type.lower() == 'round':
        logger.info(
            'Setting MXNET_GPU_MEM_POOL_TYPE="Round" may lead to higher memory '
            'usage and faster speed. If you encounter OOM errors, please unset '
            'this environment variable.')

    main(args)
