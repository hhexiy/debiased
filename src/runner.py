import os
import time
import argparse
import random
import logging
import string
import warnings
import json
import numpy as np
import pickle as pkl
import uuid
import glob

import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp

from .model.bert import BERTClassifier, BERTRegression
from .dataset import MRPCDataset, QQPDataset, RTEDataset, \
    STSBDataset, ClassificationTransform, RegressionTransform, SNLISuperficialTransform, AdditiveTransform, SNLICheatTransform, \
    QNLIDataset, COLADataset, SNLIDataset, MNLIDataset, WNLIDataset, SSTDataset
from .options import add_default_arguments, add_data_arguments, add_logging_arguments, \
    add_model_arguments, add_training_arguments
from .utils import *
from .model_builder import build_model, load_model
from .task import tasks

logger = logging.getLogger('nli')


class Runner(object):
    def __init__(self, task, runs_dir, run_id=None):
        self.report = {}
        self.task = task
        self.run_id = self.get_run_id(run_id)
        self.outdir = get_dir(os.path.join(runs_dir, self.run_id))
        logger.info('all output saved in {}'.format(self.outdir))
        logging_config(os.path.join(self.outdir, 'console.log'))
        self.update_report(('run_id',), self.run_id)

    def get_run_id(self, run_id=None):
        if not run_id:
            return str(uuid.uuid1()).replace('/', '_')
        return run_id

    def dump_vocab(self, vocab):
        vocab_path = os.path.join(self.outdir, 'vocab.jsons')
        with open(vocab_path, 'w') as fout:
            fout.write(vocab.to_json())

    def dump_report(self):
        report_path = os.path.join(self.outdir, 'report.json')
        json.dump(self.report, open(report_path, 'w'))

    def update_report(self, keys, val):
        d = self.report
        for k in keys[:-1]:
            if not k in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = val

class NLIRunner(Runner):
    def __init__(self, task, runs_dir, run_id=None):
        super().__init__(task, runs_dir, run_id)
        self.loss_function = gluon.loss.SoftmaxCELoss()
        self.labels = task.get_labels()

    def run(self, args):
        self.update_report(('config',), vars(args))

        if args.gpu_id == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(args.gpu_id)
        mx.random.seed(args.seed, ctx=ctx)

        if args.mode == 'train':
            self.run_train(args, ctx)
        else:
            self.run_test(args, ctx)

    def run_train(self, args, ctx):
        model, vocab, tokenizer = build_model(args, args, ctx)
        self.dump_vocab(vocab)
        train_data, num_train_examples = self.build_data_loader(args.train_split, args.batch_size, args.max_len, tokenizer, test=False, cheat_rate=args.cheat, max_num_examples=args.max_num_examples, ctx=ctx)
        # NOTE: If cheating is enabled, we want to randomize the cheating feature at test time (cheat_rate = 0); otherwise, we don't want cheating features.
        dev_data, _ = self.build_data_loader(args.test_split, args.batch_size, args.max_len, tokenizer, test=True, max_num_examples=args.max_num_examples, cheat_rate=0 if args.cheat > 0 else -1, ctx=ctx)
        self.train(args, model, train_data, dev_data, num_train_examples, ctx)

    def run_test(self, args, ctx):
        config = json.load(
            open(os.path.join(args.init_from, 'report.json')))['config']
        model_args = argparse.Namespace(**config)
        model, vocab, tokenizer = load_model(args, model_args, args.init_from, ctx)
        test_data, num_examples = self.build_data_loader(args.test_split, args.eval_batch_size, model_args.max_len, tokenizer, test=True, max_num_examples=args.max_num_examples, cheat_rate=args.cheat, ctx=ctx)
        metrics, preds, labels, scores = self.evaluate(test_data, model, self.task.get_metric(), ctx)
        logger.info(metric_dict_to_str(metrics))
        self.update_report(('test', args.test_split), metrics)
        return preds, scores

    def build_cheat_transformer(self, cheat_rate):
        if cheat_rate < 0:
            return None
        else:
            logger.info('cheating rate: {}'.format(cheat_rate))
            return SNLICheatTransform(self.task.get_labels(), rate=cheat_rate)

    def build_model_transformer(self, max_len, tokenizer):
        trans = ClassificationTransform(
            tokenizer, self.labels, max_len, pad=False, pair=True)
        return trans

    def build_data_transformer(self, max_len, tokenizer, cheat_rate=0):
        trans_list = []
        trans_list.append(self.build_cheat_transformer(cheat_rate))
        trans_list.append(self.build_model_transformer(max_len, tokenizer))
        return [x for x in trans_list if x]

    def build_dataset(self, split, max_len, tokenizer, cheat_rate=0, max_num_examples=-1, ctx=None):
        trans_list = self.build_data_transformer(max_len, tokenizer, cheat_rate=cheat_rate)
        dataset = self.task(split, max_num_examples=max_num_examples)
        for trans in trans_list:
            dataset = dataset.transform(trans)
        # Last transform
        trans = trans_list[-1]
        data_lengths = dataset.transform(trans.get_length)
        batchify_fn = trans.get_batcher()
        return dataset, data_lengths, batchify_fn

    def build_data_loader(self, split, batch_size, max_len, tokenizer, test=False, cheat_rate=0, max_num_examples=-1, ctx=None):
        logger.info('building data loader for data split={}'.format(split))

        dataset, data_lengths, batchify_fn = self.build_dataset(split, max_len, tokenizer, cheat_rate, max_num_examples, ctx=ctx)
        num_samples = len(dataset)

        batch_sampler = nlp.data.FixedBucketSampler(lengths=data_lengths,
                                                    batch_size=batch_size,
                                                    num_buckets=10,
                                                    ratio=0,
                                                    shuffle=(not test))
        data_loader = gluon.data.DataLoader(dataset=dataset,
                                           batch_sampler=batch_sampler,
                                           batchify_fn=batchify_fn)
        return data_loader, num_samples

    def prepare_data(self, data, ctx):
        input_ids, valid_len, type_ids, label = data
        inputs = (input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                  valid_len.astype('float32').as_in_context(ctx))
        label = label.as_in_context(ctx)
        return inputs, label

    def evaluate(self, data_loader, model, metric, ctx):
        """Evaluate the model on validation dataset.
        """
        self.loss_function.hybridize(static_alloc=True)
        loss = 0
        metric.reset()
        preds = []
        labels = []
        scores = None
        for _, seqs in enumerate(data_loader):
            Ls = []
            inputs, label = self.prepare_data(seqs, ctx)
            out = model(*inputs)
            if scores is None:
                scores = out
            else:
                scores = mx.nd.concat(scores, out, dim=0)
            _preds = mx.ndarray.argmax(out, axis=1)
            preds.extend(_preds.asnumpy().astype('int32'))
            labels.extend(label[:,0].asnumpy())
            loss += self.loss_function(out, label).mean().asscalar()
            metric.update([label], [out])
        loss /= len(data_loader)
        metric = metric_to_dict(metric)
        metric['loss'] = loss
        scores = scores.asnumpy().astype('float32')
        return metric, preds, labels, scores

    def train(self, args, model, train_data, dev_data, num_train_examples, ctx):
        task = self.task
        loss_function = self.loss_function
        metric = task.get_metric()

        model.initialize(init=mx.init.Normal(0.02), ctx=ctx, force_reinit=False)
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
        checkpoints_dir = get_dir(os.path.join(self.outdir, 'checkpoints'))

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
                    inputs, label = self.prepare_data(seqs, ctx)
                    out = model(*inputs)
                    ls = loss_function(out, label).mean()
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

            dev_metrics, _, _, _ = self.evaluate(dev_data, model, metric, ctx)
            dev_loss = dev_metrics['loss']
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                checkpoint_path = os.path.join(checkpoints_dir, 'valid_best.params')
                model.save_parameters(checkpoint_path)
                self.update_report(('train', 'best_val_results'), dev_metrics)
            metric_names = sorted(dev_metrics.keys())
            logger.info('[Epoch {}] val_loss={:.4f}, val_metrics={}'.format(
                        epoch_id, dev_loss, metric_dict_to_str(dev_metrics)))

            # Save checkpoint of last epoch
            checkpoint_path = os.path.join(checkpoints_dir, 'last.params')
            model.save_parameters(checkpoint_path)

            toc = time.time()
            logger.info('Time cost={:.1f}s'.format(toc - tic))
            tic = toc

class SuperficialNLIRunner(NLIRunner):
    def build_model_transformer(self, max_len, tokenizer):
        trans = SNLISuperficialTransform(
            tokenizer, self.labels, max_len, pad=False)
        return trans

class AdditiveNLIRunner(NLIRunner):
    """Additive model of a superficial classifier and a normal classifier.
    """
    def __init__(self, task, runs_dir, prev_runner, prev_args, run_id=None):
        # Runner for the previous model
        self.prev_runner = prev_runner
        self.prev_args = prev_args
        super().__init__(task, runs_dir, run_id)

    def run_prev_model(self, split, ctx):
        logger.info('running previous model on {}'.format(split))
        self.prev_args.test_split = split
        _, prev_scores = self.prev_runner.run_test(self.prev_args, ctx)
        return prev_scores

    def build_dataset(self, split, max_len, tokenizer, cheat_rate=0, max_num_examples=-1, ctx=None):
        dataset, data_lengths, batchify_fn = super().build_dataset(split, max_len, tokenizer, cheat_rate, max_num_examples)
        prev_scores = self.run_prev_model(split, ctx)
        assert len(dataset) == len(prev_scores)
        batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Stack(), batchify_fn)
        return mx.gluon.data.ArrayDataset(prev_scores, dataset), data_lengths, batchify_fn

    def prepare_data(self, data, ctx):
        prev_scores, model_data = data
        prev_scores = prev_scores.astype('float32').as_in_context(ctx)
        inputs, label = super().prepare_data(model_data, ctx)
        return [prev_scores, inputs], label

    def evaluate(self, data_loader, model, metric, ctx):
        metric_dict, preds, labels, scores = super().evaluate(data_loader, model, metric, ctx)
        original_mode = model.mode
        for mode in ('all', 'prev', 'last'):
            if mode != original_mode:
                metric.reset()
                _metric_dict, _, _, _ = super().evaluate(data_loader, model, metric, ctx)
                for k, v in _metric_dict.items():
                    metric_dict['{}_{}'.format(mode, k)] = v
        model.mode = original_mode
        return metric_dict, preds, labels, scores

