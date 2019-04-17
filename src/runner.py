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
    STSBDataset, ClassificationTransform, RegressionTransform, \
    SNLISuperficialTransform, SNLICheatTransform, SNLIWordDropTransform, \
    CBOWTransform, \
    QNLIDataset, COLADataset, SNLIDataset, MNLIDataset, WNLIDataset, SSTDataset
from .options import add_default_arguments, add_data_arguments, add_logging_arguments, \
    add_model_arguments, add_training_arguments
from .utils import *
from .model_builder import build_model, load_model
from .task import tasks

logger = logging.getLogger('nli')

class EarlyStopper(object):
    def __init__(self, patience=5, delta=0, monitor='loss', larger_is_better=False):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.larger_is_better = larger_is_better
        self.wait = 0

    def compare(self, metric_a, metric_b):
        if self.larger_is_better:
            return metric_a[self.monitor] > metric_b[self.monitor]
        else:
            return metric_a[self.monitor] < metric_b[self.monitor]

    def stop(self, metric, best_metric):
        no_improvement = False
        if self.larger_is_better:
            if metric[self.monitor] + self.delta < best_metric[self.monitor]:
                no_improvement = True
        else:
            if metric[self.monitor] - self.delta > best_metric[self.monitor]:
                no_improvement = True
        if no_improvement:
            if self.wait >= self.patience:
                return True
            else:
                self.wait += 1
        else:
            self.wait = 0
        return False


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
        self.vocab = None
        self.early_stopper = EarlyStopper(monitor='accuracy', larger_is_better=True)

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

    def preprocess_dataset(self, split, cheat_rate, max_num_examples, ctx=None):
        logger.info('preprocess {} data'.format(split))
        dataset = self.task(split, max_num_examples=max_num_examples)
        if cheat_rate >= 0:
            trans = self.build_cheat_transformer(cheat_rate)
            # Make sure we have the same data
            trans.reset()
            dataset = dataset.transform(trans, lazy=False)
        return dataset

    def run_train(self, args, ctx):
        train_dataset = self.preprocess_dataset(args.train_split, args.cheat, args.max_num_examples, ctx)
        dev_dataset = self.preprocess_dataset(args.test_split, args.cheat, args.max_num_examples, ctx)

        model, vocab, tokenizer = build_model(args, args, ctx, train_dataset)
        self.dump_vocab(vocab)
        self.vocab = vocab

        self.train(args, model, train_dataset, dev_dataset, ctx, tokenizer, args.noising_by_epoch)

    def run_test(self, args, ctx, dataset=None):
        model_args = read_args(args.init_from)
        model, vocab, tokenizer = load_model(args, model_args, args.init_from, ctx)
        self.vocab = vocab
        if dataset:
            test_dataset = dataset
        else:
            test_dataset = self.preprocess_dataset(args.test_split, args.cheat, args.max_num_examples, ctx)
        test_data = self.build_data_loader(test_dataset, args.eval_batch_size, model_args.max_len, tokenizer, test=True, ctx=ctx)
        metrics, preds, labels, scores, ids = self.evaluate(test_data, model, self.task.get_metric(), ctx)
        logger.info(metric_dict_to_str(metrics))
        self.update_report(('test', args.test_split), metrics)
        return preds, scores, ids

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

    def build_data_transformer(self, max_len, tokenizer, word_dropout, word_dropout_region):
        trans_list = []
        if word_dropout > 0:
            if word_dropout_region is None:
                word_dropout_region = ('premise', 'hypothesis')
            trans_list.append(SNLIWordDropTransform(rate=word_dropout, region=word_dropout_region))
        trans_list.append(self.build_model_transformer(max_len, tokenizer))
        return [x for x in trans_list if x]

    def build_dataset(self, data, max_len, tokenizer, word_dropout=0, word_dropout_region=None, ctx=None):
        trans_list = self.build_data_transformer(max_len, tokenizer, word_dropout, word_dropout_region)
        dataset = data
        for trans in trans_list:
            dataset = dataset.transform(trans)
        # Last transform
        trans = trans_list[-1]
        data_lengths = dataset.transform(trans.get_length)
        batchify_fn = trans.get_batcher()
        return dataset, data_lengths, batchify_fn

    def build_data_loader(self, dataset, batch_size, max_len, tokenizer, test=False, word_dropout=0, word_dropout_region=None, ctx=None):
        dataset, data_lengths, batchify_fn = self.build_dataset(dataset, max_len, tokenizer, word_dropout, word_dropout_region, ctx=ctx)

        batch_sampler = nlp.data.FixedBucketSampler(lengths=data_lengths,
                                                    batch_size=batch_size,
                                                    num_buckets=10,
                                                    ratio=0,
                                                    shuffle=(not test))
        data_loader = gluon.data.DataLoader(dataset=dataset,
                                           batch_sampler=batch_sampler,
                                           batchify_fn=batchify_fn)
        return data_loader

    def prepare_data(self, data, ctx):
        """Batched data to model inputs.
        """
        id_, input_ids, valid_len, type_ids, label = data
        inputs = (input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                  valid_len.astype('float32').as_in_context(ctx))
        label = label.as_in_context(ctx)
        return id_, inputs, label

    def evaluate(self, data_loader, model, metric, ctx):
        """Evaluate the model on validation dataset.
        """
        self.loss_function.hybridize(static_alloc=True)
        loss = 0
        metric.reset()
        preds = []
        labels = []
        scores = None
        ids = None
        for _, seqs in enumerate(data_loader):
            Ls = []
            id_, inputs, label = self.prepare_data(seqs, ctx)
            out = model(*inputs)
            if scores is None:
                scores = out
                ids = id_
            else:
                scores = mx.nd.concat(scores, out, dim=0)
                ids = mx.nd.concat(ids, id_, dim=0)
            _preds = mx.ndarray.argmax(out, axis=1)
            preds.extend(_preds.asnumpy().astype('int32'))
            labels.extend(label[:,0].asnumpy())
            loss += self.loss_function(out, label).mean().asscalar()
            metric.update([label], [out])
        loss /= len(data_loader)
        metric = metric_to_dict(metric)
        metric['loss'] = loss
        return metric, preds, labels, scores, ids

    def get_optimizer_params(self, optimizer, lr, param_type='weight'):
        assert param_type in ('weight', 'bias')
        if optimizer == 'bertadam':
            if param_type == 'weight':
                return {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.01}
            else:
                return {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.0}
        if optimizer == 'adagrad':
            if param_type == 'weight':
                return {'learning_rate': lr, 'wd': 0.0}
            else:
                return {'learning_rate': lr, 'wd': 0.0}

    def initialize_model(self, args, model, ctx):
        model.initialize(init=mx.init.Normal(0.02), ctx=ctx, force_reinit=False)

    def train(self, args, model, train_dataset, dev_dataset, ctx, tokenizer, data_noising_by_epoch):
        task = self.task
        loss_function = self.loss_function
        metric = task.get_metric()
        num_train_examples = len(train_dataset)

        self.initialize_model(args, model, ctx)

        model.hybridize(static_alloc=True)
        loss_function.hybridize(static_alloc=True)

        # TODO: refactor this as get_trainer
        lr = args.lr
        optimizer_params_w = self.get_optimizer_params(args.optimizer, args.lr, 'weight')
        optimizer_params_b = self.get_optimizer_params(args.optimizer, args.lr, 'bias')
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

        best_dev_metrics = None
        terminate_training = False
        checkpoints_dir = get_dir(os.path.join(self.outdir, 'checkpoints'))

        train_data = self.build_data_loader(train_dataset, args.batch_size, args.max_len, tokenizer, test=False, word_dropout=args.word_dropout, word_dropout_region=args.word_dropout_region, ctx=ctx)
        dev_data = self.build_data_loader(dev_dataset, args.batch_size, args.max_len, tokenizer, test=True, word_dropout=0, ctx=ctx)

        for epoch_id in range(args.epochs):
            metric.reset()
            step_loss = 0
            tic = time.time()

            if data_noising_by_epoch and epoch_id > 0:
                train_data = self.build_data_loader(train_dataset, args.batch_size, args.max_len, tokenizer, test=False, word_dropout=args.word_dropout, word_dropout_region=args.word_dropout_region, ctx=ctx)

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
                    id_, inputs, label = self.prepare_data(seqs, ctx)
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

            dev_metrics, _, _, _, _ = self.evaluate(dev_data, model, metric, ctx)
            if best_dev_metrics and self.early_stopper.stop(dev_metrics, best_dev_metrics):
                terminate_training = True
            if best_dev_metrics is None or self.early_stopper.compare(dev_metrics, best_dev_metrics):
                best_dev_metrics = dev_metrics
                checkpoint_path = os.path.join(checkpoints_dir, 'valid_best.params')
                model.save_parameters(checkpoint_path)
                self.update_report(('train', 'best_val_results'), dev_metrics)


            #dev_loss = dev_metrics['loss']
            #if dev_loss < best_dev_loss:
            #    best_dev_loss = dev_loss
            #    checkpoint_path = os.path.join(checkpoints_dir, 'valid_best.params')
            #    model.save_parameters(checkpoint_path)
            #    self.update_report(('train', 'best_val_results'), dev_metrics)

            metric_names = sorted(dev_metrics.keys())
            logger.info('[Epoch {}] val_metrics={}'.format(
                        epoch_id, metric_dict_to_str(dev_metrics)))

            # Save checkpoint of last epoch
            checkpoint_path = os.path.join(checkpoints_dir, 'last.params')
            model.save_parameters(checkpoint_path)

            toc = time.time()
            logger.info('Time cost={:.1f}s'.format(toc - tic))
            tic = toc

            if terminate_training:
                logger.info('early stopping')
                break


class SuperficialNLIRunner(NLIRunner):
    def build_model_transformer(self, max_len, tokenizer):
        trans = SNLISuperficialTransform(
            tokenizer, self.labels, max_len, pad=False)
        return trans

class CBOWNLIRunner(NLIRunner):
    def build_model_transformer(self, max_len, tokenizer):
        trans = CBOWTransform(self.labels, tokenizer, self.vocab, num_input_sentences=2)
        return trans

    def prepare_data(self, data, ctx):
        """Batched data to model inputs.
        """
        id_, input_ids, valid_len, label = data
        inputs = ([x.as_in_context(ctx) for x in input_ids],
                  [x.astype('float32').as_in_context(ctx) for x in valid_len])
        label = label.as_in_context(ctx)
        return id_, inputs, label

    def initialize_model(self, args, model, ctx):
        model.initialize(init=mx.init.Normal(0.02), ctx=ctx, force_reinit=False)
        # Initialize word embeddings
        if args.embedding_source:
            glove = nlp.embedding.create('glove', source=args.embedding_source)
            self.vocab.set_embedding(glove)
            model.embedding.weight.set_data(self.vocab.embedding.idx_to_vec)
            if args.fix_word_embedding:
                model.embedding.weight.req_grad = 'null'

class AdditiveNLIRunner(NLIRunner):
    """Additive model of a superficial classifier and a normal classifier.
    """
    def __init__(self, task, runs_dir, prev_runner, prev_args, run_id=None):
        # Runner for the previous model
        self.prev_runner = prev_runner
        self.prev_args = prev_args
        super().__init__(task, runs_dir, run_id)

    def run_prev_model(self, dataset, ctx):
        logger.info('running previous model on preprocessed dataset')
        _, prev_scores, ids = self.prev_runner.run_test(self.prev_args, ctx, dataset)
        return prev_scores, ids

    def preprocess_dataset(self, split, cheat_rate, max_num_examples, ctx=None):
        """Add scores from previous classifiers.
        """
        dataset = super().preprocess_dataset(split, cheat_rate, max_num_examples)

        prev_scores, ids = self.run_prev_model(dataset, ctx)
        assert len(dataset) == len(prev_scores)

        # Reorder scores by example id
        prev_scores = {id_.asscalar(): prev_scores[i] for i, id_ in enumerate(ids)}
        reordered_prev_scores = []
        for data in dataset:
            id_ = data[0]
            reordered_prev_scores.append(prev_scores[id_])
        prev_scores = mx.nd.stack(*reordered_prev_scores, axis=0)
        prev_scores = prev_scores.asnumpy()

        return gluon.data.ArrayDataset(prev_scores, dataset)

    def build_dataset(self, data, max_len, tokenizer, word_dropout=0, word_dropout_region=None, ctx=None):
        trans_list = self.build_data_transformer(max_len, tokenizer, word_dropout, word_dropout_region)
        prev_scores = [x[0] for x in data]
        dataset = gluon.data.SimpleDataset([x[1] for x in data])
        for trans in trans_list:
            dataset = dataset.transform(trans)
        # Last transform
        trans = trans_list[-1]
        data_lengths = dataset.transform(trans.get_length)
        batchify_fn = trans.get_batcher()
        # Combine with prev_scores
        batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Stack(), batchify_fn)
        dataset = gluon.data.ArrayDataset(prev_scores, dataset)
        return dataset, data_lengths, batchify_fn

    def prepare_data(self, data, ctx):
        prev_scores, model_data = data
        prev_scores = prev_scores.astype('float32').as_in_context(ctx)
        id_, inputs, label = super().prepare_data(model_data, ctx)
        return id_, [prev_scores, inputs], label

    def evaluate(self, data_loader, model, metric, ctx):
        original_mode = model.mode
        metric_dict = metric_to_dict(metric)
        results = {}
        for mode in ('all', 'prev', 'last'):
            logger.info('evaluating additive model with mode={}'.format(mode))
            model.mode = mode
            metric.reset()
            _metric_dict, preds, labels, scores, ids = super().evaluate(data_loader, model, metric, ctx)
            results[mode] = (_metric_dict, preds, labels, scores, ids)
            for k, v in _metric_dict.items():
                metric_dict['{}_{}'.format(model.mode, k)] = v
            # The original_mode result will be used for model selection
            if mode == original_mode:
                metric_dict.update(_metric_dict)
        model.mode = original_mode
        _, preds, labels, scores, ids = results[original_mode]
        return metric_dict, preds, labels, scores, ids

