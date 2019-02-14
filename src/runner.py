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

import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp

from .bert import BERTClassifier, BERTRegression
from .dataset import MRPCDataset, QQPDataset, RTEDataset, \
    STSBDataset, ClassificationTransform, RegressionTransform, SNLISuperficialTransform, AdditiveTransform, SNLICheatTransform, \
    QNLIDataset, COLADataset, SNLIDataset, MNLIDataset, WNLIDataset, SSTDataset
from .options import add_default_arguments, add_data_arguments, add_logging_arguments, \
    add_model_arguments, add_training_arguments
from .utils import logging_config, get_dir, metric_to_list
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

    def run_train(self, args, ctx):
        model, vocab, tokenizer = build_model(args, ctx)
        self.dump_vocab(vocab)
        train_data, num_train_examples = self.build_data_loader(args.train_split, args.batch_size, args.max_len, tokenizer, test=False, cheat_percent=args.cheat, max_num_examples=args.max_num_examples)
        dev_data, _ = self.build_data_loader(args.test_split, args.batch_size, args.max_len, tokenizer, test=True, max_num_examples=args.max_num_examples)
        self.train(args, model, train_data, dev_data, num_train_examples, ctx)

    def run_test(self, args, ctx):
        model_args = argparse.Namespace(**json.load(
            open(os.path.join(args.init_from, 'config.json'))))
        model, vocab, tokenizer = load_model(args, model_args, args.init_from, ctx)
        test_data, num_examples = self.build_data_loader(model_args.test_split, args.eval_batch_size, model_args.max_len, tokenizer, test=True, max_num_examples=args.max_num_examples)
        loss, metric, preds, labels = self.evaluate(test_data, model, task.get_metric(), ctx)
        metric_names, metric_vals = metric_to_list(metric)
        logger.info(('loss={:.4f}, metrics=' + \
                     ','.join([i + ':{:.4f}' for i in metric_names]))
                    .format(loss, *metric_vals))

    def build_cheat_transformer(self, cheat_percent):
        if cheat_percent <= 0:
            return None
        else:
            logger.info('cheating rate: {}'.format(cheat_percent))
            return SNLICheatTransform(self.task.get_labels(), percent=cheat_percent)

    def build_model_transformer(self, max_len, tokenizer):
        trans = ClassificationTransform(
            tokenizer, self.labels, max_len, pad=False, pair=True)
        return trans

    def build_data_transformer(self, max_len, tokenizer, cheat_percent=0):
        trans_list = []
        trans_list.append(self.build_cheat_transformer(cheat_percent))
        trans_list.append(self.build_model_transformer(max_len, tokenizer))
        return [x for x in trans_list if x]

    def build_data_loader(self, split, batch_size, max_len, tokenizer, test=False, cheat_percent=0, max_num_examples=-1):
        trans_list = self.build_data_transformer(max_len, tokenizer, cheat_percent=cheat_percent)
        dataset = self.task(split, max_num_examples=max_num_examples)
        for trans in trans_list:
            dataset = dataset.transform(trans)
        # Last transform
        trans = trans_list[-1]

        num_samples = len(dataset)
        data_lengths = dataset.transform(trans.get_length)

        batchify_fn = trans.get_batcher()
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
        for _, seqs in enumerate(data_loader):
            Ls = []
            inputs, label = self.prepare_data(seqs, ctx)
            out = model(*inputs)
            _preds = mx.ndarray.argmax(out, axis=1)
            preds.extend(_preds.asnumpy().astype('int32'))
            labels.extend(label[:,0].asnumpy())
            loss += self.loss_function(out, label).mean().asscalar()
            metric.update([label], [out])
        loss /= len(data_loader)
        return loss, metric, preds, labels

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

            dev_loss, dev_metric, _, _ = self.evaluate(dev_data, model, metric, ctx)
            metric_names, metric_vals = metric_to_list(dev_metric)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                checkpoint_path = os.path.join(checkpoints_dir, 'valid_best.params')
                model.save_parameters(checkpoint_path)
                self.update_report(('train', 'best_val_results', 'loss'), best_dev_loss)
                for name, val in zip(metric_names, metric_vals):
                    self.update_report(('train', 'best_val_results', name), val)
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

class SuperficialNLIRunner(NLIRunner):
    def build_model_transformer(self, max_len, tokenizer):
        trans = SNLISuperficialTransform(
            tokenizer, self.labels, max_len, pad=False)
        return trans

class AdditiveNLIRunner(NLIRunner):
    """Additive model of a superficial classifier and a normal classifier.
    """
    def build_model_transformer(self, max_len, tokenizer):
        sup_trans = SNLISuperficialTransform(
            tokenizer, self.labels, max_len, pad=False)
        all_trans = ClassificationTransform(
            tokenizer, self.labels, max_len, pad=False, pair=True)
        trans = AdditiveTransform([sup_trans, all_trans], use_length=1)
        return trans

    def prepare_data(self, data, ctx):
        inputs = []
        label = None
        # inputs going throught two transforms respectively
        for d in data:
            _inputs, _label = super().prepare_data(d, ctx)
            inputs.append(_inputs)
            label = _label
        return [inputs], label
