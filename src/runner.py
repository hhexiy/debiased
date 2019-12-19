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
import time
import csv
import itertools

import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from gluonnlp.model import BERTClassifier, RoBERTaClassifier
from gluonnlp.data import BERTTokenizer

from .model.additive import AdditiveClassifier
from .model.hex import ProjectClassifier
from .model.cbow import NLICBOWClassifier, NLIHandcraftedClassifier
from .model.decomposable_attention import DecomposableAttentionClassifier
from .model.esim import ESIMClassifier
from .dataset import BERTDatasetTransform, MaskedBERTDatasetTransform, \
    NLIHypothesisTransform, SNLICheatTransform, SNLIWordDropTransform, \
    CBOWTransform, NLIHandcraftedTransform, DATransform, ESIMTransform
from .utils import *
from .task import tasks
from .tokenizer import SNLITokenizer, BasicTokenizer

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
        self.tokenizer = None
        self.early_stopper = EarlyStopper(monitor='accuracy', larger_is_better=True)

    def build_vocab(self, dataset, reserved_tokens=None):
        # get_input(ex): id_, ..., label
        sentences = itertools.chain.from_iterable([self.get_input(ex)[1:-1] for ex in dataset])
        tokens = [self.tokenizer.tokenize(s) for s in sentences]
        counter = nlp.data.count_tokens(list(itertools.chain.from_iterable(tokens)))
        vocab = nlp.Vocab(counter, bos_token=None, eos_token=None, reserved_tokens=reserved_tokens)
        logger.info('built vocabulary of size {}'.format(len(vocab)))
        return vocab

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

    def preprocess_dataset(self, split, cheat_rate, remove_cheat, remove_correct, max_num_examples, ctx=None):
        dataset = self.task(segment=split, max_num_examples=max_num_examples)
        logger.info('preprocess {} {} data'.format(len(dataset), split))
        if cheat_rate >= 0:
            trans = self.build_cheat_transformer(cheat_rate, remove_cheat)
            # Make sure we have the same data
            trans.reset()
            dataset = dataset.transform(trans, lazy=False)
            if remove_cheat:
                dataset = gluon.data.SimpleDataset([ex for ex in dataset if ex is not None])
                logger.info('after remove cheated examples: {}'.format(len(dataset)))
        return dataset

    def get_input(self, example):
        """Convert an example in the preprocessed dataset to a list of values.
        """
        # id_, premise, hypothesis, label
        return example

    def build_model(self, args, model_args, ctx, dataset=None, vocab=None):
        raise NotImplementedError

    def run_train(self, args, ctx):
        train_dataset = self.preprocess_dataset(args.train_split, args.cheat, args.remove_cheat, args.remove, args.max_num_examples, ctx)
        dev_dataset = self.preprocess_dataset(args.test_split, args.cheat, args.remove_cheat, args.remove, args.max_num_examples, ctx)

        if args.init_from:
            model, vocab = self.load_model(args, args, args.init_from, ctx)
        else:
            model, vocab = self.build_model(args, args, ctx, train_dataset)
        self.dump_vocab(vocab)
        self.vocab = vocab

        self.train(args, model, train_dataset, dev_dataset, ctx, args.augment_by_epoch)

    def dump_predictions(self, dataset, preds, ids):
        ids = ids.asnumpy().astype('int32')
        preds_dict = {i: p for i, p in zip(ids, preds)}
        with open(os.path.join(self.outdir, 'predictions.tsv'), 'w', encoding='utf-8') as fout:
            writer = csv.writer(fout, delimiter='\t')
            writer.writerow(['id', 'premise', 'hypothesis', 'label', 'pred', 'correct'])
            for d in dataset:
                id_, prem, hypo, label = self.get_input(d)
                pred = self.task.get_labels()[preds_dict[id_]]
                writer.writerow([id_, prem, hypo, label, pred, pred == label])

    def load_model(self, args, model_args, path, ctx):
        vocab = nlp.Vocab.from_json(
            open(os.path.join(path, 'vocab.jsons')).read())
        model, _ = self.build_model(args, model_args, ctx, vocab=vocab)
        params_file = 'last.params' if args.use_last else None if args.use_init else 'valid_best.params'
        if params_file is None:
            logger.info('using randomly initialized model with seed={}!'.format(model_args.seed))
            self.initialize_model(model_args, model, ctx)
        else:
            logger.info('load model from {}'.format(os.path.join(
                path, 'checkpoints', params_file)))
            model.load_parameters(os.path.join(
                path, 'checkpoints', params_file), ctx=ctx)
        return model, vocab

    def run_test(self, args, ctx, dataset=None):
        model_args = read_args(args.init_from)
        model, self.vocab = self.load_model(args, model_args, args.init_from, ctx)
        if dataset:
            test_dataset = dataset
        else:
            test_dataset = self.preprocess_dataset(args.test_split, args.cheat, args.remove_cheat, args.remove, args.max_num_examples, ctx)
        test_data = self.build_data_loader(test_dataset, args.eval_batch_size, model_args.max_len, test=True, ctx=ctx)
        metrics, preds, labels, scores, ids = self.evaluate(test_data, model, self.task.get_metric(), ctx)
        self.dump_predictions(test_dataset, preds, ids)
        logger.info(metric_dict_to_str(metrics))
        self.update_report(('test', args.test_split), metrics)
        return preds, scores, ids

    def build_cheat_transformer(self, cheat_rate, remove_cheat):
        if cheat_rate < 0:
            return None
        else:
            logger.info('cheating rate: {}'.format(cheat_rate))
            return SNLICheatTransform(self.task.get_labels(), rate=cheat_rate, remove=remove_cheat)

    def build_data_transformer(self, max_len, example_augment_prob, word_mask, word_dropout, word_dropout_region):
        trans_list = []
        if word_dropout > 0:
            if word_dropout_region is None:
                word_dropout_region = ('premise', 'hypothesis')
            trans_list.append(SNLIWordDropTransform(example_augment_prob=example_augment_prob, rate=word_dropout, region=word_dropout_region))
        # TODO: separate augment transformer from model transformer
        trans_list.append(self.build_model_transformer(max_len, example_augment_prob=example_augment_prob, word_mask_prob=word_mask))
        return [x for x in trans_list if x]

    def build_dataset(self, data, max_len, example_augment_prob=0, word_mask=0, word_dropout=0, word_dropout_region=None, ctx=None):
        trans_list = self.build_data_transformer(max_len, example_augment_prob, word_mask, word_dropout, word_dropout_region)
        dataset = data
        logger.info('processing {} examples'.format(len(dataset)))
        start = time.time()
        for trans in trans_list:
            dataset = dataset.transform(trans, lazy=False)
        logger.info('elapsed time: {:.2f}s'.format(time.time() - start))
        # Last transform
        trans = trans_list[-1]
        data_lengths = dataset.transform(trans.get_length)
        batchify_fn = trans.get_batcher()
        return dataset, data_lengths, batchify_fn

    def build_data_loader(self, dataset, batch_size, max_len, test=False, example_augment_prob=0, word_mask=0, word_dropout=0, word_dropout_region=None, ctx=None):
        dataset, data_lengths, batchify_fn = self.build_dataset(dataset, max_len, example_augment_prob=example_augment_prob, word_mask=word_mask, word_dropout=word_dropout, word_dropout_region=word_dropout_region, ctx=ctx)

        batch_sampler = nlp.data.FixedBucketSampler(lengths=data_lengths,
                                                    batch_size=batch_size,
                                                    num_buckets=10,
                                                    ratio=0,
                                                    shuffle=(not test))
        data_loader = gluon.data.DataLoader(dataset=dataset,
                                           batch_sampler=batch_sampler,
                                           batchify_fn=batchify_fn)
        return data_loader

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

    def get_optimizer_params(self, optimizer, lr):
        if optimizer == 'bertadam':
            return {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.01}
        else:
            raise ValueError

    def train(self, args, model, train_dataset, dev_dataset, ctx, data_augment_by_epoch):
        task = self.task
        loss_function = self.loss_function
        metric = task.get_metric()
        num_train_examples = len(train_dataset)

        self.initialize_model(args, model, ctx)

        model.hybridize(static_alloc=True)
        loss_function.hybridize(static_alloc=True)

        lr = args.lr
        optimizer_params = self.get_optimizer_params(args.optimizer, args.lr)
        trainer = gluon.Trainer(
            model.collect_params(),
            args.optimizer,
            optimizer_params,
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

        logger.info('building data loader')
        train_data = self.build_data_loader(train_dataset, args.batch_size, args.max_len, test=False, example_augment_prob=args.example_augment_prob, word_mask=args.word_mask, word_dropout=args.word_dropout, word_dropout_region=args.word_dropout_region, ctx=ctx)
        dev_data = self.build_data_loader(dev_dataset, args.batch_size, args.max_len, test=True, ctx=ctx)

        logger.info('start training')
        for epoch_id in range(args.epochs):
            metric.reset()
            step_loss = 0
            tic = time.time()

            if data_augment_by_epoch and epoch_id > 0:
                train_data = self.build_data_loader(train_dataset, args.batch_size, args.max_len, test=False, example_augment_prob=args.example_augment_prob, word_mask=args.word_mask, word_dropout=args.word_dropout, word_dropout_region=args.word_dropout_region, ctx=ctx)

            for batch_id, seqs in enumerate(train_data):
                step_num += 1
                # learning rate schedule
                if args.warmup_ratio < 0:
                    new_lr = lr
                else:
                    if step_num < num_warmup_steps:
                        new_lr = lr * step_num / num_warmup_steps
                    else:
                        offset = (step_num - num_warmup_steps) * lr / (
                            num_train_steps - num_warmup_steps)
                        new_lr = lr - offset
                trainer.set_learning_rate(new_lr)
                # forward and backward
                with mx.autograd.record():
                    id_, inputs, label = self.prepare_data(seqs, ctx)
                    out = model(*inputs)
                    ls = loss_function(out, label).mean()
                ls.backward()
                # update
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(1)
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
                        trainer.learning_rate, *metric_val))
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

            metric_names = sorted(dev_metrics.keys())
            logger.info('[Epoch {}] val_metrics={}'.format(
                        epoch_id, metric_dict_to_str(dev_metrics)))

            # Save checkpoint of last epoch
            checkpoint_path = os.path.join(checkpoints_dir, 'last.params')
            model.save_parameters(checkpoint_path)

            toc = time.time()
            logger.info('Time cost={:.1f}s'.format(toc - tic))
            tic = toc

            if args.early_stop and terminate_training:
                logger.info('early stopping')
                break


class BERTNLIRunner(NLIRunner):
    def build_model(self, args, model_args, ctx, dataset=None, vocab=None):
        dataset = model_args.model_name
        if model_args.model_type == 'bert':
            model_name = 'bert_12_768_12'
        elif model_args.model_type == 'bertl':
            model_name = 'bert_24_1024_16'
        elif model_args.model_type == 'roberta':
            model_name = 'roberta_12_768_12'
        elif model_args.model_type == 'robertal':
            model_name = 'roberta_24_1024_16'
        else:
            raise NotImplementedError
        self.is_roberta = model_args.model_type.startswith('roberta')

        if args.model_params is None:
            pretrained = True
        else:
            pretrained = False
        bert, vocabulary = nlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=pretrained,
            ctx=ctx,
            use_pooler=False if self.is_roberta else True,
            use_decoder=False,
            use_classifier=False)
        if args.model_params:
            bert.load_parameters(args.model_params, ctx=ctx, cast_dtype=True, ignore_extra=True)
        if args.fix_bert_weights:
            bert.collect_params('.*weight|.*bias').setattr('grad_req', 'null')

        if vocab:
            vocabulary = vocab
        do_lower_case = 'uncased' in dataset
        task_name = args.task_name
        num_classes = self.task.num_classes()
        if self.is_roberta:
            model = RoBERTaClassifier(bert, dropout=0.0, num_classes=num_classes)
            self.tokenizer = nlp.data.GPT2BPETokenizer()
        else:
            model = BERTClassifier(bert, num_classes=num_classes, dropout=model_args.dropout)
            self.tokenizer = BERTTokenizer(vocabulary, lower=do_lower_case)

        return model, vocabulary

    def build_model_transformer(self, max_len, example_augment_prob=0, word_mask_prob=0):
        if word_mask_prob > 0:
            trans = MaskedBERTDatasetTransform(
                self.tokenizer, max_len, vocab=self.vocab, class_labels=self.labels, pad=False, pair=True,
                example_augment_prob=example_augment_prob, word_mask_prob=word_mask_prob)
        else:
            trans = BERTDatasetTransform(
                self.tokenizer, max_len, vocab=self.vocab, class_labels=self.labels, pad=False, pair=True)
        return trans

    def prepare_data(self, data, ctx):
        """Batched data to model inputs.
        """
        id_, input_ids, valid_len, type_ids, label = data
        if self.is_roberta:
            inputs = (input_ids.as_in_context(ctx),
                      valid_len.astype('float32').as_in_context(ctx))
        else:
            inputs = (input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                      valid_len.astype('float32').as_in_context(ctx))
        label = label.as_in_context(ctx)
        return id_, inputs, label

    def initialize_model(self, args, model, ctx):
        initializer = mx.init.Normal(0.02)
        # Only init the last layer
        model.classifier.initialize(init=initializer, ctx=ctx, force_reinit=False)


class HypothesisNLIRunner(BERTNLIRunner):
    def __init__(self, task, runs_dir, run_id=None, feature='hypothesis'):
        super().__init__(task, runs_dir, run_id)
        self.feature = feature

    def build_model_transformer(self, max_len, example_augment_prob=0, word_mask_prob=0):
        trans = NLIHypothesisTransform(
            self.tokenizer, self.labels, max_len, pad=False)
        return trans

class CBOWNLIRunner(NLIRunner):
    def __init__(self, task, runs_dir, run_id=None):
        super().__init__(task, runs_dir, run_id)
        self.tokenizer = BasicTokenizer(do_lower_case=True)

    def build_model(self, args, model_args, ctx, dataset=None, vocab=None):
        if vocab is None:
            vocab = self.build_vocab(dataset)
        num_classes = self.task.num_classes()

        model = NLICBOWClassifier(len(vocab), num_classes, model_args.embedding_size, model_args.hidden_size, model_args.num_layers, dropout=model_args.dropout)
        return model, vocab

    def build_model_transformer(self, max_len, example_augment_prob=0, word_mask_prob=0):
        trans = CBOWTransform(self.labels, self.tokenizer, self.vocab, num_input_sentences=2)
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
        model.initialize(init=mx.init.Normal(0.01), ctx=ctx, force_reinit=False)
        # Initialize word embeddings
        if args.embedding_source:
            glove = nlp.embedding.create('glove', source=args.embedding_source)
            self.vocab.set_embedding(glove)
            unk_idx = self.vocab[self.vocab.unknown_token]
            pad_idx = self.vocab[self.vocab.padding_token]
            self.vocab.embedding.idx_to_vec[unk_idx] = np.random.normal(size=args.embedding_size)
            self.vocab.embedding.idx_to_vec[pad_idx] = 0.
            model.embedding.weight.set_data(self.vocab.embedding.idx_to_vec)
            if args.fix_word_embedding:
                model.embedding.weight.grad_req = 'null'


class HandcraftedNLIRunner(CBOWNLIRunner):
    def build_model(self, args, model_args, ctx, dataset=None, vocab=None):
        # empty overlap / non-overlap tokens
        reserved_tokens = ['<empty>']
        if vocab is None:
            vocab = self.build_vocab(dataset, reserved_tokens=reserved_tokens)
        num_classes = self.task.num_classes()
        model = NLIHandcraftedClassifier(len(vocab), num_classes, model_args.embedding_size, model_args.hidden_size, model_args.num_layers, dropout=model_args.dropout)
        return model, vocab

    def build_model_transformer(self, max_len, example_augment_prob=0, word_mask_prob=0):
        trans = NLIHandcraftedTransform(self.labels, self.tokenizer, self.vocab)
        return trans

    def prepare_data(self, data, ctx):
        """Batched data to model inputs.
        """
        id_, dense_features, overlap_token_ids, non_overlap_token_ids, label = data
        inputs = (dense_features.astype('float32').as_in_context(ctx),
                  overlap_token_ids.as_in_context(ctx),
                  non_overlap_token_ids.as_in_context(ctx),
                 )
        label = label.as_in_context(ctx)
        return id_, inputs, label


class DANLIRunner(CBOWNLIRunner):
    def build_model(self, args, model_args, ctx, dataset=None, vocab=None):
        reserved_tokens = ['NULL']
        if vocab is None:
            vocab = self.build_vocab(dataset, reserved_tokens=reserved_tokens)
        num_classes = self.task.num_classes()

        model = DecomposableAttentionClassifier(len(vocab), num_classes, model_args.embedding_size, model_args.hidden_size, dropout=model_args.dropout)
        return model, vocab

    def get_optimizer_params(self, optimizer, lr):
        if optimizer == 'bertadam':
            return {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.01}
        elif optimizer == 'adagrad':
            return {'learning_rate': lr, 'wd': 1e-5, 'clip_gradient': 5}
        elif optimizer == 'adam':
            return {'learning_rate': lr}
        else:
            raise ValueError

    def build_model_transformer(self, max_len, example_augment_prob=0, word_mask_prob=0):
        trans = DATransform(self.labels, self.tokenizer, self.vocab)
        return trans

    def prepare_data(self, data, ctx):
        """Batched data to model inputs.
        """
        id_, input_ids, valid_len, label = data
        inputs = (input_ids[0].as_in_context(ctx),
                  input_ids[1].as_in_context(ctx),
                  valid_len[0].astype('float32').as_in_context(ctx),
                  valid_len[1].astype('float32').as_in_context(ctx),
                  )
        label = label.as_in_context(ctx)
        return id_, inputs, label


class ESIMNLIRunner(DANLIRunner):
    def build_model(self, args, model_args, ctx, dataset=None, vocab=None):
        reserved_tokens = None
        if vocab is None:
            vocab = self.build_vocab(dataset, reserved_tokens=reserved_tokens)
        num_classes = self.task.num_classes()

        model = ESIMClassifier(len(vocab), num_classes, model_args.embedding_size, model_args.hidden_size, model_args.hidden_size, dropout=model_args.dropout)
        return model, vocab

    def build_model_transformer(self, max_len, example_augment_prob=0, word_mask_prob=0):
        trans = ESIMTransform(self.labels, self.tokenizer, self.vocab, max_len)
        return trans

    def prepare_data(self, data, ctx):
        """Batched data to model inputs.
        """
        id_, input_ids, valid_len, label = data
        inputs = (input_ids[0].as_in_context(ctx),
                  input_ids[1].as_in_context(ctx),
                  #valid_len[0].astype('float32').as_in_context(ctx),
                  #valid_len[1].astype('float32').as_in_context(ctx),
                  )
        label = label.as_in_context(ctx)
        return id_, inputs, label


def get_additive_runner(base, project=False, remove=False):
    class AdditiveNLIRunner(base):
        """Additive model of a superficial classifier and a normal classifier.
        """
        def __init__(self, task, runs_dir, prev_runners, prev_args, run_id=None):
            super().__init__(task, runs_dir, run_id)
            # Runner for the previous model
            self.prev_runners = prev_runners
            self.prev_args = prev_args

        def build_model(self, args, model_args, ctx, dataset=None, vocab=None):
            model, vocabulary = super().build_model(args, model_args, ctx, dataset=dataset, vocab=vocab)
            model = AdditiveClassifier(model, mode=args.additive_mode)
            return model, vocabulary

        def run_prev_model(self, dataset, runner, args, ctx):
            logger.info('running previous model on preprocessed dataset')
            _, prev_scores, ids = runner.run_test(args, ctx, dataset)
            assert len(dataset) == len(prev_scores)

            # Reorder scores by example id
            prev_scores = {id_.asscalar(): prev_scores[i] for i, id_ in enumerate(ids)}
            reordered_prev_scores = []
            for data in dataset:
                id_ = data[0]
                reordered_prev_scores.append(prev_scores[id_])
            prev_scores = mx.nd.stack(*reordered_prev_scores, axis=0)
            prev_scores = prev_scores.asnumpy()

            return prev_scores

        def preprocess_dataset(self, split, cheat_rate, remove_cheat, remove_correct, max_num_examples, ctx=None):
            """Add scores from previous classifiers.
            """
            dataset = super().preprocess_dataset(split, cheat_rate, remove_cheat, remove_correct, max_num_examples)

            prev_scores = 0.
            for _prev_runner, _prev_args in zip(self.prev_runners, self.prev_args):
                _prev_scores = self.run_prev_model(dataset, _prev_runner, _prev_args, ctx)
                prev_scores += _prev_scores

            return gluon.data.ArrayDataset(prev_scores, dataset)

        def get_input(self, example):
            """Convert an example in the preprocessed dataset to a list of values.
            """
            scores, example = example
            # id_, premise, hypothesis, label
            return example

        def build_dataset(self, data, max_len, example_augment_prob=0, word_mask=0, word_dropout=0, word_dropout_region=None, ctx=None):
            trans_list = self.build_data_transformer(max_len, example_augment_prob, word_mask, word_dropout, word_dropout_region)
            prev_scores = [x[0] for x in data]
            dataset = gluon.data.SimpleDataset([x[1] for x in data])
            logger.info('processing {} examples'.format(len(dataset)))
            start = time.time()
            for trans in trans_list:
                dataset = dataset.transform(trans, lazy=False)
            logger.info('elapsed time: {:.2f}s'.format(time.time() - start))
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

    class RemoveNLIRunner(AdditiveNLIRunner):
        def preprocess_dataset(self, split, cheat_rate, remove_cheat, remove_correct, max_num_examples, ctx=None):
            """Remove examples that are classified correctly by previous models.
            """
            dataset = super().preprocess_dataset(split, cheat_rate, remove_cheat, remove_correct, max_num_examples, ctx)
            if remove_correct:
                _dataset = []
                # TODO: move label_map to runner
                _label_map = {}
                for (i, label) in enumerate(self.labels):
                    _label_map[label] = i
                for d in dataset:
                    score = d[0]
                    # d[1] = id_, premise, hypothesis, label
                    label = d[1][-1]
                    if np.argmax(score) != _label_map[label]:
                        _dataset.append(d)
                logger.info('after remove correct examples: {}'.format(len(_dataset)))
                return gluon.data.ArrayDataset([d[0] for d in _dataset], [d[1] for d in _dataset])
            else:
                return dataset

        def build_model(self, args, model_args, ctx, dataset=None, vocab=None):
            # Just use the base model
            if args.additive_mode != 'last':
                raise ValueError('Remove method only uses the base model.')
            return super().build_model(args, model_args, ctx, dataset=dataset, vocab=vocab)

        def evaluate(self, data_loader, model, metric, ctx):
            # Don't need to eval all modes
            return super(AdditiveNLIRunner, self).evaluate(data_loader, model, metric, ctx)

    class ProjectNLIRunner(AdditiveNLIRunner):
        def build_model(self, args, model_args, ctx, dataset=None, vocab=None):
            model, vocabulary = super(AdditiveNLIRunner, self).build_model(args, model_args, ctx, dataset=dataset, vocab=vocab)
            model = ProjectClassifier(model)
            return model, vocabulary

        def evaluate(self, data_loader, model, metric, ctx):
            return super(AdditiveNLIRunner, self).evaluate(data_loader, model, metric, ctx)

    if project:
        return ProjectNLIRunner
    elif remove:
        return RemoveNLIRunner
    return AdditiveNLIRunner
