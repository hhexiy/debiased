import os
import logging
import argparse
import json
import itertools
import re

import mxnet as mx
import gluonnlp as nlp
from gluonnlp.model import bert_12_768_12

from .model.bert import BERTClassifier
from .model.additive import AdditiveClassifier
from .model.cbow import NLICBOWClassifier, NLIHandcraftedClassifier
from .model.decomposable_attention import DecomposableAttentionClassifier
from .task import tasks
from .tokenizer import FullTokenizer, BasicTokenizer
from .utils import read_args

logger = logging.getLogger('nli')

class VocabBuilder(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def preprocess(self, example):
        """Return strings to be tokenized.

        Parameters
        ----------
        example : tuple, (id_, input_0, ..., input_n, label)
        """
        return example[1:-1]

    def build_vocab(self, dataset, reserved_tokens=None):
        # Each example is a sequence to tokens
        sentences = itertools.chain.from_iterable([self.preprocess(ex) for ex in dataset])
        tokens = [self.tokenizer.tokenize(s) for s in sentences]
        counter = nlp.data.count_tokens(list(itertools.chain.from_iterable(tokens)))
        vocab = nlp.Vocab(counter, bos_token=None, eos_token=None, reserved_tokens=reserved_tokens)
        logger.info('built vocabulary of size {}'.format(len(vocab)))
        return vocab


def build_cbow_model(args, ctx, dataset, tokenizer, vocab=None):
    if args.superficial == 'handcrafted':
        # empty overlap / non-overlap tokens
        reserved_tokens = ['<empty>']
    else:
        reserved_tokens = None
    if vocab is None:
        vocab = VocabBuilder(tokenizer).build_vocab(dataset, reserved_tokens=reserved_tokens)
    task_name = args.task_name
    num_classes = len(tasks[task_name].get_labels())

    if args.superficial == 'handcrafted':
        model = NLIHandcraftedClassifier(len(vocab), num_classes, args.embedding_size, args.hidden_size, args.num_layers, dropout=args.dropout)
    else:
        model = NLICBOWClassifier(len(vocab), num_classes, args.embedding_size, args.hidden_size, args.num_layers, dropout=args.dropout)
    return model, vocab, tokenizer


def build_da_model(args, ctx, dataset, tokenizer, vocab=None):
    reserved_tokens = ['NULL']
    if vocab is None:
        vocab = VocabBuilder(tokenizer).build_vocab(dataset, reserved_tokens=reserved_tokens)
    task_name = args.task_name
    num_classes = len(tasks[task_name].get_labels())

    model = DecomposableAttentionClassifier(len(vocab), num_classes, args.embedding_size, args.hidden_size, dropout=args.dropout)
    return model, vocab, tokenizer


def build_bert_model(args, ctx, vocab=None):
    dataset = 'book_corpus_wiki_en_uncased'
    bert, vocabulary = bert_12_768_12(
        dataset_name=dataset,
        pretrained=True,
        ctx=ctx,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False)
    if vocab:
        vocabulary = vocab
    task_name = args.task_name
    num_classes = len(tasks[task_name].get_labels())
    model = BERTClassifier(bert, num_classes=num_classes, dropout=args.dropout)
    do_lower_case = 'uncased' in dataset
    tokenizer = FullTokenizer(vocabulary, do_lower_case=do_lower_case)
    return model, vocabulary, tokenizer

def load_model(args, model_args, path, ctx, tokenizer=None):
    vocab = nlp.Vocab.from_json(
        open(os.path.join(path, 'vocab.jsons')).read())
    model, _, tokenizer = build_model(args, model_args, ctx, vocab=vocab, tokenizer=tokenizer)
    params_file = 'last.params' if args.use_last else 'valid_best.params'
    logger.info('load model from {}'.format(os.path.join(
        path, 'checkpoints', params_file)))
    model.load_parameters(os.path.join(
        path, 'checkpoints', params_file), ctx=ctx)
    return model, vocab, tokenizer

def build_model(args, model_args, ctx, dataset=None, vocab=None, tokenizer=None):
    if model_args.model_type == 'cbow':
        model, vocabulary, tokenizer = build_cbow_model(model_args, ctx, dataset, tokenizer, vocab=vocab)
    elif model_args.model_type == 'bert':
        model, vocabulary, tokenizer = build_bert_model(model_args, ctx, vocab=vocab)
    elif model_args.model_type == 'da':
        model, vocabulary, tokenizer = build_da_model(model_args, ctx, dataset, tokenizer, vocab=vocab)
    else:
        raise ValueError

    if model_args.additive:
        model = AdditiveClassifier(model, mode=args.additive_mode)

    logger.debug(model)
    return model, vocabulary, tokenizer
