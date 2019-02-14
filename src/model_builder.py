import os
import logging
import argparse
import json

import gluonnlp as nlp
from gluonnlp.model import bert_12_768_12

from .bert import BERTClassifier
from .additive import AdditiveClassifier
from .task import tasks
from .tokenizer import FullTokenizer

logger = logging.getLogger('nli')


def build_bert_model(args, ctx):
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
    return model, vocabulary

def load_model(args, model_args, path, ctx):
    model, _, tokenizer = build_model(model_args, ctx)
    params_file = 'last.params' #if args.use_last else 'valid_best.params'
    #if path == 'output/snli-bert-cheat':
    if True:
        print('load params from', os.path.join(
            path, 'checkpoints', params_file))
        model.load_parameters(os.path.join(
            path, 'checkpoints', params_file), ctx=ctx)
    else:
        print('pass')
    vocab = nlp.Vocab.from_json(
        open(os.path.join(path, 'vocab.jsons')).read())
    return model, vocab, tokenizer

def build_model(args, ctx):
    model, vocabulary = build_bert_model(args, ctx)
    if 'additive' in vars(args) and args.additive:
        path = args.additive
        #path = 'output/snli-bert-cheat'
        model_args = argparse.Namespace(**json.load(
            open(os.path.join(path, 'config.json'))))
        a_model, _, _ = load_model(args, model_args, path, ctx)
        model = AdditiveClassifier(classifiers=[a_model, model],
                                            active=[True, True],
                                            no_grad=[True, False],
                                            names=['cheat', 'normal'])
        #model = AdditiveClassifier(classifiers=[a_model],
        #                                    active=[True],
        #                                    no_grad=[False],
        #                                    names=['cheat'])

    #do_lower_case = 'uncased' in dataset
    do_lower_case = True
    tokenizer = FullTokenizer(vocabulary, do_lower_case=do_lower_case)

    logger.debug(model)
    return model, vocabulary, tokenizer

