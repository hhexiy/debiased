"""Dataset transforms."""

import random
import logging
import numpy as np

import mxnet
from mxnet.gluon.data import Sampler
from mxnet.gluon.data import RandomSampler
import gluonnlp as nlp
from gluonnlp.data import BERTSentenceTransform

from .tokenizer import BasicTokenizer

logger = logging.getLogger('nli')

class ParaOverlapSampler(Sampler):
    """Sort non-paraphrase examples by the jaccard similarity in ascending order.
    Remove low similarity ones.
    """
    def __init__(self, dataset, num_remove):
        self.dataset = dataset
        if num_remove < 1.:
            num_remove = int(len(self.dataset) * num_remove)
        if num_remove > len(self.dataset):
            logger.warning('asked to remove more examples than the dataset size ({} clipped to {}).'.format(num_remove, len(self.dataset)))
            num_remove = len(self.dataset)
        self.tokenizer = BasicTokenizer(do_lower_case=True)
        np_indices = [i for i, e in enumerate(self.dataset) if e[-1] != '1']
        logger.info('sorting non-paraphrase examples by jaccard similarity between two sentences')
        np_overlap = [self.compute_overlap(self.dataset[i]) for i in np_indices]
        logger.info('average NE overlap: {:.4f}'.format(np.mean(np_overlap)))
        buckets = [0, 0.2, 0.4, 0.6, 0.8, 1]
        logger.info('histograms of similarity: {}'.format(np.histogram(np_overlap, bins=buckets)))

        sorted_np_indices = [np_indices[i] for i in np.argsort(np_overlap).tolist()]
        remove_indices = set(sorted_np_indices[:num_remove])
        self.indices = [i for i in range(len(dataset)) if not i in remove_indices]

    def compute_overlap(self, example):
        id_, s1, s2, label = example
        s1_tokens = set(self.tokenizer.tokenize(s1))
        s2_tokens = set(self.tokenizer.tokenize(s2))
        overlap = len(s1_tokens.intersection(s2_tokens)) / len(s1_tokens.union(s2_tokens))
        return overlap

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class NLIOverlapSampler(Sampler):
    """Sort non-entailment examples by the amount of overlap in descending order.
    Remove highly overlapping ones.
    """
    def __init__(self, dataset, num_remove):
        self.dataset = dataset
        if num_remove < 1.:
            num_remove = int(len(self.dataset) * num_remove)
        if num_remove > len(self.dataset):
            logger.warning('asked to remove more examples than the dataset size ({} clipped to {}).'.format(num_remove, len(self.dataset)))
            num_remove = len(self.dataset)
        self.tokenizer = BasicTokenizer(do_lower_case=True)
        ne_indices = [i for i, e in enumerate(self.dataset) if e[-1] != 'entailment']
        logger.info('sorting non-entailment examples by amount of overlap between hypo and prem')
        ne_overlap = [self.compute_overlap(self.dataset[i]) for i in ne_indices]
        logger.info('average NE overlap: {:.4f}'.format(np.mean(ne_overlap)))
        buckets = [0, 0.2, 0.4, 0.6, 0.8, 1]
        logger.info('histograms of NE overlap: {}'.format(np.histogram(ne_overlap, bins=buckets)))

        sorted_ne_indices = [ne_indices[i] for i in np.argsort(ne_overlap).tolist()[::-1]]
        remove_indices = set(sorted_ne_indices[:num_remove])
        self.indices = [i for i in range(len(dataset)) if not i in remove_indices]

    def compute_overlap(self, example):
        id_, premise, hypothesis, label = example
        premise_tokens = self.tokenizer.tokenize(premise)
        hypothesis_tokens = self.tokenizer.tokenize(hypothesis)
        overlap = len([w for w in hypothesis_tokens if w in premise_tokens]) / float(len(hypothesis_tokens))
        return overlap

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class BERTDatasetTransform(object):
    """Dataset transformation for BERT-style sentence classification or regression.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    vocab : Vocab or BERTVocab
        The vocabulary.
    labels : list of int , float or None. defaults None
        List of all label ids for the classification task and regressing task.
        If labels is None, the default task is regression
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    label_dtype: int32 or float32, default float32
        label_dtype = int32 for classification task
        label_dtype = float32 for regression task
    """

    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 vocab=None,
                 class_labels=None,
                 label_alias=None,
                 pad=True,
                 pair=True,
                 has_label=True):
        self.class_labels = class_labels
        self.has_label = has_label
        self._label_dtype = 'int32' if class_labels else 'float32'
        if has_label and class_labels:
            self._label_map = {}
            for (i, label) in enumerate(class_labels):
                self._label_map[label] = i
            if label_alias:
                for key in label_alias:
                    self._label_map[key] = self._label_map[label_alias[key]]
        self._bert_xform = BERTSentenceTransform(
            tokenizer, max_seq_length, vocab=vocab, pad=pad, pair=pair)

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 3 strings:
        text_a, text_b and label.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
            label: '0'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14
            label: 0

        For single sequences, the input is a tuple of 2 strings: text_a and label.
        Inputs:
            text_a: 'the dog is hairy .'
            label: '1'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7
            label: 1

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b, label). For single sequences, the input is a tuple
            of 2 strings: (text_a, label).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        np.array: classification task: label id in 'int32', shape (batch_size, 1),
            regression task: label in 'float32', shape (batch_size, 1)
        """
        id_ = line[0]
        line = line[1:]
        if self.has_label:
            input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])
            label = line[-1]
            # map to int if class labels are available
            if self.class_labels:
                label = self._label_map[label]
            label = np.array([label], dtype=self._label_dtype)
        else:
            input_ids, valid_length, segment_ids = self._bert_xform(line)
        return id_, input_ids, valid_length, segment_ids, label

    def get_length(self, *data):
        return data[2]

    def get_batcher(self):
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack())
        return batchify_fn


class MaskedBERTDatasetTransform(BERTDatasetTransform):
    """BERTDatasetTransform where words are replaced with [MASK] with
    a certain probability.
    """
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 vocab=None,
                 class_labels=None,
                 label_alias=None,
                 pad=True,
                 pair=True,
                 has_label=True,
                 example_augment_prob=0.,
                 word_mask_prob=0.):
        super().__init__(tokenizer,
                 max_seq_length,
                 vocab,
                 class_labels,
                 label_alias,
                 pad,
                 pair,
                 has_label
                )
        self.example_augment_prob = example_augment_prob
        self.word_mask_prob = word_mask_prob
        self.mask_id = vocab[vocab.mask_token]
        self.cls_id = vocab[vocab.cls_token]
        self.sep_id = vocab[vocab.sep_token]
        self.vocab = vocab

    def mask(self, seq):
        if self.word_mask_prob > 0:
            seq = seq.tolist()
            mask = np.random.binomial(n=1, p=self.word_mask_prob, size=len(seq)).tolist()
            seq = [s if m == 0 else
                    (self.mask_id if not s in (self.cls_id, self.sep_id)
                     else s)
                    for m, s in zip(mask, seq)]
            seq = np.array(seq)
        return seq

    def __call__(self, line):
        id_, input_ids, valid_length, segment_ids, label = super().__call__(line)
        if np.random.uniform() < self.example_augment_prob:
            input_ids = self.mask(input_ids)
        return id_, input_ids, valid_length, segment_ids, label


class CBOWTransform(object):
    """Dataset Transformation for CBOW Classification.
    """
    def __init__(self, labels, tokenizer, vocab, num_input_sentences=2):
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
        self._vocab = vocab
        self._tokenizer = tokenizer
        self.num_input_sentences = num_input_sentences

    def __call__(self, line):
        id_ = line[0]
        inputs = line[1:-1]  # list of text strings
        label = line[-1]
        label = convert_to_unicode(label)
        label_id = self._label_map[label]
        label_id = np.array([label_id], dtype='int32')
        input_ids = [self._vocab(self._tokenizer.tokenize(s)) for s in inputs]
        valid_lengths = [len(s) for s in input_ids]
        return id_, input_ids, valid_lengths, label_id

    def get_length(self, *data):
        return max(data[2])

    def get_batcher(self):
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Stack(),
            nlp.data.batchify.Tuple(*[nlp.data.batchify.Pad(axis=0, pad_val=self._vocab[self._vocab.padding_token]) for _ in range(self.num_input_sentences)]),
            nlp.data.batchify.Tuple(*[nlp.data.batchify.Stack() for _ in range(self.num_input_sentences)]),
            nlp.data.batchify.Stack())
        return batchify_fn


class DATransform(CBOWTransform):
    """Dataset Transformation for Decomposable Attention.
    """
    def __call__(self, line):
        id_ = line[0]
        inputs = line[1:-1]  # list of text strings
        label = line[-1]
        label = convert_to_unicode(label)
        label_id = self._label_map[label]
        label_id = np.array([label_id], dtype='int32')
        input_ids = [self._vocab(['NULL'] + self._tokenizer.tokenize(s)) for s in inputs]
        valid_lengths = [len(s) for s in input_ids]
        return id_, input_ids, valid_lengths, label_id


class ESIMTransform(CBOWTransform):
    def __init__(self, labels, tokenizer, vocab, max_len=60, num_input_sentences=2):
        super().__init__(labels, tokenizer, vocab, num_input_sentences)
        self.max_seq_length = max_len

    def __call__(self, line):
        id_ = line[0]
        inputs = line[1:-1]  # list of text strings
        label = line[-1]
        label = convert_to_unicode(label)
        label_id = self._label_map[label]
        label_id = np.array([label_id], dtype='int32')
        input_ids = [self._vocab(self._tokenizer.tokenize(s)[:self.max_seq_length]) for s in inputs]
        valid_lengths = [len(s) for s in input_ids]
        return id_, input_ids, valid_lengths, label_id


# SNLI -> NLI
class SNLICheatTransform(object):
    def __init__(self, labels, rate=1., remove=False):
        self.rate = rate
        self.labels = labels
        self.rng = random.Random(42)
        self.remove = remove

    def __call__(self, line):
        id_, premise, hypothesis, label = line[0], line[1], line[2], line[3]
        if self.rng.random() < self.rate:
            label = label
            if self.remove:
                return None
        else:
            label = self.rng.choice(self.labels)
        line[2] = '{} and {}'.format(label, hypothesis)
        return line

    def reset(self):
        self.rng.seed(42)


# TODO: SNLI -> NLI
class SNLIWordDropTransform(object):
    def __init__(self, example_augment_prob=0., rate=0., region=('premise', 'hypothesis'), tokenizer=str.split):
        self.example_augment_prob = example_augment_prob
        self.rate = rate
        self.region = region
        self.tokenizer = tokenizer

    def dropout(self, seq):
        mask = np.random.binomial(n=1, p=1-self.rate, size=len(seq))
        seq = [s for m, s in zip(mask, seq) if m == 1]
        return seq

    def __call__(self, line):
        if np.random.uniform() < self.example_augment_prob:
            idx, premise, hypothesis, label = line[0], line[1], line[2], line[3]
            if 'premise' in self.region:
                premise = ' '.join(self.dropout(self.tokenizer(premise)))
            if 'hypothesis' in self.region:
                hypothesis = ' '.join(self.dropout(self.tokenizer(hypothesis)))
            line = [idx, premise, hypothesis, label]
        return line


class NLIHandcraftedTransform(object):
    """Dataset Transformation for CBOW Classification.
    """
    def __init__(self, labels, tokenizer, vocab):
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
        self._vocab = vocab
        self._tokenizer = tokenizer

    def __call__(self, line):
        id_ = line[0]
        inputs = line[1:-1]  # list of text strings
        label = line[-1]
        label = convert_to_unicode(label)
        label_id = self._label_map[label]
        label_id = np.array([label_id], dtype='int32')

        assert len(inputs) == 2
        prem, hypo = [self._tokenizer.tokenize(s) for s in inputs]
        len_diff = abs(len(hypo) - len(prem)) / (len(hypo) + len(prem))
        negation = 1 if ('not' in hypo or "n't" in hypo) else 0
        jaccard_sim = len(set(prem).intersection(set(hypo))) / float(len(set(prem).union(set(hypo))))
        overlap_tokens = [w for w in hypo if w in prem]
        non_overlap_tokens = [w for w in hypo if not w in prem]
        if not overlap_tokens:
            overlap_tokens = ['<empty>']
        if not non_overlap_tokens:
            non_overlap_tokens = ['<empty>']
        overlap_token_ids = [self._vocab(w) for w in overlap_tokens]
        non_overlap_token_ids = [self._vocab(w) for w in non_overlap_tokens]
        dense_features = [len_diff, negation, jaccard_sim]

        return id_, dense_features, overlap_token_ids, non_overlap_token_ids, label_id

    def get_length(self, *data):
        return len(data[1]) + len(data[2]) + len(data[3])

    def get_batcher(self):
        pad_val = self._vocab[self._vocab.padding_token]
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Stack(), nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0, pad_val=pad_val),
            nlp.data.batchify.Pad(axis=0, pad_val=pad_val),
            nlp.data.batchify.Stack())
        return batchify_fn


class NLIHypothesisTransform(BERTDatasetTransform):
    def __init__(self, tokenizer, labels, max_seq_length, pad=True):
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
        self._bert_xform = BERTSentenceTransform(
            tokenizer, max_seq_length, pad=pad, pair=False)

    def __call__(self, line):
        id_ = line[0]
        line = line[1:]
        # Ignore premise (sentence 0)
        line = line[1:]
        label = line[-1]
        label = convert_to_unicode(label)
        label_id = self._label_map[label]
        label_id = np.array([label_id], dtype='int32')
        input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])
        return id_, input_ids, valid_length, segment_ids, label_id

    def get_length(self, *data):
        return data[2]

    def get_batcher(self):
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack())
        return batchify_fn
