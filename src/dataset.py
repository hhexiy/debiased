# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and DMLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT datasets."""

__all__ = [
    'MRPCDataset', 'QQPDataset', 'BERTTransform', 'QNLIDataset', 'RTEDataset',
    'STSBDataset', 'COLADataset', 'MNLIDataset', 'WNLIDataset', 'SSTDataset',
    'ClassificationTransform'
]

import os
import numpy as np
import glob
import random
import re
import logging
from mxnet.metric import Accuracy, F1, MCC, PearsonCorrelation, CompositeEvalMetric
try:
    from tokenizer import convert_to_unicode
except ImportError:
    from .tokenizer import convert_to_unicode
from gluonnlp.data import TSVDataset
from gluonnlp.data.registry import register
import gluonnlp as nlp

logger = logging.getLogger('nli')

@register(segment=['train', 'dev', 'test'])
class MRPCDataset(TSVDataset):
    """The Microsoft Research Paraphrase Corpus dataset.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/MRPC'
        Path to the folder which stores the MRPC dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'MRPC')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        A_IDX, B_IDX, LABEL_IDX = 3, 4, 0
        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(MRPCDataset, self).__init__(
            path, num_discard_samples=1, field_indices=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']

    @staticmethod
    def get_metric():
        """Get metrics Accuracy and F1"""
        metric = CompositeEvalMetric()
        for child_metric in [Accuracy(), F1()]:
            metric.add(child_metric)
        return metric


class GLUEDataset(TSVDataset):
    """GLUEDataset class"""

    def __init__(self, path, num_discard_samples, fields, label_field=None, max_num_examples=-1):
        self.fields = fields
        self.max_num_examples = max_num_examples
        self.label_field = label_field
        super(GLUEDataset, self).__init__(
            path, num_discard_samples=num_discard_samples)

    def _read(self):
        all_samples = super(GLUEDataset, self)._read()
        logger.info('read {} examples'.format(len(all_samples)))
        largest_field = max(self.fields)
        # filter out error records
        final_samples = [[id_] + [s[f] for f in self.fields] for id_, s in enumerate(all_samples)
                         if len(s) >= largest_field + 1]
        logger.info('{} examples after filtering by number of fields'.format(len(final_samples)))
        # filter wrong labels
        label_field = self.label_field + 1  # we inserted id_ before fields
        if self.label_field is not None:
            final_samples = [s for s in final_samples if s[label_field] in self.get_labels()]
            logger.info('{} examples after filtering by valid labels'.format(len(final_samples)))
        if self.max_num_examples > 0:
            return final_samples[:self.max_num_examples]
        return final_samples


@register(segment=['train', 'dev', 'test'])
class QQPDataset(GLUEDataset):
    """Dataset for Quora Question Pairs.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/QQP'
        Path to the folder which stores the QQP dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'), 'QQP')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 3, 4, 5
            fields = [A_IDX, B_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX, B_IDX = 1, 2
            fields = [A_IDX, B_IDX]
        super(QQPDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @classmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']

    @classmethod
    def get_metric():
        """Get metrics Accuracy and F1"""
        metric = CompositeEvalMetric()
        for child_metric in [Accuracy(), F1()]:
            metric.add(child_metric)
        return metric


class QQPWangDataset(QQPDataset):
    """QQP dataset split by Wang et al., IJCAI 2017.
    https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing
    """
    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'), 'QQP-wang')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev', 'test']:
            A_IDX, B_IDX, LABEL_IDX = 3, 4, 5
            fields = [A_IDX, B_IDX, LABEL_IDX]
        super(QQPDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)


@register(segment=['train', 'dev', 'test'])
class RTEDataset(GLUEDataset):
    """Task class for Recognizing Textual Entailment

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/RTE'
        Path to the folder which stores the RTE dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'), 'RTE')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 1, 2, 3
            fields = [A_IDX, B_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX, B_IDX = 1, 2
            fields = [A_IDX, B_IDX]
        super(RTEDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['not_entailment', 'entailment']

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        return Accuracy()


@register(segment=['train', 'dev', 'test'])
class QNLIDataset(GLUEDataset):
    """Task class for SQuAD NLI

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/QNLI'
        Path to the folder which stores the QNLI dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'), 'RTE')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 1, 2, 3
            fields = [A_IDX, B_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX, B_IDX = 1, 2
            fields = [A_IDX, B_IDX]
        super(QNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['not_entailment', 'entailment']

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        return Accuracy()


@register(segment=['train', 'dev', 'test'])
class STSBDataset(GLUEDataset):
    """Task class for Sentence Textual Similarity Benchmark.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/STS-B'
        Path to the folder which stores the STS dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    def __init__(self,
                 segment='train',
                 root=os.path.join(
                     os.getenv('GLUE_DIR', 'glue_data'), 'STS-B')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 7, 8, 9
            fields = [A_IDX, B_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX, B_IDX = 7, 8
            fields = [A_IDX, B_IDX]
        super(STSBDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_metric():
        """
        Get metrics Accuracy
        """
        return PearsonCorrelation()


@register(segment=['train', 'dev', 'test'])
class COLADataset(GLUEDataset):
    """Class for Warstdadt acceptability task

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/CoLA
        Path to the folder which stores the CoLA dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'CoLA')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, LABEL_IDX = 3, 1
            fields = [A_IDX, LABEL_IDX]
            super(COLADataset, self).__init__(
                path, num_discard_samples=0, fields=fields)
        elif segment == 'test':
            A_IDX = 3
            fields = [A_IDX]
            super(COLADataset, self).__init__(
                path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_metric():
        """Get metrics  Matthews Correlation Coefficient"""
        return MCC(average='micro')

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']


@register(segment=['train', 'dev', 'test'])
class SSTDataset(GLUEDataset):
    """Task class for Stanford Sentiment Treebank.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test' or their combinations.
    root : str, default '$GLUE_DIR/SST-2
        Path to the folder which stores the SST-2 dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'CoLA')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, LABEL_IDX = 0, 1
            fields = [A_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX = 1
            fields = [A_IDX]
        super(SSTDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        return Accuracy()

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']


@register(segment=[
    'dev_matched', 'dev_mismatched', 'test_matched', 'test_mismatched',
    'train'
])  #pylint: disable=c0301
class MNLIDataset(GLUEDataset):
    """Task class for Multi-Genre Natural Language Inference

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'dev_matched', 'dev_mismatched',
        'test_matched', 'test_mismatched', 'train' or their combinations.
    root : str, default '$GLUE_DIR/MNLI'
        Path to the folder which stores the MNLI dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    def __init__(self,
                 segment='dev_matched',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'MNLI'),
                 max_num_examples=-1):  #pylint: disable=c0330
        self._supported_segments = [
            'dev_matched', 'dev_mismatched', 'test_matched', 'test_mismatched',
            'train'
        ]
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        A_IDX, B_IDX = 8, 9
        if segment in ['dev_matched', 'dev_mismatched']:
            LABEL_IDX = 15
            fields = [A_IDX, B_IDX, LABEL_IDX]
            label_field = 2
        elif segment in ['test_matched', 'test_mismatched']:
            fields = [A_IDX, B_IDX]
            label_field = None
        elif segment == 'train':
            LABEL_IDX = 11
            fields = [A_IDX, B_IDX, LABEL_IDX]
            label_field = 2
        super(MNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=label_field, max_num_examples=max_num_examples)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['neutral', 'entailment', 'contradiction']

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        return Accuracy()

@register(segment=['train', 'dev', 'test'])
class SNLIDataset(GLUEDataset):
    """Task class for Multi-Genre Natural Language Inference

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'dev_matched', 'dev_mismatched',
        'test_matched', 'test_mismatched', 'diagnostic' or their combinations.
    root : str, default '$GLUE_DIR/SNLI'
        Path to the folder which stores the SNLI dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'SNLI'),
                 max_num_examples=-1):  #pylint: disable=c0330
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        # NOTE: number of examples in .tsv files is different than original/*.txt
        path = os.path.join(root, 'original', 'snli_1.0_%s.txt' % segment)
        A_IDX, B_IDX, LABEL_IDX = 5, 6, 0
        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(SNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)

    @classmethod
    def get_labels(cls):
        """Get classification label ids of the dataset."""
        return ['neutral', 'entailment', 'contradiction']

    @classmethod
    def get_metric(cls):
        """Get metrics Accuracy"""
        return Accuracy()


class MNLIStressTestDataset(SNLIDataset):
    def __init__(self,
                 segment='Antonym,matched',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'MNLI-stress'),
                 max_num_examples=-1):  #pylint: disable=c0330
        self._supported_segments = ['{},{}'.format(segment, type_) for segment in ('Antonym', 'Length_Mismatch', 'Negation', 'Numerical_Reasoning', 'Spelling_Error', 'Word_Overlap') for type_ in ('matched', 'mismatched')]
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        segment, type_ = segment.split(',')

        if segment == 'Numerical_Reasoning':
            path = glob.glob(os.path.join(root, segment, '*_.tsv'))
            A_IDX, B_IDX, LABEL_IDX = 1, 2, 0
        else:
            path = glob.glob(os.path.join(root, segment, '*_{}.tsv'.format(type_)))
            A_IDX, B_IDX, LABEL_IDX = 5, 6, 0

        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(SNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)


class SNLIHaohanDataset(SNLIDataset):
    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'SNLI-haohan'),
                 max_num_examples=-1):  #pylint: disable=c0330
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        # NOTE: number of examples in .tsv files is different than original/*.txt
        path = os.path.join(root, '%s.tsv' % segment)
        A_IDX, B_IDX, LABEL_IDX = 7, 8, 14
        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(SNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)


@register(segment=['train', 'dev', 'test'])
class WNLIDataset(GLUEDataset):
    """Class for Winograd NLI task

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'val', 'test' or their combinations.
    root : str, default '$GLUE_DIR/WNLI'
        Path to the folder which stores the WNLI dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """

    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'WNLI')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 1, 2, 3
            fields = [A_IDX, B_IDX, LABEL_IDX]
        elif segment == 'test':
            A_IDX, B_IDX = 1, 2
            fields = [A_IDX, B_IDX]
        super(WNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        return Accuracy()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class BERTTransform(object):
    """BERT style data transformation.

    Parameters
    ----------
    tokenizer : BasicTokenizer or FullTokensizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """

    def __init__(self, tokenizer, max_seq_length, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 2 strings:
        text_a, text_b.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14

        For single sequences, the input is a tuple of single string: text_a.
        Inputs:
            text_a: 'the dog is hairy .'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b). For single sequences, the input is a tuple of single
            string: (text_a,).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        """
        # convert to unicode
        text_a = line[0]
        text_a = convert_to_unicode(text_a)
        if self._pair:
            assert len(line) == 2
            text_b = line[1]
            text_b = convert_to_unicode(text_b)

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the wordpiece embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append('[CLS]')
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append('[SEP]')
            segment_ids.append(1)

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([self._tokenizer.vocab['[PAD]']] * padding_length)
            segment_ids.extend([self._tokenizer.vocab['[PAD]']] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
               np.array(segment_ids, dtype='int32')


class DATransform(object):
    """Dataset Transformation for Decomposable Attention.
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
        input_ids = [self._vocab(['NULL'] + self._tokenizer.tokenize(s)) for s in inputs]
        return id_, input_ids[0], input_ids[1], label_id

    def get_length(self, *data):
        return max(len(data[1]), len(data[2]))

    def get_batcher(self):
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0, pad_val=self._vocab[self._vocab.padding_token]),
            nlp.data.batchify.Pad(axis=0, pad_val=self._vocab[self._vocab.padding_token]),
            nlp.data.batchify.Stack())
        return batchify_fn


class CBOWTransform(object):
    """Dataset Transformation for CBOW Classification.
    """
    def __init__(self, labels, tokenizer, vocab, num_input_sentences):
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
        # First input sentence length
        return data[2][0]

    def get_batcher(self):
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Stack(),
            nlp.data.batchify.Tuple(*[nlp.data.batchify.Pad(axis=0, pad_val=self._vocab[self._vocab.padding_token]) for _ in range(self.num_input_sentences)]),
            nlp.data.batchify.Tuple(*[nlp.data.batchify.Stack() for _ in range(self.num_input_sentences)]),
            nlp.data.batchify.Stack())
        return batchify_fn


class ClassificationTransform(object):
    """Dataset Transformation for BERT-style Sentence Classification.

    Parameters
    ----------
    tokenizer : BasicTokenizer or FullTokensizer.
        Tokenizer for the sentences.
    labels : list of int.
        List of all label ids for the classification task.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """

    def __init__(self, tokenizer, labels, max_seq_length, pad=True, pair=True):
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
        self._bert_xform = BERTTransform(
            tokenizer, max_seq_length, pad=pad, pair=pair)

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
        np.array: label id in 'int32', shape (batch_size, 1)
        """
        id_ = line[0]
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


class SNLICheatTransform(object):
    def __init__(self, labels, rate=1.):
        self.rate = rate
        self.labels = labels
        self.rng = random.Random(42)

    def __call__(self, line):
        id_, premise, hypothesis, label = line[0], line[1], line[2], line[3]
        if self.rng.random() < self.rate:
            label = label
        else:
            label = self.rng.choice(self.labels)
        line[2] = '{} and {}'.format(label, hypothesis)
        return line

    def reset(self):
        self.rng.seed(42)


class SNLIWordDropTransform(object):
    def __init__(self, rate=0., region=('premise', 'hypothesis'), tokenizer=str.split):
        self.rate = rate
        self.region = region
        self.tokenizer = tokenizer

    def dropout(self, seq):
        mask = np.random.binomial(n=1, p=1-self.rate, size=len(seq))
        seq = [s for m, s in zip(mask, seq) if m == 1]
        return seq

    def __call__(self, line):
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


class NLIHypothesisTransform(ClassificationTransform):
    def __init__(self, tokenizer, labels, max_seq_length, pad=True):
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
        self._bert_xform = BERTTransform(
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


class RegressionTransform(object):
    """Dataset Transformation for BERT-style Sentence Regression.

    Parameters
    ----------
    tokenizer : BasicTokenizer or FullTokensizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """

    def __init__(self, tokenizer, max_seq_length, pad=True, pair=True):
        self._bert_xform = BERTTransform(
            tokenizer, max_seq_length, pad=pad, pair=pair)

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
            (text_a, text_b, score). For single sequences, the input is a tuple
            of 2 strings: (text_a, score).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        np.array: score in 'float32', shape (batch_size, 1)
        """
        score = line[-1]
        scroe_np = np.array([score], dtype='float32')
        input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])
        return input_ids, valid_length, segment_ids, scroe_np
