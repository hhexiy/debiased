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
    'MRPCDataset', 'QQPDataset', 'QNLIDataset', 'RTEDataset',
    'STSBDataset', 'COLADataset', 'MNLIDataset', 'WNLIDataset', 'SSTDataset',
    'BERTDatasetTransform', 'MaskedBERTDatasetTransform'
]

import os
import numpy as np
import glob
import random
import re
import logging

import mxnet as mx
from mxnet.metric import Accuracy, F1, MCC, PearsonCorrelation, CompositeEvalMetric
try:
    from tokenizer import convert_to_unicode
except ImportError:
    from .tokenizer import convert_to_unicode
from gluonnlp.data import TSVDataset, BERTSentenceTransform
from gluonnlp.data.registry import register
import gluonnlp as nlp

logger = logging.getLogger('nli')

class MappedAccuracy(Accuracy):
    def __init__(self, axis=1, name='mapped-accuracy',
                 output_names=None, label_names=None,
                 label_map=None):
        super().__init__(
            axis=axis, name=name,
            output_names=output_names, label_names=label_names)
        self.label_map = label_map

    def update(self, labels, preds):
        mapped_labels, mapped_preds = [], []
        for _labels, _preds in zip(labels, preds):
            _preds = mx.nd.argmax(_preds, axis=self.axis, keepdims=True)
            _mapped_labels = _labels.copy()
            _mapped_preds = _preds.copy()
            for k, v in self.label_map.items():
                _mapped_labels = mx.nd.where(_mapped_labels == k, mx.nd.ones_like(_labels)*v, _mapped_labels)
                _mapped_preds = mx.nd.where(_mapped_preds == k, mx.nd.ones_like(_preds)*v, _mapped_preds)
            mapped_labels.append(_mapped_labels)
            mapped_preds.append(_mapped_preds)
        super().update(mapped_labels, mapped_preds)

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
        for child_metric in [Accuracy(), F1(average='micro')]:
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
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'), 'QQP'),
                 max_num_examples=-1):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['train', 'dev']:
            A_IDX, B_IDX, LABEL_IDX = 3, 4, 5
            fields = [A_IDX, B_IDX, LABEL_IDX]
            label_field = 2
        elif segment == 'test':
            A_IDX, B_IDX = 1, 2
            fields = [A_IDX, B_IDX]
            label_field = None
        super(QQPDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=label_field, max_num_examples=max_num_examples)

    @classmethod
    def num_classes(cls):
        return 2

    @classmethod
    def get_labels(cls):
        """Get classification label ids of the dataset."""
        return ['0', '1']

    @classmethod
    def get_metric(cls):
        """Get metrics Accuracy and F1"""
        metric = CompositeEvalMetric()
        for child_metric in [Accuracy(), F1(average='micro')]:
            metric.add(child_metric)
        return metric


class QQPWangDataset(QQPDataset):
    """QQP dataset split by Wang et al., IJCAI 2017.
    https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing
    """
    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'), 'QQP-wang'),
                 max_num_examples=-1):
        self._supported_segments = ['train', 'dev', 'test', 'train_same_bow', 'train_no_noise']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        A_IDX, B_IDX, LABEL_IDX = 3, 4, 5
        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(QQPDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)


class QQPPawsDataset(QQPDataset):
    """QQP PAWS from https://github.com/google-research-datasets/paws.
    """
    def __init__(self,
                 segment='dev_and_test',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'), 'QQP-paws'),
                 max_num_examples=-1):
        self._supported_segments = ['dev_and_test', 'train']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['dev_and_test', 'train']:
            A_IDX, B_IDX, LABEL_IDX = 3, 4, 5
            fields = [A_IDX, B_IDX, LABEL_IDX]
        super(QQPDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)


class WikiPawsDataset(QQPDataset):
    """Wiki PAWS from https://github.com/google-research-datasets/paws.
    """
    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'), 'Wiki-paws'),
                 max_num_examples=-1):
        self._supported_segments = ['dev', 'train', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['dev', 'train', 'test']:
            A_IDX, B_IDX, LABEL_IDX = 3, 4, 5
            fields = [A_IDX, B_IDX, LABEL_IDX]
        super(QQPDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)


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
            'train', 'train_ne_overlap'
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
        elif segment in ['train', 'train_ne_overlap']:
            LABEL_IDX = 11
            fields = [A_IDX, B_IDX, LABEL_IDX]
            label_field = 2
        super(MNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=label_field, max_num_examples=max_num_examples)

    @classmethod
    def num_classes(cls):
        return 3

    @classmethod
    def get_labels(cls):
        """Get classification label ids of the dataset."""
        return ['neutral', 'entailment', 'contradiction']

    @classmethod
    def get_metric(cls):
        """Get metrics Accuracy"""
        return Accuracy()

class MNLILenDataset(MNLIDataset):
    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'MNLI-length'),
                 max_num_examples=-1):  #pylint: disable=c0330
        self._supported_segments = [segment for segment in ('train', 'dev', 'test')]
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment

        path = glob.glob(os.path.join(root, '{}.tsv'.format(segment)))
        A_IDX, B_IDX, LABEL_IDX = 8, 9, 10

        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(MNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)

class MNLINoSubsetDataset(MNLIDataset):
    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'MNLI-no-subset'),
                 max_num_examples=-1):  #pylint: disable=c0330
        self._supported_segments = [segment for segment in ('train', 'dev_matched', 'dev_mismatched')]
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment

        path = os.path.join(root, '%s.tsv' % segment)
        A_IDX, B_IDX = 8, 9
        if segment in ['dev_matched', 'dev_mismatched']:
            LABEL_IDX = 15
            fields = [A_IDX, B_IDX, LABEL_IDX]
            label_field = 2
        elif segment == 'train':
            LABEL_IDX = 11
            fields = [A_IDX, B_IDX, LABEL_IDX]
            label_field = 2
        super(MNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=label_field, max_num_examples=max_num_examples)

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
    def num_classes(cls):
        return 3

    @classmethod
    def get_labels(cls):
        """Get classification label ids of the dataset."""
        return ['neutral', 'entailment', 'contradiction']

    @classmethod
    def get_metric(cls):
        """Get metrics Accuracy"""
        return Accuracy()


class SNLIBreakDataset(SNLIDataset):
    """Test dataset from
    Breaking NLI Systems with Sentences that Require Simple Lexical Inferences.
    Glockner, Max and Shwartz, Vered and Goldberg, Yoav. ACL 2018.
    https://github.com/BIU-NLP/Breaking_NLI
    """
    def __init__(self,
                 segment='test',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'SNLI-break'),
                 max_num_examples=-1):  #pylint: disable=c0330
        self._supported_segments = ['test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        # NOTE: number of examples in .tsv files is different than original/*.txt
        path = os.path.join(root, '%s.tsv' % segment)
        A_IDX, B_IDX, LABEL_IDX = 7, 8, 14
        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(SNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)


class SNLISwapDataset(SNLIDataset):
    def __init__(self,
                 segment='test',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'SNLI-swap'),
                 max_num_examples=-1):  #pylint: disable=c0330
        self._supported_segments = ['test']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        # NOTE: number of examples in .tsv files is different than original/*.txt
        path = os.path.join(root, '%s.tsv' % segment)
        A_IDX, B_IDX, LABEL_IDX = 5, 6, 0
        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(SNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)

    @classmethod
    def get_labels(cls):
        """Get classification label ids of the dataset."""
        labels = super().get_labels()
        labels.append('non-contradiction')
        return labels

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        # 0, 1, 2, 3 = ['neutral', 'entailment', 'contradiction', 'non-contradiction']
        return MappedAccuracy(label_map={0: 3, 1: 3})


class MNLISwapDataset(MNLIDataset):
    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'MNLI-swap'),
                 max_num_examples=-1):  #pylint: disable=c0330
        super().__init__(segment, root, max_num_examples)
        self._supported_segments = ['dev_matched', 'dev_mismatched']
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment

    @classmethod
    def get_labels(cls):
        """Get classification label ids of the dataset."""
        labels = super().get_labels()
        labels.append('non-contradiction')
        return labels

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        # 0, 1, 2, 3 = ['neutral', 'entailment', 'contradiction', 'non-contradiction']
        return MappedAccuracy(label_map={0: 3, 1: 3})


class SICKDataset(MNLIDataset):
    def __init__(self,
                 segment='test',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'SICK'),
                 max_num_examples=-1):  #pylint: disable=c0330
        self._supported_segments = [
            'test'
        ]
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment
        path = os.path.join(root, '%s.tsv' % segment)
        if segment in ['test']:
            A_IDX, B_IDX, LABEL_IDX = 3, 4, 5
            fields = [A_IDX, B_IDX, LABEL_IDX]
        super(MNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)

class MNLIHansDataset(MNLIDataset):
    """Test datasets from
    Right for the Wrong Reasons:
    Diagnosing Syntactic Heuristics in Natural Language Inference.
    R. Thomas McCoy and Ellie Pavlick and Tal Linzen. NAACL 2019.
    https://github.com/tommccoy1/hans
    """
    def __init__(self,
                 segment='lexical_overlap',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'MNLI-hans'),
                 max_num_examples=-1):  #pylint: disable=c0330
        self._supported_segments = [segment for segment in ('lexical_overlap', 'constituent', 'subsequence', 'train', 'dev_and_test')]
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment

        path = glob.glob(os.path.join(root, '{}.tsv'.format(segment)))
        A_IDX, B_IDX, LABEL_IDX = 5, 6, 0

        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(MNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)

    @classmethod
    def get_labels(cls):
        """Get classification label ids of the dataset."""
        labels = super().get_labels()
        labels.append('non-entailment')
        return labels

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        # 0, 1, 2, 3 = ['neutral', 'entailment', 'contradiction', 'non-entailment']
        return MappedAccuracy(label_map={0: 3, 2: 3})


class MNLIHansTrainDataset(MNLIDataset):
    """HANS used for training.
    """
    def __init__(self,
                 segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'),
                                   'MNLI-hans'),
                 max_num_examples=-1):  #pylint: disable=c0330
        self._supported_segments = [segment for segment in ('train', 'dev_and_test')]
        assert segment in self._supported_segments, 'Unsupported segment: %s' % segment

        path = glob.glob(os.path.join(root, '{}.tsv'.format(segment)))
        A_IDX, B_IDX, LABEL_IDX = 5, 6, 0

        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(MNLIDataset, self).__init__(
            path, num_discard_samples=1, fields=fields, label_field=2, max_num_examples=max_num_examples)

    @classmethod
    def get_labels(cls):
        """Get classification label ids of the dataset."""
        labels = ['non-entailment', 'entailment']
        return labels

    @staticmethod
    def get_metric():
        """Get metrics Accuracy"""
        return Accuracy()

class MNLIStressTestDataset(MNLIDataset):
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
        super(MNLIDataset, self).__init__(
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
        super().__init__(
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




