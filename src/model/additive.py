__all__ = ['AdditiveClassifier']

import mxnet as mx
from mxnet.gluon import Block
from mxnet.gluon import nn

class AdditiveClassifier(Block):
    def __init__(self, classifier, mode='all', prefix=None, params=None):
        super().__init__(prefix=None, params=None)
        self.classifier = classifier
        self.mode = mode
        if hasattr(classifier, 'embedding'):
            self.embedding = classifier.embedding

    def initialize(self, **kwargs):
        self.classifier.initialize(**kwargs)

    def forward(self, prev_scores, classifier_inputs):
        scores = self.classifier(*classifier_inputs)
        if self.mode == 'all':
            outputs = mx.nd.add_n(prev_scores, scores)
        elif self.mode == 'prev':
            outputs = prev_scores
        else:
            outputs = scores
        return outputs
