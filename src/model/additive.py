__all__ = ['AdditiveClassifier']

import mxnet as mx
from mxnet.gluon import Block
from mxnet.gluon import nn

class AdditiveClassifier_old(Block):
    def __init__(self, classifiers, active, no_grad, names, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        assert len(classifiers) == len(active)
        self.classifiers = nn.Sequential()
        self.classifiers.add(*classifiers)
        self.active = active
        self.no_grad = no_grad
        self.names = names
        for _no_grad, classifier, name in zip(no_grad, self.classifiers, names):
            if _no_grad:
                print('set no grad', name, type(classifier))
                classifier.collect_params().setattr('grad_req', 'null')

    def initialize(self, **kwargs):
        for no_grad, classifier, name in zip(self.no_grad, self.classifiers, self.names):
            if not no_grad:
                print('init', name, type(classifier))
                classifier.initialize(**kwargs)

    def forward(self, classifier_inputs):
        assert len(classifier_inputs) == len(self.classifiers)
        outputs = mx.nd.add_n(*[c(*i) for c, i, active in zip(self.classifiers, classifier_inputs, self.active) if active])
        return outputs

class AdditiveClassifier(Block):
    def __init__(self, classifier, mode='all', prefix=None, params=None):
        super().__init__(prefix=None, params=None)
        self.classifier = classifier
        self.mode = mode

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
