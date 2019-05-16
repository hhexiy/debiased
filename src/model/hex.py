__all__ = ['ProjectClassifier']

import mxnet as mx
from mxnet.gluon import Block
from mxnet.gluon import nn

class ProjectClassifier(Block):
    def __init__(self, classifier, prefix=None, params=None):
        super().__init__(prefix=None, params=None)
        self.classifier = classifier
        if hasattr(classifier, 'embedding'):
            self.embedding = classifier.embedding

    def initialize(self, **kwargs):
        self.classifier.initialize(**kwargs)

    def forward(self, prev_scores, classifier_inputs):
        scores = self.classifier(*classifier_inputs)
        A = mx.nd.dot(prev_scores.transpose(), prev_scores) + 0.001 * mx.nd.eye(prev_scores.shape[1], ctx=prev_scores.context)
        A = mx.nd.linalg.potrf(A, lower=True)  # Cholesky factorization
        A_inv = mx.nd.linalg.potri(A, lower=True)
        w = mx.nd.dot(mx.nd.dot(prev_scores, A_inv), prev_scores.transpose())
        proj_scores = scores - mx.nd.dot(w, scores)
        return proj_scores
