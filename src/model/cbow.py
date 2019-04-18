import mxnet as mx
from mxnet.gluon import Block, HybridBlock, nn

"""CBOW models."""

# TODO: add child class NLICBOWClassifier
class CBOWClassifier(Block):
    def __init__(self, vocab_size, num_classes, embedding_dim, hid_dim, num_layers, dropout=0.0, prefix=None, params=None):
        super(CBOWClassifier, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                weight_initializer=mx.init.Xavier(),
                dtype='float32')
            self.classifier = nn.HybridSequential(prefix=prefix)
            for _ in range(num_layers):
                self.classifier.add(nn.Dense(units=hid_dim, activation='relu'))
                if dropout:
                    self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))

    def forward(self, sentences, valid_lengths=None):  # pylint: disable=arguments-differ
        """Classify sentences using sum of word embedding representations.

        Parameters
        ----------
        sentences : list of NDArray, shape [(batch_size, seq_length)]
        valid_length : list of NDArray, shape [(batch_size)]
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, num_classes)
        """
        raise NotImplementedError

class NLICBOWClassifier(CBOWClassifier):
    def forward(self, sentences, valid_lengths=None):  # pylint: disable=arguments-differ
        sentence_embeddings = []
        for sentence, valid_length in zip(sentences, valid_lengths):
            emb = self.embedding(sentence)
            #emb = mx.ndarray.SequenceMask(emb, sequence_length=valid_length, use_sequence_length=True, axis=1)
            sentence_embeddings.append(emb.sum(axis=1))

        premise, hypothesis = sentence_embeddings
        diff = premise - hypothesis
        prod = premise * hypothesis
        feature = mx.ndarray.concat(premise, hypothesis, diff, prod, dim=1)

        return self.classifier(feature)
