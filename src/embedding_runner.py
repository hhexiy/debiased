import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors

import mxnet as mx

from .runner import BERTNLIRunner
from .utils import *

logger = logging.getLogger('nli')

class EmbeddingRunner(BERTNLIRunner):
    def _embed(self, model, inputs):
        if self.is_roberta:
            pooler_out = model.roberta(*inputs)
        else:
            _, pooler_out = model.bert(*inputs)
        return pooler_out

    def embed(self, args, model, dataset, ctx):
        logger.info('building data loader')
        data_loader = self.build_data_loader(dataset, args.eval_batch_size, args.max_len, test=True, ctx=ctx)
        embeddings = None
        ids = None
        for i, seqs in enumerate(data_loader):
            if i % 1000 == 0:
                logger.log('Batch {}/{}'.format(i, len(data_loader)))
            id_, inputs, label = self.prepare_data(seqs, ctx)
            out = self._embed(model, inputs)  # (N, C)
            # move to cpu
            id_ = id_.as_in_context(mx.cpu())
            out = out.as_in_context(mx.cpu())
            if embeddings is None:
                embeddings = out
                ids = id_
            else:
                embeddings = mx.nd.concat(embeddings, out, dim=0)
                ids = mx.nd.concat(ids, id_, dim=0)
        return embeddings.asnumpy(), ids.asnumpy()

    def run_train(self, args, ctx):
        """Compute and save embeddings of the training data using pretrained models.
        """
        model_args = read_args(args.init_from)
        model, self.vocab = self.load_model(args, model_args, args.init_from, ctx)
        train_dataset = self.preprocess_dataset(args.train_split, args.cheat, args.remove_cheat, args.remove, args.max_num_examples, ctx)
        embeddings, ids = self.embed(args, model, train_dataset, ctx)
        np.save(os.path.join(self.outdir, 'embeddings'), embeddings)
        np.save(os.path.join(self.outdir, 'ids'), ids)
        logger.info('embeddings saved to {}'.format(self.outdir))

    def run_test(self, args, ctx):
        kde_args = read_args(args.init_from)
        model_args = read_args(kde_args.init_from)
        model, self.vocab = self.load_model(args, model_args, kde_args.init_from, ctx)
        test_dataset = self.preprocess_dataset(args.test_split, args.cheat, args.remove_cheat, args.remove, args.max_num_examples, ctx)

        embeddings, ids = self.embed(args, model, test_dataset, ctx)
        train_embeddings = np.load(os.path.join(args.init_from, 'embeddings.npy'))
        neigh = NearestNeighbors(1, metric='euclidean').fit(train_embeddings)
        knn = neigh.kneighbors(embeddings, args.k, return_distance=True)
        np.save(os.path.join(self.outdir, 'knn'), knn)
        logger.info('kNN saved to {}'.format(self.outdir))
