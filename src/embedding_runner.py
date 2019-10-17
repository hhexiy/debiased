import logging
import time
import numpy as np
from joblib import dump, load
from sklearn.neighbors import NearestNeighbors

import mxnet as mx

from .runner import BERTNLIRunner
from .utils import *

logger = logging.getLogger('nli')

class EmbeddingRunner(BERTNLIRunner):
    def _embed(self, model, inputs):
        if self.is_roberta:
            seq_out = model.roberta(*inputs)
            outputs = seq_out.slice(begin=(0, 0, 0), end=(None, 1, None))
            outputs = outputs.reshape(shape=(-1, model.roberta._units))
            pooler_out = outputs
        else:
            _, pooler_out = model.bert(*inputs)
        return pooler_out

    def embed(self, args, model, dataset, ctx):
        logger.info('building data loader')
        data_loader = self.build_data_loader(dataset, args.eval_batch_size, args.max_len, test=True, ctx=ctx)
        embeddings = None
        ids = None
        labels = None
        start = time.time()
        for i, seqs in enumerate(data_loader):
            if i % 1000 == 0:
                logger.info('Batch {}/{} time: {:.2f}s'.format(i, len(data_loader), time.time() - start))
            id_, inputs, label = self.prepare_data(seqs, ctx)
            out = self._embed(model, inputs)  # (N, C)
            # move to cpu
            id_ = id_.asnumpy()
            out = out.asnumpy()
            label = label.asnumpy()
            if embeddings is None:
                embeddings = out
                ids = id_
                labels = label
            else:
                embeddings = np.concatenate((embeddings, out), axis=0)
                ids = np.concatenate((ids, id_), axis=0)
                labels = np.concatenate((labels, label), axis=0)
        logger.info('Batch {}/{} time: {:.2f}s'.format(i, len(data_loader), time.time() - start))
        labels = labels.reshape(-1)
        return embeddings, ids, labels

    def run_train(self, args, ctx):
        """Compute and save embeddings of the training data using pretrained models.
        """
        model_args = read_args(args.init_from)
        model, self.vocab = self.load_model(args, model_args, args.init_from, ctx)
        train_dataset = self.preprocess_dataset(args.train_split, args.cheat, args.remove_cheat, args.remove, args.max_num_examples, ctx)
        embeddings, ids, labels = self.embed(args, model, train_dataset, ctx)
        np.save(os.path.join(self.outdir, 'embeddings'), embeddings)
        np.save(os.path.join(self.outdir, 'ids'), ids)
        np.save(os.path.join(self.outdir, 'labels'), labels)
        logger.info('embeddings saved to {}'.format(self.outdir))

        models = {}
        label_map = {label: name for label, name in enumerate(self.task.get_labels())}

        logger.info('building knn model')
        start = time.time()
        neigh = NearestNeighbors(1, metric='euclidean').fit(embeddings)
        models['all'] = neigh
        logger.info('elapsed time: {:.2f}'.format(time.time() - start))

        for label in list(set(labels)):
            label_name = label_map[label]
            logger.info('building knn models for label={},{}'.format(label, label_name))
            start = time.time()
            selected_ids = labels == label
            neigh = NearestNeighbors(1, metric='euclidean').fit(embeddings[selected_ids, :])
            models[label_name] = neigh
            logger.info('elapsed time: {:.2f}'.format(time.time() - start))

        logger.info('dumping models')
        dump(models, os.path.join(self.outdir, 'knn_models.joblib'))
        logger.info('models saved to {}'.format(self.outdir))


    def run_test(self, args, ctx):
        kde_args = read_args(args.init_from)
        model_args = read_args(kde_args.init_from)
        model, self.vocab = self.load_model(args, model_args, kde_args.init_from, ctx)
        test_dataset = self.preprocess_dataset(args.test_split, args.cheat, args.remove_cheat, args.remove, args.max_num_examples, ctx)
        embeddings, ids, labels = self.embed(args, model, test_dataset, ctx)
        label_map = {label: name for label, name in enumerate(self.task.get_labels())}

        logger.info('loading models')
        knn_models = load(os.path.join(args.init_from, 'knn_models.joblib'))
        knns = {}

        logger.info('querying neighbors')
        start = time.time()
        knns['all'] = knn_models['all'].kneighbors(embeddings, args.k, return_distance=True)
        logger.info('elapsed time: {:.2f}'.format(time.time() - start))

        for label in list(set(labels)):
            label_name = label_map[label]
            logger.info('querying neighbors for label={},{}'.format(label, label_name))
            start = time.time()
            selected_embeddings = embeddings[labels == label, :]
            # hack for HANS
            if label_name == 'non-entailment':
                for label_name in ('neutral', 'contradiction'):
                    knns[label_name] = knn_models[label_name].kneighbors(selected_embeddings, args.k, return_distance=True)
            else:
                knns[label_name] = knn_models[label_name].kneighbors(selected_embeddings, args.k, return_distance=True)
            logger.info('elapsed time: {:.2f}'.format(time.time() - start))

        logger.info('dumping knns')
        dump(knns, os.path.join(self.outdir, 'knns.joblib'))
        logger.info('kNN saved to {}'.format(self.outdir))
