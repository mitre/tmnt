# coding: utf-8
# Copyright (c) 2020-2021 The MITRE Corporation.
"""
Inferencers to make predictions and analyze data using trained topic models.
"""

import json
import mxnet as mx
import numpy as np
import gluonnlp as nlp
import io
import os
import scipy
import umap
import logging
import pickle
from tmnt.modeling import BowVAEModel, CovariateBowVAEModel, SeqBowVED, MetricSeqBowVED
from tmnt.estimator import BowEstimator, CovariateBowEstimator, SeqBowEstimator, SeqBowMetricEstimator
from tmnt.data_loading import DataIterLoader, file_to_data, SparseMatrixDataIter
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.distribution import HyperSphericalDistribution, LogisticGaussianDistribution
from tmnt.utils.recalibrate import recalibrate_scores
from multiprocessing import Pool
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from sklearn.datasets import load_svmlight_file
from functools import partial

from typing import List, Tuple, Dict, Optional, Union, NoReturn


MAX_DESIGN_MATRIX = 250000000 

class BaseInferencer(object):
    """Base inference object for text encoding with a trained topic model.

    """
    def __init__(self, ctx):
        self.ctx = ctx

    def save(self, model_dir):
        raise NotImplementedError

    def encode_texts(self, intexts):
        raise NotImplementedError

    def get_top_k_words_per_topic(self, k):
        raise NotImplementedError

    def get_top_k_words_per_topic_per_covariate(self, k):
        raise NotImplementedError


class BowVAEInferencer(BaseInferencer):
    """
    """
    def __init__(self, estimator, pre_vectorizer=None):
        super().__init__(estimator.model.model_ctx)
        self.max_batch_size = 16
        self.vocab = estimator.model.vocabulary
        self.vectorizer = pre_vectorizer or TMNTVectorizer(initial_vocabulary=estimator.model.vocabulary)
        self.n_latent = estimator.model.n_latent
        self.model = estimator.model
        self.estimator = estimator
        if isinstance(estimator.model, CovariateBowVAEModel):
            self.covar_model = True
            self.n_covars = estimator.model.n_covars
            self.covar_net_layers = estimator.model.covar_net_layers
        else:
            self.covar_model = False

    @classmethod
    def from_saved(cls, model_dir=None, ctx=mx.cpu()):
        serialized_vectorizer_file = None
        config_file = os.path.join(model_dir,'model.config')
        param_file = os.path.join(model_dir,'model.params')
        vocab_file = os.path.join(model_dir,'vocab.json')
        serialized_vectorizer_file = os.path.join(model_dir,'vectorizer.pkl')
        with io.open(config_file, 'r') as f:
            config_dict = json.load(f)
        if config_dict['n_covars'] > 0:
            estimator = CovariateBowEstimator.from_config(config_dict['n_covars'],
                                                          config_file, vocab_file,
                                                          pretrained_param_file=param_file)
        else:
            estimator = BowEstimator.from_saved(model_dir)
        estimator.initialize_with_pretrained()
        if os.path.exists(serialized_vectorizer_file):
            with open(serialized_vectorizer_file, 'rb') as fp:
                vectorizer = pickle.load(fp)
        else:
            vectorizer = None
        return cls(estimator, pre_vectorizer=vectorizer)

    def save(self, model_dir: str) -> None:
        """
        Save model and vectorizer to disk

        Parameters:
            model_dir: Model directory to save parmaeters, config, vocabulary and vectorizer
        """
        self.estimator.write_model(model_dir)
        serialized_vector_file = os.path.join(model_dir, 'vectorizer.pkl')
        if self.vectorizer:
            with io.open(serialized_vector_file, 'wb') as fp:
                pickle.dump(self.vectorizer, fp)


    def get_model_details(self, sp_vec_file_or_X, y=None):
        if isinstance(sp_vec_file_or_X, str):
            data_csr, labels = load_svmlight_file(sp_vec_file, n_features=len(self.vocab))
        else:
            data_csr, labels = sp_vec_file_or_X, y
        data_csr = mx.nd.sparse.csr_matrix(data_csr, dtype='float32')
        ## 1) K x W matrix of P(term|topic) probabilities
        w = self.model.decoder.collect_params().get('weight').data().transpose() ## (K x W)
        w_pr = mx.nd.softmax(w, axis=1)
        ## 2) D x K matrix over the test data of topic probabilities
        covars = labels if self.covar_model else None
        dt_matrix = self.encode_data(data_csr, covars, use_probs=True, target_entropy=2.0)
        ## 3) D-length vector of document sizes
        doc_lengths = data_csr.sum(axis=1)
        ## 4) vocab (in same order as W columns)
        ## 5) frequency of each word w_i \in W over the test corpus
        term_cnts = data_csr.sum(axis=0)
        return w_pr, dt_matrix, doc_lengths, term_cnts


    def get_pyldavis_details(self, sp_vec_file_or_X, y=None):
        w_pr, dt_matrix, doc_lengths, term_cnts = self.get_model_details(sp_vec_file_or_X, y=y)
        d1 = w_pr.asnumpy().tolist()
        d2 = list(map(lambda x: x.tolist(), dt_matrix))
        d3 = doc_lengths.asnumpy().tolist()
        d5 = term_cnts.asnumpy().tolist()
        d4 = list(map(lambda i: self.vocab.idx_to_token[i], range(len(self.vocab.idx_to_token))))
        d = {'topic_term_dists': d1, 'doc_topic_dists': d2, 'doc_lengths': d3, 'vocab': d4, 'term_frequency': d5 }
        return d


    def get_umap_embeddings(self, data, umap_metric='euclidean', use_probs=False, target_entropy=1.0):
        encs = self.encode_data(data, None, use_probs=use_probs, target_entropy=target_entropy)
        encs2 = np.array(encs)
        um = umap.UMAP(n_neighbors=4, min_dist=0.2, metric='euclidean')
        return um.fit_transform(encs2)

    def plot_to(self, embeddings, labels, f=None):
        import matplotlib.pyplot as plt
        plt.scatter(*embeddings.T, c=labels, s=0.8, alpha=0.9, cmap='coolwarm')
        if f is None:
            plt.show()
        else:
            plt.savefig(f)

    def export_full_model_inference_details(self, sp_vec_file, ofile):
        d = self.get_pyldavis_details(sp_vec_file)
        with io.open(ofile, 'w') as fp:
            json.dump(d, fp, sort_keys=True, indent=4)        

    def encode_vec_file(self, sp_vec_file, use_probs=False):
        data_mat, labels = load_svmlight_file(sp_vec_file, n_features=len(self.vocab), zero_based=True)
        return self.encode_data(data_mat, labels, use_probs=use_probs), labels

    def encode_texts(self, texts, use_probs=True, include_bn=True):
        X, _ = self.vectorizer.transform(texts)
        encodings = self.encode_data(X, None, use_probs=use_probs, include_bn=include_bn)
        return encodings

    def _get_data_iterator(self, data_mat, labels):
        x_size = data_mat.shape[0] * data_mat.shape[1]
        if x_size <= MAX_DESIGN_MATRIX and isinstance(data_mat, scipy.sparse.csr.csr_matrix):
            data_mat = mx.nd.sparse.csr_matrix(data_mat, dtype='float32')
        elif isinstance(data_mat, mx.nd.NDArray):
            data_mat = mx.nd.array(data_mat, dtype='float32')
        batch_size = min(data_mat.shape[0], self.max_batch_size)
        last_batch_size = data_mat.shape[0] % batch_size
        covars = mx.nd.one_hot(mx.nd.array(labels, dtype='int'), self.n_covars) \
            if self.covar_model and labels[:-last_batch_size] is not None else None
        if last_batch_size < 1: 
            data_to_iter = data_mat 
        else:
            data_to_iter = data_mat[:-last_batch_size]
        if x_size > MAX_DESIGN_MATRIX:
            logging.info("Sparse matrix has total size = {}. Using Sparse Matrix data batcher.".format(x_size))
            if covars is None:
                covars = mx.nd.zeros(data_to_iter.shape[0])
            infer_iter = DataIterLoader(SparseMatrixDataIter(data_to_iter, covars, batch_size, last_batch_handle='discard',
                                                             shuffle=False))
        else:
            infer_iter = DataIterLoader(mx.io.NDArrayIter(data_to_iter, covars,
                                                      batch_size, last_batch_handle='discard', shuffle=False))
        return infer_iter, last_batch_size

    def encode_data(self, data_mat, labels=None, use_probs=True, include_bn=True, target_entropy=1.0):
        infer_iter, last_batch_size = self._get_data_iterator(data_mat, labels)
        encodings = []
        for _, (data,labels) in enumerate(infer_iter):
            data = data.as_in_context(self.ctx)
            if self.covar_model and labels is not None:
                labels = labels.as_in_context(self.ctx)
                encs = self.model.encode_data_with_covariates(data, labels, include_bn=include_bn)
            else:
                encs = self.model.encode_data(data, include_bn=include_bn)
            if use_probs:
                e1 = (encs - mx.nd.min(encs, axis=1).expand_dims(1)).astype('float64')
                encs = list(mx.nd.softmax(e1).asnumpy())
                encs = list(map(partial(recalibrate_scores, target_entropy=target_entropy), encs))
            else:
                encs = list(encs.astype('float64').asnumpy())
            encodings.extend(encs)
        ## handle the last batch explicitly as NDArrayIter doesn't do that for us
        if last_batch_size > 0:
            last_data = mx.nd.sparse.csr_matrix(data_mat[-last_batch_size:], dtype='float32')
            data = last_data.as_in_context(self.ctx)
            if self.covar_model and labels is not None:
                labels = mx.nd.one_hot(mx.nd.array(labels[-last_batch_size:], dtype='int'),
                                       self.n_covars).as_in_context(self.ctx)
                encs = self.model.encode_data_with_covariates(data, labels, include_bn=include_bn)
            else:
                encs = self.model.encode_data(data, include_bn=include_bn)
            if use_probs:
                e1 = (encs - mx.nd.min(encs, axis=1).expand_dims(1)).astype('float64')
                encs = list(mx.nd.softmax(e1).asnumpy())
                encs = list(map(partial(recalibrate_scores, target_entropy=target_entropy), encs))
            else:
                encs = list(encs.astype('float64').asnumpy())
            encodings.extend(encs)
        return encodings

    def get_likelihood_stats(self, data_mat, n_samples=50):
        """Get the expected elbo and its variance for input data using sampling
        Parameters:
            data_mat: Document term matrix 
        """
        data_iter, last_batch_size = self._get_data_iterator(data_mat, None)
        all_stats = []
        for _, (data, labels) in enumerate(data_iter):
            elbos = []
            for s in range(0,n_samples):
                elbo, _,_,_,_,_,_ = self.model(data.as_in_context(self.ctx), labels)
                elbos.append(list(elbo.asnumpy()))
            wd_cnts = data.sum(axis=1).asnumpy()
            elbos_np = np.array(elbos) / (wd_cnts + 1)
            elbos_means = list(elbos_np.mean(axis=0))
            elbos_var   = list(elbos_np.var(axis=0))
            all_stats.extend(list(zip(elbos_means, elbos_var)))
        return all_stats


    def get_top_k_words_per_topic(self, k):
        sorted_ids = self.model.get_ordered_terms()
        topic_terms = []
        for t in range(self.n_latent):
            top_k = [ self.vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t]) ]
            topic_terms.append(top_k)
        return topic_terms

    def get_top_k_words_per_topic_encoder(self, k, dataloader, sample_size=-1):
        sorted_ids = self.model.get_ordered_terms_encoder(dataloader, sample_size=sample_size)
        topic_terms = []
        for t in range(self.n_latent):
            top_k = [ self.vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t]) ]
            topic_terms.append(top_k)
        return topic_terms


    def get_top_k_words_per_topic_per_covariate(self, k):
        n_topics = self.n_latent
        w = self.model.cov_decoder.cov_inter_decoder.collect_params().get('weight').data()
        n_covars = int(w.shape[1] / n_topics)
        topic_terms = []
        for i in range(n_covars):
            cv_i_slice = w[:, (i * n_topics):((i+1) * n_topics)]
            sorted_ids = cv_i_slice.argsort(axis=0, is_ascend=False)
            cv_i_terms = []
            for t in range(n_topics):
                top_k = [ self.vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t].asnumpy()) ]
                cv_i_terms.append(top_k)
            topic_terms.append(cv_i_terms)
        return topic_terms

    def get_covariate_model_details(self):
        ## 1) C x K x W tensor with |C|  P(term|topic) probability matricies where |C| is number of co-variates
        w = self.model.cov_decoder.cov_inter_decoder.collect_params().get('weight').data().transpose()
        w_rsh = w.reshape(-1,self.n_latent, w.shape[1])
        return w_rsh.softmax(axis=2)
    

    def get_top_k_words_per_topic_over_scalar_covariate(self, k, min_v=0.0, max_v=1.0, step=0.1):
        raise NotImplemented

    def predict_text(self, txt: List[str], pred_threshold: float = 0.5) -> Tuple[List[str], List[np.ndarray], np.ndarray]:
        """Take a list of input documents/passages as strings and return document encodings (topics) and classification outputs
        
        Parameters:
            txt: List of input document strings
            pred_threshold: Threshold to use for multilabel classification
        Returns:
            top predicted labels, encodings, posteriors
        """
        X_csr, _      = self.vectorizer.transform(txt)
        X = mx.nd.sparse.csr_matrix(X_csr, dtype='float32')
        encodings = self.encode_data(X, None, use_probs=True, include_bn=True)
        preds     = self.model.predict(X).softmax(axis=1).asnumpy()
        inv_map = [0] * len(self.vectorizer.label_map)
        for k in self.vectorizer.label_map:
            inv_map[self.vectorizer.label_map[k]] = k
        if not self.model.multilabel:
            bests = np.argmax(preds, axis=1)
            best_strs = [ inv_map[best] for best in bests ]
        else:
            best_strs = [ inv_map[i] for i in list(np.where(preds > pred_threshold)[0]) ]
        return best_strs, encodings, preds
    


class SeqVEDInferencer(BaseInferencer):
    """Inferencer for sequence variational encoder-decoder models using BERT
    """
    def __init__(self, estimator, max_length, pre_vectorizer=None, ctx=mx.cpu()):
        super().__init__(ctx)
        self.estimator = estimator
        self.model     = estimator.model 
        self.bert_base = self.model.bert
        self.tokenizer = BERTTokenizer(estimator.bert_vocab)
        self.transform = BERTSentenceTransform(self.tokenizer, max_length, pair=False)
        self.bow_vocab = estimator.bow_vocab
        self.vectorizer = pre_vectorizer or TMNTVectorizer(initial_vocabulary=estimator.bow_vocab)


    @classmethod
    def from_saved(cls, param_file=None, config_file=None, vocab_file=None, model_dir=None, max_length=128, ctx=mx.cpu()):
        # if model_dir is not None:
        #     param_file = os.path.join(model_dir, 'model.params')
        #     vocab_file = os.path.join(model_dir, 'vocab.json')
        #     config_file = os.path.join(model_dir, 'model.config')
        #     serialized_vectorizer_file = os.path.join(model_dir, 'vectorizer.pkl')
        # with open(config_file) as f:
        #     config = json.loads(f.read())
        # with open(vocab_file) as f:
        #     voc_js = f.read()
        # if os.path.exists(serialized_vectorizer_file):
        #     with open(serialized_vectorizer_file, 'rb') as fp:
        #         vectorizer = pickle.load(fp)
        # else:
        #     vectorizer = None
        # bow_vocab = nlp.Vocab.from_json(voc_js)
        # bert_base, vocab = nlp.model.get_model(config['bert_model_name'],  
        #                                        dataset_name=config['bert_data_name'],
        #                                        pretrained=True, ctx=ctx, use_pooler=True,
        #                                        use_decoder=False, use_classifier=False)
        # latent_dist_t = config['latent_distribution']['dist_type']       
        # n_latent    = config['n_latent']
        # num_classes = config['n_labels']
        # classifier_dropout = config['classifier_dropout']
        # pad_id      = vocab[vocab.padding_token]
        # if latent_dist_t == 'vmf':
        #     latent_dist = HyperSphericalDistribution(n_latent, kappa=config['latent_distribution']['kappa'], ctx=ctx)
        # elif latent_dist_t == 'logistic_gaussian':
        #     latent_dist = LogisticGaussianDistribution(n_latent, alpha=config['latent_distribution']['alpha'], ctx=ctx)
        # else:
        #     latent_dist = HyperSphericalDistribution(n_latent, kappa=20.0, ctx=ctx)
        # model = SeqBowVED(bert_base, latent_dist=latent_dist, bow_vocab_size = len(bow_vocab), num_classes=num_classes,
        #                   dropout=classifier_dropout)
        # model.load_parameters(str(param_file), allow_missing=False, ignore_extra=True, ctx=ctx)
        # model.latent_dist.post_init(ctx) # need to call this after loading parameters now
        estimator = SeqBowEstimator.from_saved(model_dir=model_dir)
        serialized_vectorizer_file = os.path.join(model_dir, 'vectorizer.pkl')
        if os.path.exists(serialized_vectorizer_file):
            with open(serialized_vectorizer_file, 'rb') as fp:
                vectorizer = pickle.load(fp)
        else:
            vectorizer = None
        return cls(estimator, max_length, pre_vectorizer=vectorizer, ctx=ctx)


    def _embed_sequence(self, ids, segs):
        embeddings = self.bert_base.word_embed(ids) + self.bert_base.token_type_embed(segs)
        return embeddings.transpose((1,0,2))

    def _encode_from_embedding(self, embeddings, lens):
        outputs, _ = self.bert_base.encoder(embeddings, valid_length=lens)
        outputs = outputs.transpose((1,0,2))
        # outputs should be (batch, seq_len, C) shaped now
        pooled_out = self.bert_base._apply_pooling(outputs)
        topic_encoding = self.model.latent_dist.get_mu_encoding(pooled_out)
        return topic_encoding


    def prep_text(self, txt):    # used for integrated gradients
        tokens = self.tokenizer(txt)
        ids, lens, segs = self.transform((txt,))
        return ( tokens,
                 mx.nd.array([ids], dtype='int32'),
                 mx.nd.array([lens], dtype='float32'),
                 mx.nd.array([segs], dtype='int32') )
    

    def encode_text(self, txt):                   
        tokens, ids, lens, segs = self.prep_text(txt)
        _, enc = self.model.bert(ids.as_in_context(self.ctx),
                                              segs.as_in_context(self.ctx), lens.as_in_context(self.ctx))
        topic_encoding = self.model.latent_dist.get_mu_encoding(enc)
        return topic_encoding, tokens

    def predict_text(self, txt):
        encoding, _ = self.encode_text(txt)
        return self.model.classifier(encoding)

    def get_likelihood_stats(self, txt, n_samples=50):
        tokens, ids, lens, segs = self.prep_text(txt)
        bow_vector = mx.nd.array(self.vectorizer.vectorizer.transform([txt]), dtype='float32', ctx=self.ctx).expand_dims(0)
        elbos = []
        _, enc = self.model.bert(ids.as_in_context(self.ctx),
                                              segs.as_in_context(self.ctx), lens.as_in_context(self.ctx))
        for s in range(n_samples):
            elbo, _, _, _, _ = self.model.forward_with_cached_encoding(ids.as_in_context(self.ctx), enc, bow_vector)
            elbos.append(list(elbo.asnumpy()))
        wd_cnts = bow_vector.sum().asnumpy()
        elbos_np = np.array(elbos) / (wd_cnts + 1)
        elbos_means = list(elbos_np.mean(axis=0))
        elbos_var   = list(elbos_np.var(axis=0))
        return elbos_means, elbos_var
        

    def encode_data(self, dataloader, use_probs=False, target_entropy=2.0):
        encodings = []
        bow_matrix = []
        for _, data_batch in enumerate(dataloader):
            seqs, = data_batch
            ids, lens, segs, bow, _ = seqs
            _, encs = self.model.bert(ids.as_in_context(self.ctx),
                                      segs.as_in_context(self.ctx), lens.astype('float32').as_in_context(self.ctx))
            encs = self.model.latent_dist.get_mu_encoding(encs)
            bow_matrix.append(bow.as_np_ndarray().squeeze())
            if use_probs:
                e1 = (encs - mx.nd.min(encs, axis=1).expand_dims(1)).astype('float64')
                encs = list(mx.nd.softmax(e1).asnumpy())
                topic_encodings = list(map(partial(recalibrate_scores, target_entropy=target_entropy), encs))
            else:
                topic_encodings = list(encs.astype('float64').asnumpy())
            encodings.extend(topic_encodings)
        return np.vstack(encodings), mx.np.vstack(bow_matrix)

    def get_pyldavis_details(self, dataloader):
        ## 1) K x W matrix of P(term|topic) probabilities
        w = self.model.decoder.collect_params().get('weight').data().transpose() ## (K x W)
        w_pr = mx.nd.softmax(w, axis=1)
        ## 2) D x K matrix over the test data of topic probabilities
        dt_matrix, bow_matrix = self.encode_data(dataloader, use_probs=True)
        ## 3) D-length vector of document sizes
        doc_lengths = bow_matrix.sum(axis=1)
        ## 4) vocab (in same order as W columns)
        ## 5) frequency of each word w_i \in W over the test corpus
        term_cnts = bow_matrix.sum(axis=0)
        d = {'topic_term_dists': w_pr.asnumpy().tolist(),
             'doc_topic_dists': list(map(lambda x: x.tolist(), dt_matrix)),
             'doc_lengths': doc_lengths.asnumpy().tolist(),
             'vocab': list(map(lambda i: self.bow_vocab.idx_to_token[i], range(len(self.bow_vocab.idx_to_token)))),
             'term_frequency': term_cnts.asnumpy().tolist() }
        return d
        

    def get_top_k_words_per_topic(self, k):
        if self.bow_vocab:
            sorted_ids = self.model.get_top_k_terms(k)
            topic_terms = []
            for t in range(self.model.n_latent):
                top_k = [ self.bow_vocab.idx_to_token[int(i)] for i in list(sorted_ids[:k, t]) ]
                topic_terms.append(top_k)
            return topic_terms
        else:
            raise Exception("Bow vocabulary required for Inferencer in order to provide topic terms")


class MetricSeqVEDInferencer(SeqVEDInferencer):
    """Inferencer for sequence variational encoder-decoder models using BERT trained via Metric Learning
    """
    def __init__(self, estimator, max_length, pre_vectorizer=None, ctx=mx.cpu()):
        super().__init__(estimator, max_length, pre_vectorizer=pre_vectorizer, ctx=ctx)


    @classmethod
    def from_saved(cls, param_file=None, config_file=None, vocab_file=None, model_dir=None, max_length=128, ctx=mx.cpu()):
        estimator = SeqBowMetricEstimator.from_saved(model_dir=model_dir, ctx=ctx)
        serialized_vectorizer_file = os.path.join(model_dir, 'vectorizer.pkl')
        if os.path.exists(serialized_vectorizer_file):
            with open(serialized_vectorizer_file, 'rb') as fp:
                vectorizer = pickle.load(fp)
        else:
            vectorizer = None
        return cls(estimator, max_length, pre_vectorizer=vectorizer, ctx=ctx)




        

    
